"""End-to-end demonstration of the learning loop, for STT and TTS:

  Base model -> incorrect output -> human correction -> training data ->
  fine-tuning -> updated model -> retest -> improved output

Two correction rounds are run.  Round 2 adds *new* kinds of corrections
(dates, alphanumerics, units) on top of round 1 (names, Greek symbols,
currency) and the results show that round-1 learning is retained.

The "human" is simulated with the curated ground truth of the seed set so the
run is reproducible; the Gradio app (app/app.py) performs exactly the same
steps with a real person in the loop.

Usage:  python scripts/run_pipeline.py [--stage stt|tts|all] [--langs ta hi en]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaaksetu.config import setup_console  # noqa: E402

setup_console()

from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

from vaaksetu.config import LANGS, RESULTS_DIR, ROOT, load_config  # noqa: E402
from vaaksetu.data.seed_catalog import EVAL_SETS, build_items  # noqa: E402
from vaaksetu.data.store import DataStore  # noqa: E402
from vaaksetu.eval.evaluate import (eval_stt, eval_tts, load_manifest, save_results,  # noqa: E402
                                    summarize, tts_eval_items)
from vaaksetu.audio import save_wav  # noqa: E402
from vaaksetu.learning import (free_gpu, gate, record_stt_correction, record_tts_correction,  # noqa: E402
                               train_next_stt, train_next_tts)
from vaaksetu.registry import Registry  # noqa: E402

ROUNDS = ["round1", "round2"]
CACHE = RESULTS_DIR / "cache"
CACHE.mkdir(exist_ok=True)


def cached(name: str, fn):
    """Resumability: evaluation rows are cached per model version, so an
    interrupted run restarts without redoing finished work."""
    import json
    p = CACHE / f"{name}.json"
    if p.exists():
        print(f"    (cached) {name}")
        return json.loads(p.read_text(encoding="utf-8"))
    rows = fn()
    p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def done_version(reg: Registry, key: str, rnd: str) -> str | None:
    for v, meta in reg.versions(key).items():
        if meta.get("round") == rnd:
            return v
    return None


def fmt(summary: dict, metric: str) -> str:
    cells = []
    for s in EVAL_SETS:
        if s in summary:
            v = summary[s]["all"]
            t = f" term {v['term_acc']:.0%}" if "term_acc" in v else ""
            cells.append(f"{s}: {metric} {v[metric]:.3f}{t}")
    return " | ".join(cells)


# ============================================================================ STT
def run_stt(langs: list[str]) -> None:
    from vaaksetu.stt.engine import STTEngine

    cfg = load_config("stt")
    store, reg = DataStore(), Registry()
    clips = [c for c in load_manifest() if c["set"] in EVAL_SETS and c["lang"] in langs]
    t0 = time.time()
    print(f"\n=== STT  ({len(clips)} evaluation clips, languages: {', '.join(langs)})")

    eng = STTEngine()
    rows = {"base": cached("stt_base", lambda: eval_stt(eng, "base", clips))}
    summ = {"base": summarize(rows["base"])}
    print("  base :", fmt(summ["base"], "wer"))
    rounds, prev_sets = [], []

    for rnd in ROUNDS:
        active = reg.active("stt")
        # --- human review of the active model's output on this round's recordings
        n_err = n = 0
        for r in rows[active]:
            if r["set"] != f"{rnd}_train":
                continue
            record_stt_correction(store, audio=ROOT / r["audio"], lang=r["lang"], model_output=r["hyp"],
                                  corrected=r["ref"], model_version=active, uid=f"stt:{r['item_id']}:v{r['voice']}",
                                  category=r["category"], batch=rnd, source="seed-sim")
            store.restore(f"stt:{r['item_id']}:v{r['voice']}")   # re-activate if a reset retired it
            n += 1; n_err += not r["correct"]
        print(f"\n  [{rnd}] {n} utterances reviewed, {n_err} corrected by the human (model: {active})")

        # --- fine-tune the next version on all corrections so far (+ replay)
        version = done_version(reg, "stt", rnd)
        if version:
            print(f"    (resume) {version} already trained for {rnd}")
            stats = {k: v for k, v in reg.versions("stt")[version].items() if k not in ("gate",)}
        else:
            del eng; free_gpu()
            version, stats = train_next_stt(store, reg, log=lambda m: print("   ", m))
            reg.update("stt", version, round=rnd)
            eng = STTEngine()
        rows[version] = cached(f"stt_{version}", lambda: eval_stt(eng, version, clips))
        summ[version] = summarize(rows[version])
        ok, reasons = gate(rows[version], rows[active], "wer", cfg["gate"]["max_general_wer_increase"],
                           cfg["gate"]["max_prev_corrections_wer_increase"], prev_sets,
                           n_boot=cfg["gate"].get("bootstrap_samples", 300))
        if ok:
            reg.promote("stt", version)
        reg.update("stt", version, promoted=ok, gate=reasons)
        store.update_run("stt", version, summ[version], ok)
        print(f"  {version}:", fmt(summ[version], "wer"))
        print(f"  gate -> {'PROMOTED' if ok else 'REJECTED'}; " + "; ".join(reasons))
        rounds.append({"round": rnd, "parent": active, "version": version, "reviewed": n,
                       "corrected": n_err, "promoted": ok, "gate": reasons, "train_stats": stats})
        prev_sets += [f"{rnd}_train", f"{rnd}_test"]

    save_results("stt_results", {"model": cfg["base_model"], "languages": langs, "summary": summ,
                                 "rounds": rounds, "rows": rows, "seconds": round(time.time() - t0)})
    del eng; free_gpu()


# ============================================================================ TTS
def run_tts(langs: list[str]) -> None:
    from vaaksetu.stt.engine import STTEngine
    from vaaksetu.tts.engine import TTSEngine

    cfg = load_config("tts")
    store, reg = DataStore(), Registry()
    tts, stt = TTSEngine(), STTEngine()          # base Whisper = independent round-trip judge
    out = {"languages": langs, "per_lang": {}}
    t0 = time.time()

    for lang in langs:
        key = f"tts-{lang}"
        print(f"\n=== TTS {LANGS[lang]['name']} ({LANGS[lang]['mms']})")
        audio_root = RESULTS_DIR / "audio" / "tts" / lang
        # target pronunciation = frozen base voice speaking the human's spoken form
        target = {}
        for it in tts_eval_items(lang):
            if it["set"] != "general":
                wav, sr = tts.synthesize(it["spoken"], lang, "base")
                target[it["item_id"]] = save_wav(audio_root / "target" / f"{it['item_id']}.wav", wav, sr)
        ref_cache: dict[str, str] = {}
        rows = {"base": cached(f"tts_{lang}_base",
                               lambda: eval_tts(tts, stt, lang, "base", target, ref_cache, audio_root / "base"))}
        ref_cache.update({r["item_id"]: r["ref"] for r in rows["base"] if r["set"] != "general"})
        summ = {"base": summarize(rows["base"])}
        print("  base :", fmt(summ["base"], "cer"))
        rounds, prev_sets = [], []

        for rnd in ROUNDS:
            active = reg.active(key)
            items = [i for i in build_items() if i["lang"] == lang and i["set"] == f"{rnd}_train"
                     and i["text"] != i["spoken"]]
            for it in items:
                record_tts_correction(store, tts, lang=lang, text=it["text"], spoken=it["spoken"],
                                      model_version=active, uid=f"tts:{it['item_id']}",
                                      category=it["category"], batch=rnd, source="seed-sim")
                store.restore(f"tts:{it['item_id']}")            # re-activate if a reset retired it
            print(f"\n  [{rnd}] {len(items)} mispronounced sentences corrected with a spoken form (model: {active})")
            version = done_version(reg, key, rnd)
            if version:
                print(f"    (resume) {version} already trained for {rnd}")
                stats = {k: v for k, v in reg.versions(key)[version].items() if k not in ("gate",)}
            else:
                version, stats = train_next_tts(store, tts, lang, reg, log=lambda m: print("   ", m))
                reg.update(key, version, round=rnd)
            rows[version] = cached(f"tts_{lang}_{version}", lambda: eval_tts(
                tts, stt, lang, version, target, ref_cache, audio_root / version))
            summ[version] = summarize(rows[version])
            ok, reasons = gate(rows[version], rows[active], "cer", cfg["gate"]["max_general_cer_increase"],
                               cfg["gate"]["max_prev_corrections_cer_increase"], prev_sets,
                               n_boot=cfg["gate"].get("bootstrap_samples", 300))
            if ok:
                reg.promote(key, version)
            reg.update(key, version, promoted=ok, gate=reasons)
            store.update_run(key, version, summ[version], ok)
            print(f"  {version}:", fmt(summ[version], "cer"))
            print(f"  gate -> {'PROMOTED' if ok else 'REJECTED'}; " + "; ".join(reasons))
            rounds.append({"round": rnd, "parent": active, "version": version, "corrections": len(items),
                           "promoted": ok, "gate": reasons, "train_stats": stats})
            prev_sets += [f"{rnd}_train", f"{rnd}_test"]
        out["per_lang"][lang] = {"summary": summ, "rounds": rounds, "rows": rows}
        tts.drop(lang)

    out["seconds"] = round(time.time() - t0)
    save_results("tts_results", out)
    del stt; free_gpu()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stt", "tts", "all"], default="all")
    ap.add_argument("--langs", nargs="+", default=["ta", "hi", "en"], choices=list(LANGS))
    a = ap.parse_args()
    if a.stage in ("stt", "all"):
        run_stt(a.langs)
    if a.stage in ("tts", "all"):
        run_tts(a.langs)
    print("\nDone. Build the report with:  python scripts/make_report.py")
