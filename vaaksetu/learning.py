"""The correction -> training-data -> fine-tune -> version loop, shared by the
demo pipeline, the CLI (`scripts/train.py`) and the Gradio app."""
from __future__ import annotations

import gc
import json
from collections import defaultdict

import numpy as np
import torch

from .config import LANGS, ROOT, load_config
from .data.store import DataStore
from .eval.evaluate import load_manifest
from .eval.metrics import cer, wer
from .registry import Registry, model_dir
from .text import nfc, norm_for_scoring


# ============================================================ STT corrections
def record_stt_correction(store: DataStore, *, audio, lang: str, model_output: str,
                          corrected: str, model_version: str, uid: str | None = None,
                          category: str | None = None, batch: str = "ui",
                          source: str = "human-ui") -> str:
    """Store one human-reviewed transcript (audio + corrected text)."""
    sha = store.put_audio(audio)
    uid = uid or f"stt:{lang}:{sha[:16]}"
    store.add_sample(uid=uid, modality="stt", lang=lang, category=category, text=nfc(corrected).strip(),
                     audio_sha=sha, model_output=model_output, model_version=model_version,
                     was_error=norm_for_scoring(model_output) != norm_for_scoring(corrected),
                     role="correction", batch=batch, source=source)
    return uid


def ensure_stt_anchors(store: DataStore) -> int:
    """Replay data: general sentences with correct transcripts (from the seed set)."""
    n = 0
    for c in load_manifest():
        if c["set"] == "anchor":
            sha = store.put_audio(ROOT / c["audio"])
            n += store.add_sample(uid=f"stt-anchor:{c['item_id']}:v{c['voice_idx']}", modality="stt",
                                  lang=c["lang"], category="general", text=c["text"], audio_sha=sha,
                                  was_error=False, role="anchor", batch="anchor", source="seed-anchor")
    return n


def train_next_stt(store: DataStore, reg: Registry | None = None, log=print) -> tuple[str, dict]:
    """Train the next STT adapter version on *all* active corrections (+ replay).
    Registers it (not promoted - call gate/promote after evaluation)."""
    from .stt.train import train_stt

    reg = reg or Registry()
    cfg = load_config("stt")
    ensure_stt_anchors(store)
    parent = reg.active("stt")
    version = reg.next_version("stt")
    out = model_dir("stt", version)
    corr = store.samples("stt", role="correction")
    anch = store.samples("stt", role="anchor")
    if not corr:
        raise RuntimeError("no STT corrections stored yet")
    snap, snap_sha = store.snapshot(f"stt_{version}", corr + anch)
    to_row = lambda r: {"audio_path": str(store.audio_path(r["audio_sha"])), "text": r["text"],
                        "lang": r["lang"], "batch": r.get("batch")}
    log(f"[stt] training {version} (parent={parent}) on {len(corr)} corrections + {len(anch)} anchors")
    stats = train_stt([to_row(r) for r in corr], [to_row(r) for r in anch], out,
                      parent_adapter=None if parent == "base" else model_dir("stt", parent) / "adapter",
                      cfg=cfg, log=log)
    meta = {"parent": parent, "snapshot": str(snap.relative_to(ROOT)), "snapshot_sha256": snap_sha,
            "languages": sorted({r["lang"] for r in corr}), **stats}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    reg.register("stt", version, meta)
    store.record_run(model_key="stt", version=version, parent_version=parent, base_model=cfg["base_model"],
                     snapshot_file=snap, snapshot_sha=snap_sha, n_samples=len(corr) + len(anch), params=stats)
    return version, stats


# ============================================================ TTS corrections
def record_tts_correction(store: DataStore, tts, *, lang: str, text: str, spoken: str | None = None,
                          recording=None, model_version: str, uid: str | None = None,
                          category: str | None = None, batch: str = "ui",
                          source: str = "human-ui") -> str:
    """Store a pronunciation correction.

    The *target* speech is either the human's recording, or - preferred, as it
    keeps the voice identical - the frozen **base** model speaking the
    human-provided spoken form.  The model later learns text -> target."""
    if recording is None:
        if not spoken:
            raise ValueError("provide a spoken form or a recording")
        recording, _ = tts.synthesize(spoken, lang, "base")
    sha = store.put_audio(recording)
    uid = uid or f"tts:{lang}:{sha[:16]}"
    store.add_sample(uid=uid, modality="tts", lang=lang, category=category, text=nfc(text).strip(),
                     spoken=spoken, audio_sha=sha, model_version=model_version, was_error=True,
                     role="correction", batch=batch, source=source)
    return uid


def ensure_tts_anchors(store: DataStore, tts, lang: str) -> int:
    """Self-distilled replay data: the base model reading general sentences."""
    from .data.seed_catalog import build_items
    n = 0
    existing = {r["uid"] for r in store.samples("tts", lang=lang, role="anchor", include_retired=True)}
    for it in build_items():
        if it["lang"] == lang and it["set"] == "anchor":
            uid = f"tts-anchor:{it['item_id']}"
            if uid not in existing:
                wav, _ = tts.synthesize(it["text"], lang, "base")
                n += store.add_sample(uid=uid, modality="tts", lang=lang, category="general",
                                      text=it["text"], spoken=it["text"], audio_sha=store.put_audio(wav),
                                      was_error=False, role="anchor", batch="anchor", source="self-distill")
    return n


def train_next_tts(store: DataStore, tts, lang: str, reg: Registry | None = None,
                   log=print) -> tuple[str, dict]:
    from .tts.train import train_tts

    reg = reg or Registry()
    key = f"tts-{lang}"
    ensure_tts_anchors(store, tts, lang)
    parent = reg.active(key)
    version = reg.next_version(key)
    out = model_dir(key, version)
    corr = store.samples("tts", lang=lang, role="correction")
    anch = store.samples("tts", lang=lang, role="anchor")
    if not corr:
        raise RuntimeError(f"no TTS corrections stored for {lang}")
    snap, snap_sha = store.snapshot(f"{key}_{version}", corr + anch)
    to_row = lambda r: {"text": r["text"], "audio_path": str(store.audio_path(r["audio_sha"]))}
    log(f"[{key}] training {version} (parent={parent}) on {len(corr)} corrections + {len(anch)} anchors")
    stats = train_tts(lang, [to_row(r) for r in corr], [to_row(r) for r in anch], out,
                      parent_version=parent, log=log)
    meta = {"parent": parent, "snapshot": str(snap.relative_to(ROOT)), "snapshot_sha256": snap_sha, **stats}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    reg.register(key, version, meta)
    store.record_run(model_key=key, version=version, parent_version=parent, base_model=LANGS[lang]["mms"],
                     snapshot_file=snap, snapshot_sha=snap_sha, n_samples=len(corr) + len(anch), params=stats)
    return version, stats


# ===================================================================== gating
def _bootstrap_regression(old_rows: list[dict], new_rows: list[dict], metric_fn, n_boot: int = 300,
                          seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap over matched eval items (same clips scored by both
    versions).  Returns (point_diff, ci_low): point_diff is new - old on the
    full set; ci_low is the 5th percentile of that difference across
    resamples.  If ci_low is still above a tolerance, the regression is very
    unlikely to be an artefact of which eval items happened to be hard."""
    old_by_id = {r["item_id"]: r for r in old_rows}
    new_by_id = {r["item_id"]: r for r in new_rows}
    ids = sorted(set(old_by_id) & set(new_by_id))
    if not ids:
        return 0.0, 0.0
    old_ref = [old_by_id[i]["ref"] for i in ids]; old_hyp = [old_by_id[i]["hyp"] for i in ids]
    new_ref = [new_by_id[i]["ref"] for i in ids]; new_hyp = [new_by_id[i]["hyp"] for i in ids]
    point = metric_fn(new_ref, new_hyp) - metric_fn(old_ref, old_hyp)

    rng = np.random.default_rng(seed)
    n = len(ids)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = (metric_fn([new_ref[i] for i in idx], [new_hyp[i] for i in idx])
                    - metric_fn([old_ref[i] for i in idx], [old_hyp[i] for i in idx]))
    return float(point), float(np.quantile(diffs, 0.05))


def gate(new_rows: list[dict], old_rows: list[dict], metric: str, general_tol: float, prev_tol: float,
         prev_sets: list[str], n_boot: int = 300) -> tuple[bool, list[str]]:
    """Promotion gate on raw per-item eval rows (ref/hyp/set), not just the
    summarized numbers.  A set only fails the gate when its regression both
    (a) exceeds the flat tolerance on the observed sample, and (b) survives a
    paired bootstrap: the 5th-percentile lower bound of the resampled
    regression is *still* above tolerance.  This keeps a lucky/unlucky draw
    of eval clips from flipping the promotion decision, while a real, large
    regression (like STT v004's Tamil WER jump) still fails easily."""
    metric_fn = wer if metric == "wer" else cer
    by_set_new: dict[str, list[dict]] = defaultdict(list)
    by_set_old: dict[str, list[dict]] = defaultdict(list)
    for r in new_rows:
        by_set_new[r["set"]].append(r)
    for r in old_rows:
        by_set_old[r["set"]].append(r)

    reasons, ok = [], True

    def check(label: str, tol: float) -> None:
        nonlocal ok
        rows_new, rows_old = by_set_new.get(label, []), by_set_old.get(label, [])
        if not rows_new or not rows_old:
            return
        point, ci_low = _bootstrap_regression(rows_old, rows_new, metric_fn, n_boot=n_boot)
        if point > tol and ci_low > tol:
            ok = False
        reasons.append(f"{label} {metric} {point:+.3f} (tolerance +{tol}, bootstrap 5th pct {ci_low:+.3f})")

    check("general", general_tol)
    for s in prev_sets:
        check(s, prev_tol)
    return ok, reasons


def cycle_stt(store: DataStore, reg: Registry | None = None, force: bool = False, log=print) -> dict:
    """Train next STT version -> evaluate vs. current active -> gate -> promote."""
    from .data.seed_catalog import EVAL_SETS
    from .eval.evaluate import eval_stt, summarize
    from .stt.engine import STTEngine

    reg, cfg = reg or Registry(), load_config("stt")
    parent = reg.active("stt")
    version, stats = train_next_stt(store, reg, log=log)
    free_gpu()
    clips = [c for c in load_manifest() if c["set"] in EVAL_SETS]
    eng = STTEngine()
    old_rows, new_rows = eval_stt(eng, parent, clips, log=log), eval_stt(eng, version, clips, log=log)
    old, new = summarize(old_rows), summarize(new_rows)
    del eng; free_gpu()
    ok, reasons = gate(new_rows, old_rows, "wer", cfg["gate"]["max_general_wer_increase"],
                       cfg["gate"]["max_prev_corrections_wer_increase"], [s for s in EVAL_SETS if s != "general"],
                       n_boot=cfg["gate"].get("bootstrap_samples", 300))
    return _finish(store, reg, "stt", version, parent, new, old, ok or force, reasons, log)


def cycle_tts(store: DataStore, tts, stt, lang: str, reg: Registry | None = None,
              force: bool = False, log=print) -> dict:
    """Train next TTS version for `lang` -> round-trip evaluate -> gate -> promote."""
    import tempfile
    from pathlib import Path

    from .audio import save_wav
    from .data.seed_catalog import EVAL_SETS
    from .eval.evaluate import eval_tts, summarize, tts_eval_items

    reg, cfg = reg or Registry(), load_config("tts")
    key = f"tts-{lang}"
    parent = reg.active(key)
    version, _ = train_next_tts(store, tts, lang, reg, log=log)
    tmp = Path(tempfile.mkdtemp(prefix="vaaksetu_"))
    target = {}
    for it in tts_eval_items(lang):
        if it["set"] != "general":
            w, sr = tts.synthesize(it["spoken"], lang, "base")
            target[it["item_id"]] = save_wav(tmp / f"{it['item_id']}.wav", w, sr)
    cache: dict = {}
    old_rows, new_rows = (eval_tts(tts, stt, lang, parent, target, cache, log=log),
                          eval_tts(tts, stt, lang, version, target, cache, log=log))
    old, new = summarize(old_rows), summarize(new_rows)
    ok, reasons = gate(new_rows, old_rows, "cer", cfg["gate"]["max_general_cer_increase"],
                       cfg["gate"]["max_prev_corrections_cer_increase"], [s for s in EVAL_SETS if s != "general"],
                       n_boot=cfg["gate"].get("bootstrap_samples", 300))
    return _finish(store, reg, key, version, parent, new, old, ok or force, reasons, log)


def _finish(store, reg, key, version, parent, new, old, ok, reasons, log) -> dict:
    if ok:
        reg.promote(key, version)
    reg.update(key, version, promoted=ok, gate=reasons)
    store.update_run(key, version, new, ok)
    for r in reasons:
        log("  " + r)
    log(f"{key} {version}: {'PROMOTED to active' if ok else 'kept as candidate (not promoted)'}")
    return {"key": key, "version": version, "parent": parent, "promoted": ok,
            "gate": reasons, "summary": new, "parent_summary": old}


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
