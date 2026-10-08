"""Evaluation of STT and TTS versions on the seed evaluation sets."""
from __future__ import annotations

import json
from pathlib import Path

from ..audio import load_audio, save_wav
from ..config import RESULTS_DIR, ROOT, SEED_DIR, load_config
from ..data.seed_catalog import EVAL_SETS, build_items
from ..text import norm_for_scoring
from ..tts.vocab import dropped_fraction
from .metrics import summarize


def load_manifest() -> list[dict]:
    with open(SEED_DIR / "manifest.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ------------------------------------------------------------------------ STT
def eval_stt(engine, version: str, clips: list[dict] | None = None, batch_size: int = 16,
             log=print) -> list[dict]:
    """Transcribe every eval clip with `version`; returns per-clip rows."""
    clips = clips if clips is not None else [c for c in load_manifest() if c["set"] in EVAL_SETS]
    engine.use(version)
    rows = []
    for lang in sorted({c["lang"] for c in clips}):
        lc = [c for c in clips if c["lang"] == lang]
        hyps = engine.transcribe([load_audio(ROOT / c["audio"]) for c in lc], lang, batch_size)
        for c, h in zip(lc, hyps):
            rows.append({"item_id": c["item_id"], "voice": c["voice_idx"], "set": c["set"],
                         "lang": lang, "category": c["category"], "term": c["term"],
                         "ref": c["text"], "hyp": h, "audio": c["audio"],
                         "correct": norm_for_scoring(h) == norm_for_scoring(c["text"])})
        log(f"    {version} {lang}: {len(lc)} clips")
    return rows


# ------------------------------------------------------------------------ TTS
def tts_eval_items(lang: str) -> list[dict]:
    """TTS is evaluated on sentences that need special reading (text != spoken)
    plus the general sentences (regression)."""
    return [i for i in build_items() if i["lang"] == lang and i["set"] in EVAL_SETS
            and (i["set"] == "general" or i["text"] != i["spoken"])]


def eval_tts(tts, stt, lang: str, version: str, target_audio: dict[str, Path],
             ref_cache: dict[str, str], save_dir: Path | None = None, log=print) -> list[dict]:
    """Round-trip intelligibility: synthesize the *written* text with `version`,
    transcribe with the base Whisper model and compare to

      * correction sets: the transcript of the target pronunciation
        (base voice speaking the human-provided spoken form) - so ASR
        formatting preferences cancel out;
      * general set: the sentence itself.
    """
    items = tts_eval_items(lang)
    wavs, rows = [], []
    for it in items:
        wav, sr = tts.synthesize(it["text"], lang, version)
        wavs.append(wav)
        if save_dir is not None:
            p = save_dir / f"{it['item_id']}.wav"
            save_wav(p, wav, sr)
    # base-Whisper transcripts of target audio (computed once, cached)
    todo = [it for it in items if it["set"] != "general" and it["item_id"] not in ref_cache]
    if todo:
        refs = stt.transcribe([load_audio(target_audio[it["item_id"]]) for it in todo], lang)
        ref_cache.update({it["item_id"]: r for it, r in zip(todo, refs)})
    hyps = stt.transcribe(wavs, lang)
    tok = tts.get(lang, version)[1]
    skip = load_config("tts")["vocab_skip_chars"]
    for it, h in zip(items, hyps):
        ref = it["text"] if it["set"] == "general" else ref_cache[it["item_id"]]
        rows.append({"item_id": it["item_id"], "set": it["set"], "lang": lang,
                     "category": it["category"], "term": None, "text": it["text"],
                     "spoken": it["spoken"], "ref": ref, "hyp": h,
                     "dropped_chars": round(dropped_fraction(tok, it["text"], skip), 4),
                     "audio": str((save_dir / f"{it['item_id']}.wav").relative_to(ROOT)).replace("\\", "/")
                     if save_dir is not None else None})
    log(f"    tts-{lang} {version}: {len(items)} sentences")
    return rows


def save_results(name: str, payload: dict) -> Path:
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


__all__ = ["eval_stt", "eval_tts", "load_manifest", "save_results", "summarize", "tts_eval_items"]
