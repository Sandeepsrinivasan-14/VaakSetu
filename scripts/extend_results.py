"""One-off: evaluate the newest promoted versions (STT v005, TTS-en v003) on the
same eval clips/sentences run_pipeline.py used, and append their real rows/summary
into results/stt_results.json and results/tts_results.json so make_report.py's
before/after report reflects the currently active registry.

Usage:  python scripts/extend_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaaksetu.config import setup_console  # noqa: E402

setup_console()
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

from vaaksetu.config import RESULTS_DIR  # noqa: E402
from vaaksetu.data.seed_catalog import EVAL_SETS  # noqa: E402
from vaaksetu.eval.evaluate import eval_stt, eval_tts, load_manifest, save_results, summarize, tts_eval_items  # noqa: E402
from vaaksetu.audio import save_wav  # noqa: E402
from vaaksetu.learning import free_gpu  # noqa: E402


def extend_stt(version: str) -> None:
    from vaaksetu.stt.engine import STTEngine

    path = RESULTS_DIR / "stt_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if version in data["summary"]:
        print(f"stt_results.json already has {version}, skipping")
        return
    clips = [c for c in load_manifest() if c["set"] in EVAL_SETS]
    eng = STTEngine()
    rows = eval_stt(eng, version, clips)
    del eng
    free_gpu()
    data["rows"][version] = rows
    data["summary"][version] = summarize(rows)
    save_results("stt_results", data)
    print(f"stt_results.json: added {version}")


def extend_tts(lang: str, version: str) -> None:
    from vaaksetu.stt.engine import STTEngine
    from vaaksetu.tts.engine import TTSEngine

    path = RESULTS_DIR / "tts_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["per_lang"][lang]
    if version in entry["summary"]:
        print(f"tts_results.json[{lang}] already has {version}, skipping")
        return
    tts, stt = TTSEngine(), STTEngine()
    audio_root = RESULTS_DIR / "audio" / "tts" / lang
    target = {}
    for it in tts_eval_items(lang):
        if it["set"] != "general":
            wav, sr = tts.synthesize(it["spoken"], lang, "base")
            target[it["item_id"]] = save_wav(audio_root / "target" / f"{it['item_id']}.wav", wav, sr)
    ref_cache: dict[str, str] = {}
    rows = eval_tts(tts, stt, lang, version, target, ref_cache, audio_root / version)
    entry["rows"][version] = rows
    entry["summary"][version] = summarize(rows)
    save_results("tts_results", data)
    print(f"tts_results.json[{lang}]: added {version}")


if __name__ == "__main__":
    extend_stt("v005")
    extend_tts("en", "v003")
    print("Done. Rebuild the report with:  python scripts/make_report.py")
