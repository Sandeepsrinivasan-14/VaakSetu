"""Mandatory 'fleet alert' Tamil sample - an additional requirement added after the
original exercise brief. This single real-world sentence stresses:

  - a hyphenated date (24-07-2026, DD-MM-YYYY) in addition to the house DD/MM/YYYY form
  - 24-hour clock time (14:30, 18:00) - a brand new category, not in round1/round2
  - a hyphenated vehicle registration (TN88-AB-1234) in addition to the unhyphenated form
  - decimal-paise currency (Rs.5,850.50) in addition to whole-rupee amounts
  - code-mixed English-in-Tamil technical/domain terms (LPG, GPS, drop status,
    dashboard, fleet manager)

Per the requirement this exact sentence must be used for training, testing AND the final
demonstration - see vaaksetu/data/seed_catalog.py TA["round3"]["fleet_alert"]. It is trained
the same way as round1/round2: cumulative data, warm start from the current active version,
replay/anchors, evaluated + gated against every earlier round and general speech so the
project's own continued-learning guarantee covers it too.

  python scripts/round3_fleet_alert.py --before-only   # fast: show base vs. current active only
  python scripts/round3_fleet_alert.py                 # full: record corrections, train, show after
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaaksetu.config import setup_console  # noqa: E402

setup_console()
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

from vaaksetu.audio import load_audio  # noqa: E402
from vaaksetu.config import ROOT  # noqa: E402
from vaaksetu.data.seed_catalog import build_items  # noqa: E402
from vaaksetu.data.store import DataStore  # noqa: E402
from vaaksetu.eval.evaluate import load_manifest  # noqa: E402
from vaaksetu.learning import (cycle_stt, cycle_tts, free_gpu, record_stt_correction,  # noqa: E402
                               record_tts_correction)
from vaaksetu.registry import Registry  # noqa: E402
from vaaksetu.text import norm_for_scoring  # noqa: E402

LANG = "ta"


def stt_round3_clips() -> list[dict]:
    return [c for c in load_manifest() if c["lang"] == LANG and c["set"] in ("round3_train", "round3_test")]


def tts_round3_item() -> dict:
    return next(i for i in build_items() if i.get("round") == "round3" and i["set"] == "round3_train")


def show_stt(eng, version: str, clips: list[dict], label: str) -> None:
    v = eng.use(version)
    print(f"\n--- STT {label} (version={v}) ---")
    for c in clips:
        hyp = eng.transcribe([load_audio(ROOT / c["audio"])], c["lang"])[0]
        ok = norm_for_scoring(hyp) == norm_for_scoring(c["text"])
        print(f"  [{c['set']} voice{c['voice_idx']}] {'EXACT MATCH' if ok else 'differs from target'}")
        print(f"    target: {c['text']}")
        print(f"    output: {hyp}")


def show_tts(tts, stt_judge, item: dict, version: str, label: str) -> None:
    wav, sr = tts.synthesize(item["text"], LANG, version)
    hyp = stt_judge.transcribe([wav], LANG)[0]
    print(f"\n--- TTS {label} (version={version}) ---")
    print(f"  written text  : {item['text']}")
    print(f"  heard back as : {hyp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-only", action="store_true",
                    help="only show base vs. current active output; don't record corrections or train")
    a = ap.parse_args()

    store, reg = DataStore(), Registry()

    # =============================================================================== STT
    from vaaksetu.stt.engine import STTEngine

    clips = stt_round3_clips()
    train_clips = [c for c in clips if c["set"] == "round3_train"]
    active_stt = reg.active("stt")

    eng = STTEngine()
    show_stt(eng, "base", clips, "BASE MODEL (before any training)")
    show_stt(eng, active_stt, clips, f"CURRENT ACTIVE, before round3 ({active_stt})")

    if not a.before_only:
        for c in train_clips:
            hyp = eng.transcribe([load_audio(ROOT / c["audio"])], c["lang"])[0]
            uid = f"stt:{c['item_id']}:v{c['voice_idx']}"
            record_stt_correction(store, audio=ROOT / c["audio"], lang=c["lang"], model_output=hyp,
                                  corrected=c["text"], model_version=active_stt, uid=uid,
                                  category=c["category"], batch="round3", source="seed-sim")
            store.restore(uid)
        print(f"\n[stt] recorded {len(train_clips)} round3 correction(s); training next version "
              f"(warm start from {active_stt}, cumulative data: round1 + round2 + round3 + imports)...")
        del eng
        free_gpu()
        result = cycle_stt(store, reg, log=print)
        print(f"\n[stt] {result['key']} {result['version']}: "
              f"{'PROMOTED to active' if result['promoted'] else 'NOT PROMOTED (kept as candidate)'}")

        eng = STTEngine()
        show_stt(eng, result["version"], clips, f"AFTER round3 TRAINING ({result['version']})")
    del eng
    free_gpu()

    # =============================================================================== TTS
    from vaaksetu.stt.engine import STTEngine as _JudgeSTT
    from vaaksetu.tts.engine import TTSEngine

    tts = TTSEngine()
    stt_judge = _JudgeSTT()
    item = tts_round3_item()
    active_tts = reg.active(f"tts-{LANG}")

    show_tts(tts, stt_judge, item, "base", "BASE MODEL (before any training)")
    show_tts(tts, stt_judge, item, active_tts, f"CURRENT ACTIVE, before round3 ({active_tts})")

    if not a.before_only:
        uid = f"tts:{item['item_id']}"
        record_tts_correction(store, tts, lang=LANG, text=item["text"], spoken=item["spoken"],
                              model_version=active_tts, uid=uid, category=item["category"],
                              batch="round3", source="seed-sim")
        print(f"\n[tts-{LANG}] recorded round3 correction; training next version "
              f"(warm start from {active_tts}, cumulative data: round1 + round2 + round3)...")
        result = cycle_tts(store, tts, stt_judge, LANG, reg, log=print)
        print(f"\n[tts-{LANG}] {result['key']} {result['version']}: "
              f"{'PROMOTED to active' if result['promoted'] else 'NOT PROMOTED (kept as candidate)'}")
        show_tts(tts, stt_judge, item, result["version"], f"AFTER round3 TRAINING ({result['version']})")


if __name__ == "__main__":
    main()
