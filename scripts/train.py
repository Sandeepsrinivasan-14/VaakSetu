"""Continued training from newly collected corrections.

  python scripts/train.py status                     # data + model versions
  python scripts/train.py import corrections.jsonl   # bulk-add corrections
  python scripts/train.py stt                        # train next STT version, evaluate, gate, promote
  python scripts/train.py tts --lang ta              # same for Tamil TTS
  python scripts/train.py rollback stt v001          # point `active` back to an older version
  python scripts/train.py export                     # dump the whole dataset to data/exports/*.jsonl
  python scripts/train.py augment-numbers --n 30     # synthetic English currency pairs for number
                                                      # generalisation (see vaaksetu/tts/augment.py)

Import format (one JSON object per line):
  {"modality": "stt", "lang": "ta", "audio": "path/to/clip.wav", "text": "corrected transcript",
   "model_output": "optional original output", "category": "name"}
  {"modality": "tts", "lang": "hi", "text": "written text", "spoken": "how it is read",
   "audio": "optional path to a human recording"}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaaksetu.config import setup_console  # noqa: E402

setup_console()
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

from vaaksetu.config import LANGS  # noqa: E402
from vaaksetu.data.store import DataStore  # noqa: E402
from vaaksetu.learning import (cycle_stt, cycle_tts, record_stt_correction,  # noqa: E402
                               record_tts_correction)
from vaaksetu.registry import Registry  # noqa: E402


def cmd_status(_):
    store, reg = DataStore(), Registry()
    print("Stored samples (active):")
    for r in store.stats():
        print(f"  {r['modality']:3} {r['lang']:2} {r['role']:10} {str(r['batch']):14} n={r['n']:4}  errors={r['errors']}")
    print("\nModels:")
    for key, e in reg.data.items():
        print(f"  {key:6} active={e['active']}  versions={', '.join(e['versions']) or '-'}")


def cmd_import(a):
    store = DataStore()
    tts = None
    n = 0
    base = Path(a.file).resolve().parent
    for line in open(a.file, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        audio = (base / r["audio"]) if r.get("audio") else None
        if r["modality"] == "stt":
            record_stt_correction(store, audio=audio, lang=r["lang"], model_output=r.get("model_output", ""),
                                  corrected=r["text"], model_version=r.get("model_version", "unknown"),
                                  category=r.get("category"), batch=a.batch, source="import")
        else:
            if tts is None and audio is None:
                from vaaksetu.tts.engine import TTSEngine
                tts = TTSEngine()
            record_tts_correction(store, tts, lang=r["lang"], text=r["text"], spoken=r.get("spoken"),
                                  recording=audio, model_version=r.get("model_version", "unknown"),
                                  category=r.get("category"), batch=a.batch, source="import")
        n += 1
    print(f"imported {n} corrections into batch '{a.batch}'")


def cmd_stt(a):
    cycle_stt(DataStore(), Registry(), force=a.force)


def cmd_tts(a):
    from vaaksetu.stt.engine import STTEngine
    from vaaksetu.tts.engine import TTSEngine
    cycle_tts(DataStore(), TTSEngine(), STTEngine(), a.lang, Registry(), force=a.force)


def cmd_rollback(a):
    Registry().rollback(a.key, a.version)
    print(f"{a.key} active -> {a.version}")


def cmd_export(_):
    print(DataStore().export_jsonl())


def cmd_augment_numbers(a):
    """Generate synthetic (written, spoken) English currency pairs so TTS
    number-reading generalises past the exact amounts a human corrected -
    see vaaksetu/tts/augment.py for why only English is supported."""
    from vaaksetu.tts.augment import augment_currency_corrections_en
    from vaaksetu.tts.engine import TTSEngine

    store, tts = DataStore(), TTSEngine()
    pairs = augment_currency_corrections_en(a.n, seed=a.seed)
    for text, spoken in pairs:
        record_tts_correction(store, tts, lang="en", text=text, spoken=spoken, model_version="base",
                              category="currency", batch="augment-numbers", source="synthetic-numbers")
    print(f"added {len(pairs)} synthetic English currency corrections (batch 'augment-numbers')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("import"); p.add_argument("file"); p.add_argument("--batch", default="import")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("stt"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_stt)
    p = sub.add_parser("tts"); p.add_argument("--lang", required=True, choices=list(LANGS))
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_tts)
    p = sub.add_parser("rollback"); p.add_argument("key"); p.add_argument("version")
    p.set_defaults(fn=cmd_rollback)
    sub.add_parser("export").set_defaults(fn=cmd_export)
    p = sub.add_parser("augment-numbers"); p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_augment_numbers)
    a = ap.parse_args()
    a.fn(a)
