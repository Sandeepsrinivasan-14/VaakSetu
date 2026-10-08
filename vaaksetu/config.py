"""Central paths, language table and config loading for VaakSetu."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AUDIO_STORE = DATA_DIR / "audio"          # content-addressed audio (never overwritten)
DB_PATH = DATA_DIR / "vaaksetu.db"        # append-only correction / training database
SNAPSHOT_DIR = DATA_DIR / "snapshots"     # frozen JSONL manifest per training run
EXPORT_DIR = DATA_DIR / "exports"
SEED_DIR = DATA_DIR / "seed"              # generated seed recordings used by the demo pipeline
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"

SAMPLE_RATE = 16_000

# One row per supported language.  `whisper` is the Whisper language code,
# `mms` the per-language MMS-TTS checkpoint.
LANGS: dict[str, dict] = {
    "en": {"name": "English (Indian)", "whisper": "en", "mms": "facebook/mms-tts-eng",
           "voices": ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]},
    "hi": {"name": "Hindi", "whisper": "hi", "mms": "facebook/mms-tts-hin",
           "voices": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]},
    "ta": {"name": "Tamil", "whisper": "ta", "mms": "facebook/mms-tts-tam",
           "voices": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"]},
}


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_console() -> None:
    """Windows consoles default to cp1252; Tamil/Hindi output needs UTF-8."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Once every base model is cached, run fully offline (no flaky HEAD requests).
    # (Checked on disk *before* huggingface_hub is imported, since it reads the flag at import.)
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    repos = [load_config("stt")["base_model"]] + [l["mms"] for l in LANGS.values()]
    def cached(repo: str) -> bool:   # config AND weights present (a config alone is not enough)
        snaps = hub / f"models--{repo.replace('/', '--')}" / "snapshots"
        return any(any(d.glob(w) for w in ("model.safetensors", "pytorch_model.bin"))
                   for d in snaps.glob("*") if (d / "config.json").exists())
    if all(cached(r) for r in repos):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


for _d in (DATA_DIR, AUDIO_STORE, SNAPSHOT_DIR, EXPORT_DIR, SEED_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
