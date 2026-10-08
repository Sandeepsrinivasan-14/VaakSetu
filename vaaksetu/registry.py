"""Model registry: versioned model/adapter directories + an `active` pointer.

models/registry.json
models/stt/v001/{adapter/, processor/, metadata.json}
models/tts/ta/v001/{model files, tokenizer files, metadata.json}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import MODELS_DIR

REGISTRY_FILE = MODELS_DIR / "registry.json"


def model_dir(model_key: str, version: str) -> Path:
    if model_key == "stt":
        return MODELS_DIR / "stt" / version
    modality, lang = model_key.split("-", 1)
    return MODELS_DIR / modality / lang / version


class Registry:
    def __init__(self, path: Path = REGISTRY_FILE):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _entry(self, key: str) -> dict:
        return self.data.setdefault(key, {"active": "base", "versions": {}})

    def versions(self, key: str) -> dict:
        return self._entry(key)["versions"]

    def active(self, key: str) -> str:
        return self._entry(key)["active"]

    def next_version(self, key: str) -> str:
        return f"v{len(self.versions(key)) + 1:03d}"

    def register(self, key: str, version: str, meta: dict) -> None:
        meta = {**meta, "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.versions(key)[version] = meta
        self._save()

    def update(self, key: str, version: str, **fields) -> None:
        self.versions(key)[version].update(fields)
        self._save()

    def promote(self, key: str, version: str) -> None:
        if version != "base" and version not in self.versions(key):
            raise KeyError(f"{key}:{version} not registered")
        self._entry(key)["active"] = version
        self._save()

    rollback = promote
