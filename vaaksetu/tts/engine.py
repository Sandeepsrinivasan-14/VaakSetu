"""MMS-TTS (VITS) inference for base and fine-tuned versions."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import VitsModel

from ..config import LANGS, load_config
from ..registry import Registry, model_dir
from ..text import nfc
from . import tokenization_vaaksetu
from .tokenization_vaaksetu import VaakSetuVitsTokenizer


def load_tts(lang: str, version: str = "base", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src = LANGS[lang]["mms"] if version == "base" else str(model_dir(f"tts-{lang}", version))
    tok = VaakSetuVitsTokenizer.from_pretrained(src)
    model = VitsModel.from_pretrained(src).to(device).eval()
    return model, tok


def save_tts(model, tok, out_dir: Path) -> None:
    """Save weights + tokenizer + the tokenizer class, loadable anywhere with
    VitsModel.from_pretrained / AutoTokenizer.from_pretrained(trust_remote_code=True)."""
    out_dir = Path(out_dir)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    shutil.copy(tokenization_vaaksetu.__file__, out_dir / "tokenization_vaaksetu.py")
    cfg_path = out_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["char_slots"] = tok.char_slots
    cfg["tokenizer_class"] = "VaakSetuVitsTokenizer"
    cfg["auto_map"] = {"AutoTokenizer": ["tokenization_vaaksetu.VaakSetuVitsTokenizer", None]}
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


class TTSEngine:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config("tts")["inference"]
        self._cache: dict[tuple[str, str], tuple] = {}

    def resolve(self, lang: str, version: str | None) -> str:
        if version in (None, "active"):
            return Registry().active(f"tts-{lang}")
        return version

    def get(self, lang: str, version: str | None = "active"):
        version = self.resolve(lang, version)
        key = (lang, version)
        if key not in self._cache:
            self._cache[key] = load_tts(lang, version, self.device)
        return self._cache[key]

    def drop(self, lang: str | None = None) -> None:
        for k in [k for k in self._cache if lang is None or k[0] == lang]:
            del self._cache[k]
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def synthesize(self, text: str, lang: str, version: str | None = "active",
                   seed: int | None = None) -> tuple[np.ndarray, int]:
        model, tok = self.get(lang, version)
        model.noise_scale = self.cfg["noise_scale"]
        model.noise_scale_duration = self.cfg["noise_scale_duration"]
        inputs = tok(nfc(text), return_tensors="pt").to(self.device)
        sr = model.config.sampling_rate
        if inputs["input_ids"].shape[-1] <= 1:          # nothing speakable survived tokenisation
            return np.zeros(int(0.3 * sr), dtype=np.float32), sr
        torch.manual_seed(self.cfg["seed"] if seed is None else seed)
        wav = model(**inputs).waveform[0].float().cpu().numpy()
        return wav, sr
