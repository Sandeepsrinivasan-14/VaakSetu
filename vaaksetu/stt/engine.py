"""Whisper inference with hot-swappable LoRA adapter versions.

One copy of the base model stays in memory; each trained version is a small
LoRA adapter loaded by name, so comparing base / v001 / v002 is cheap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ..config import LANGS, SAMPLE_RATE, load_config
from ..registry import Registry, model_dir


def pick_device_dtype(device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def allow_symbols(gen_cfg, tokenizer, symbols: str) -> int:
    """Whisper's default generation config *suppresses* 'non-speech' symbol
    tokens (/, @, :, +, #, ...), so e.g. '15/08/2024' can never be emitted no
    matter what the model has learned.  Our target conventions need some of
    them, so those tokens are un-suppressed (same setting for base and every
    version - this is decoding configuration, not a text rule)."""
    if not gen_cfg.suppress_tokens:
        return 0
    keep = [t for t in gen_cfg.suppress_tokens
            if not any(s in tokenizer.decode([t]) for s in symbols)]
    removed = len(gen_cfg.suppress_tokens) - len(keep)
    gen_cfg.suppress_tokens = keep
    return removed


class STTEngine:
    def __init__(self, base_model: str | None = None, device: str | None = None):
        cfg = load_config("stt")
        self.base_model = base_model or cfg["base_model"]
        self.max_new_tokens = cfg["eval"]["max_new_tokens"]
        self.device, self.dtype = pick_device_dtype(device)
        self.processor = WhisperProcessor.from_pretrained(self.base_model)
        base = WhisperForConditionalGeneration.from_pretrained(self.base_model, dtype=self.dtype)
        base.generation_config.forced_decoder_ids = None
        allow_symbols(base.generation_config, self.processor.tokenizer, cfg["eval"]["allow_symbols"])
        self.model = base.to(self.device).eval()
        self.loaded: set[str] = set()
        self.current = "base"

    # ---------------------------------------------------------------- versions
    def use(self, version: str | None) -> str:
        """Switch to `version` ('base', 'active', 'v001', ...). Returns resolved name."""
        if version in (None, "active"):
            version = Registry().active("stt")
        if version != "base" and version not in self.loaded:
            adapter = model_dir("stt", version) / "adapter"
            if not adapter.exists():
                raise FileNotFoundError(adapter)
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, str(adapter), adapter_name=version)
            else:
                self.model.load_adapter(str(adapter), adapter_name=version)
            self.model.to(self.device).eval()
            self.loaded.add(version)
        if version != "base":
            self.model.set_adapter(version)
        self.current = version
        return version

    # --------------------------------------------------------------- inference
    @torch.inference_mode()
    def transcribe(self, audios: list[np.ndarray], lang: str, batch_size: int = 16) -> list[str]:
        out: list[str] = []
        for i in range(0, len(audios), batch_size):
            chunk = audios[i:i + batch_size]
            feats = self.processor.feature_extractor(
                chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            feats = feats.to(self.device, self.dtype)
            kwargs = dict(input_features=feats, language=LANGS[lang]["whisper"], task="transcribe",
                          max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False)
            if self.current == "base" and isinstance(self.model, PeftModel):
                with self.model.disable_adapter():
                    ids = self.model.generate(**kwargs)
            else:
                ids = self.model.generate(**kwargs)
            out.extend(t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True))
        return out
