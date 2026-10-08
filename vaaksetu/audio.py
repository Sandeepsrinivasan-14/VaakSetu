"""Audio I/O and augmentation helpers (librosa/soundfile only - no torchaudio)."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE


def load_audio(path_or_bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file (wav/mp3/flac/...) as mono float32 at `sr`."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        path_or_bytes = io.BytesIO(path_or_bytes)
    y, _ = librosa.load(path_or_bytes, sr=sr, mono=True)
    return y.astype(np.float32)


def to_mono_16k(y: np.ndarray, orig_sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1) if y.shape[1] < y.shape[0] else y.mean(axis=0)
    if np.abs(y).max(initial=0) > 1.5:  # int16 from gradio
        y = y / 32768.0
    if orig_sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
    return y.astype(np.float32)


def wav_bytes(y: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def save_wav(path: Path, y: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(y, -1, 1), sr, subtype="PCM_16")
    return path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation so a handful of corrections generalise:
    speed perturbation, random gain and additive noise."""
    if rng.random() < 0.7:
        rate = float(rng.uniform(0.9, 1.1))
        y = librosa.effects.time_stretch(y, rate=rate)
    y = y * float(rng.uniform(0.6, 1.2))
    if rng.random() < 0.7:
        snr_db = float(rng.uniform(15, 30))
        p = float(np.mean(y ** 2)) + 1e-9
        y = y + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), size=y.shape).astype(np.float32)
    return np.clip(y, -1, 1).astype(np.float32)
