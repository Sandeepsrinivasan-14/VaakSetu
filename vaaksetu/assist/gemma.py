"""Gemma correction assistant (local, via Ollama).

Gemma never changes model output on its own and is never used at inference
time.  It only *suggests* a correction that the human accepts or edits before
it is stored as training data:

  * STT: suggest a corrected transcript following the house conventions;
  * TTS: suggest the spoken form (how a written sentence should be read).

Default model: `gemma3n:e2b` (runs on a 4 GB GPU / CPU).  Override with the
VAAKSETU_GEMMA_MODEL env var, e.g. a Gemma 4 tag once pulled into Ollama.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from ..config import LANGS

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("VAAKSETU_GEMMA_MODEL", "gemma3n:e2b")
# first call includes loading the model (slow on CPU / when the GPU is busy training)
TIMEOUT = float(os.environ.get("VAAKSETU_GEMMA_TIMEOUT", "600"))
# Whisper + VITS need the (small) GPU; Gemma runs on CPU by default so the two never fight
# for VRAM (that contention made the STT tab hang).  VAAKSETU_GEMMA_GPU_LAYERS=999 -> GPU.
GPU_LAYERS = int(os.environ.get("VAAKSETU_GEMMA_GPU_LAYERS", "0"))
KEEP_ALIVE = os.environ.get("VAAKSETU_GEMMA_KEEP_ALIVE", "60m")   # stay loaded between demo clicks

CONVENTIONS = """Target writing conventions:
- names: correct spelling in the sentence's own script
- Greek letters used as scientific terms: Unicode symbol (alpha -> α, beta -> β, gamma -> γ)
- money: ₹ followed by digits with thousands separators (₹2,500); decimal paise are kept
  as given (₹5,850.50)
- dates: DD/MM/YYYY (15/08/2024) or DD-MM-YYYY (24-07-2026) - use whichever separator the
  source already uses, don't rewrite one into the other
- times: 24-hour HH:MM (14:30, 18:00)
- vehicle numbers / IDs / codes: uppercase; no spaces, or hyphens exactly where the source
  has them (TN09AB1234 or TN88-AB-1234) - don't add or remove hyphens that weren't asked for
- measurements: symbol form (37°C, 45%, 5 µg)
- technical/domain acronyms transliterated into the sentence's script (எல்பிஜி/LPG,
  ஜிபிஎஸ்/GPS) and code-mixed loanwords (டாஷ்போர்டு/dashboard, ஃப்ளீட் மேனேஜர்/fleet
  manager): keep the existing transliteration, don't translate or romanize it"""


_resolved: str | None = None


def _resolve_model() -> str | None:
    """Name of the installed Ollama model to use, or None if none matches.

    Prefers the configured tag; otherwise accepts any installed model of the same
    family (e.g. a `llamacpp:<sha>` import of gemma3n) so a renamed install still works.
    """
    global _resolved
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            models = json.loads(r.read())["models"]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return _resolved
    family = MODEL.split(":")[0]
    names = [m["name"] for m in models]
    if MODEL in names:
        _resolved = MODEL
    else:
        _resolved = next((m["name"] for m in models
                          if m["name"].startswith(family)
                          or family in (m.get("details", {}).get("families") or [])
                          or m.get("details", {}).get("family") == family), None)
    return _resolved


def _generate(prompt: str, timeout: float | None = None) -> str:
    model = _resolved or _resolve_model() or MODEL
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "keep_alive": KEEP_ALIVE,
                       "options": {"temperature": 0.1, "num_predict": 200, "num_ctx": 2048,
                                   "num_gpu": GPU_LAYERS}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
        text = json.loads(r.read().decode("utf-8"))["response"].strip()
    # models sometimes wrap the answer in quotes / code fences
    return text.strip("`\"' \n").splitlines()[0].strip() if text else ""


def available() -> bool:
    return _resolve_model() is not None


def suggest_transcript(asr_text: str, lang: str, hint: str = "") -> str:
    """Suggest a corrected transcript for an ASR output."""
    prompt = (f"You fix speech-recognition transcripts in {LANGS[lang]['name']}.\n{CONVENTIONS}\n"
              f"{('Context from the user: ' + hint) if hint else ''}\n"
              f"Rewrite the transcript applying the conventions and fixing obvious recognition "
              f"errors. Keep the language and script. Output only the corrected transcript.\n\n"
              f"Transcript: {asr_text}\nCorrected:")
    return _generate(prompt)


_SCRIPT = {"en": ("lowercase English words", r"[a-z]"),
           "hi": ("Devanagari script", r"[ऀ-ॿ]"),
           "ta": ("Tamil script", r"[஀-௿]")}
_EXAMPLE = {"en": ("Pay ₹2,500 on 15/08/2024.",
                   "pay two thousand five hundred rupees on fifteenth of august two thousand twenty four"),
            "hi": ("तापमान 37°C है।", "तापमान सैंतीस डिग्री सेल्सियस है"),
            "ta": ("டிக்கெட் விலை ₹500.", "டிக்கெட் விலை ஐநூறு ரூபாய்")}


def script_ok(spoken: str, lang: str) -> bool:
    """True if the spoken form is written only in the language's script (no digits / other scripts)."""
    letters = [ch for ch in spoken if ch.isalpha()]
    pat = re.compile(_SCRIPT[lang][1])
    return bool(letters) and all(pat.match(ch) for ch in letters) and not any(ch.isdigit() for ch in spoken)


def suggest_spoken_form(text: str, lang: str) -> str:
    """Suggest how a written sentence should be read aloud, fully in words.

    Retries once with a stricter prompt if the answer leaves the target script
    (small models like to romanize Hindi/Tamil)."""
    script = _SCRIPT[lang][0]
    ex_in, ex_out = _EXAMPLE[lang]
    out = ""
    for attempt in range(2):
        strict = ("" if attempt == 0 else
                  f"IMPORTANT: use ONLY {script}. No Latin letters, no digits, no symbols.\n")
        prompt = (f"Write exactly how this {LANGS[lang]['name']} sentence is read aloud by a native "
                  f"speaker. Expand every number, date, currency symbol, unit, Greek letter and "
                  f"abbreviation into words, using only {script} (letters of codes are spelled out "
                  f"one by one). Output only the spoken sentence.\n{strict}\n"
                  f"Sentence: {ex_in}\nSpoken: {ex_out}\n\nSentence: {text}\nSpoken:")
        out = _generate(prompt)
        if script_ok(out, lang):
            break
    return out


def warmup() -> bool:
    """Load the model into memory now (first call otherwise costs ~30 s in front of the audience)."""
    if not available():
        return False
    try:
        _generate("Say OK.", timeout=300)
        return True
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return False
