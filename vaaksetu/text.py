"""Text normalisation used for scoring (never used to alter model output)."""
from __future__ import annotations

import re
import unicodedata

# Punctuation that carries no lexical content.  Symbols such as ₹ % ° α / @
# are deliberately *kept* - getting them right is part of what we measure.
_STRIP = re.compile(r"[.,!?;:\"'“”‘’()\[\]{}।॥…\-–—]")
_WS = re.compile(r"\s+")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def norm_for_scoring(text: str) -> str:
    """Canonical form for WER/CER: NFKC (µ == μ), lower-case, drop punctuation,
    drop thousands separators (₹2,500 == ₹2500), collapse whitespace."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = t.lower()
    t = _STRIP.sub(" ", t.replace("/", " / "))
    return _WS.sub(" ", t).strip()


def norm_term(text: str) -> str:
    """Canonical form for exact-term matching: like scoring but spaces removed,
    so 'TN 09 AB 1234' and 'TN09AB1234' are distinguished only by content."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"(?<=\d),(?=\d)", "", t).lower()
    t = re.sub(r"[.,!?;:\"'“”‘’()\[\]{}।॥…]", "", t)
    return re.sub(r"\s+", "", t)


def term_hit(output: str, term: str) -> bool:
    """True if the target term appears verbatim (modulo case/punctuation)."""
    return norm_term(term) in norm_term(output)
