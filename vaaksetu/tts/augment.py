"""Optional TTS training-data augmentation for number generalisation.

The documented weakness: TTS learns to read the *exact* digit sequences it
was corrected on, but a new, unseen amount ("₹4,250") does not reliably
generalise from a few corrected examples ("₹500", "₹2,500", ...).

This module does not change inference and adds no runtime rules - it only
generates additional (written text, target audio) TRAINING pairs, the same
way the existing self-distilled anchors work: a correctly computed spoken
form is read aloud by the frozen base voice, and the model later learns
written -> speech end-to-end from that pair, same as for a human correction.

Only English is implemented.  Hindi and Tamil cardinal numbers have
irregular, language-specific counting words (and Tamil currency phrasing
differs again), so a hand-written converter risks silently teaching the
model *wrong* pronunciations - worse than not augmenting at all.  Doing
those languages properly needs either a native-speaker-verified converter or
a well-supported library; neither is wired in here.
"""
from __future__ import annotations

import random

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, rem = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[rem]}" if rem else "")


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    hundreds, rem = divmod(n, 100)
    s = f"{_ONES[hundreds]} hundred"
    return s + (f" and {_below_100(rem)}" if rem else "")


def number_to_words_en(n: int) -> str:
    """English cardinal, matching the seed catalogue's style
    ("seven hundred and fifty"). Supports 0-9,999,999: comfortably above any
    realistic currency amount in this project."""
    if n == 0:
        return "zero"
    if n < 0 or n > 9_999_999:
        raise ValueError(f"number_to_words_en supports 0-9,999,999, got {n}")
    parts = []
    for scale, name in ((1_000_000, "million"), (1_000, "thousand")):
        if n >= scale:
            count, n = divmod(n, scale)
            parts.append(f"{_below_1000(count)} {name}")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def currency_amount_to_words_en(amount: int) -> str:
    return f"{number_to_words_en(amount)} rupees"


def augment_currency_corrections_en(n: int, *, lo: int = 100, hi: int = 99_999, step: int = 50,
                                    seed: int = 0) -> list[tuple[str, str]]:
    """`n` synthetic (written, spoken) English currency pairs, e.g.
    ("₹4,250.", "four thousand two hundred and fifty rupees"). Amounts are
    multiples of `step` since that is how the seed data's currency amounts
    look (real invoices round to 50/100), not because the model needs it."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    seen: set[int] = set()
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        amount = rng.randrange(lo, hi, step)
        if amount in seen:
            continue
        seen.add(amount)
        out.append((f"₹{amount:,}.", currency_amount_to_words_en(amount)))
    return out
