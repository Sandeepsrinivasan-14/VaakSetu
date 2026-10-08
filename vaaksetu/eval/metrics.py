"""Scoring helpers: WER / CER / exact-term accuracy, grouped by set, language, category."""
from __future__ import annotations

from collections import defaultdict

import jiwer

from ..text import norm_for_scoring, term_hit


def _safe(refs: list[str], hyps: list[str], fn) -> float:
    pairs = [(r, h if h else "∅") for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0
    return float(fn([p[0] for p in pairs], [p[1] for p in pairs]))


def wer(refs, hyps) -> float:
    return _safe([norm_for_scoring(r) for r in refs], [norm_for_scoring(h) for h in hyps], jiwer.wer)


def cer(refs, hyps) -> float:
    return _safe([norm_for_scoring(r) for r in refs], [norm_for_scoring(h) for h in hyps], jiwer.cer)


def score_rows(rows: list[dict]) -> dict:
    """rows need: ref, hyp, (optional) term."""
    refs, hyps = [r["ref"] for r in rows], [r["hyp"] for r in rows]
    out = {"n": len(rows), "wer": round(wer(refs, hyps), 4), "cer": round(cer(refs, hyps), 4)}
    termed = [r for r in rows if r.get("term")]
    if termed:
        out["term_acc"] = round(sum(term_hit(r["hyp"], r["term"]) for r in termed) / len(termed), 4)
    return out


def summarize(rows: list[dict], keys=("set", "lang")) -> dict:
    """Nested summary: {set: {lang: metrics, 'all': metrics, 'by_category': {...}}}."""
    out: dict = {}
    by_set = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    for s, srows in by_set.items():
        entry = {"all": score_rows(srows)}
        by_lang = defaultdict(list)
        for r in srows:
            by_lang[r["lang"]].append(r)
        for lang, lrows in by_lang.items():
            entry[lang] = score_rows(lrows)
            cats = defaultdict(list)
            for r in lrows:
                cats[r["category"]].append(r)
            entry[lang]["by_category"] = {c: score_rows(cr) for c, cr in cats.items()}
        out[s] = entry
    return out
