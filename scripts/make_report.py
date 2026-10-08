"""Build the before/after report from results/*.json.

Outputs
  docs/RESULTS.md       - tables + example corrections (markdown)
  results/report.html   - same, with audio players for the TTS/STT examples
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaaksetu.config import LANGS, RESULTS_DIR, ROOT, setup_console  # noqa: E402
from vaaksetu.data.seed_catalog import EVAL_SETS  # noqa: E402
from vaaksetu.text import norm_for_scoring, term_hit  # noqa: E402

setup_console()
SET_LABEL = {"round1_train": "Round-1 corrected utterances", "round1_test": "Round-1 held-out",
             "round2_train": "Round-2 corrected utterances", "round2_test": "Round-2 held-out",
             "general": "General speech (regression)"}
CAT_ORDER = ["name", "greek", "currency", "date", "alnum", "unit", "general"]


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def load(name):
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------ tables
def version_table(summary: dict, metric: str, lang: str = "all", with_term=False) -> list[list[str]]:
    versions = list(summary)
    head = ["Evaluation set"] + [f"{v} {metric.upper()}" for v in versions]
    if with_term:
        head += [f"{v} exact-term" for v in versions]
    rows = [head]
    for s in EVAL_SETS:
        if s not in summary[versions[0]]:
            continue
        get = lambda v: summary[v][s].get(lang) if lang != "all" else summary[v][s]["all"]
        r = [SET_LABEL[s]] + [pct(get(v)[metric]) if get(v) else "–" for v in versions]
        if with_term:
            r += [pct(get(v).get("term_acc")) if get(v) else "–" for v in versions]
        rows.append(r)
    return rows


def category_table(summary: dict, lang: str) -> list[list[str]]:
    versions = list(summary)
    rows = [["Category (held-out + corrected)"] + [f"{v} exact-term" for v in versions]]
    for cat in CAT_ORDER:
        vals = []
        for v in versions:
            hits = n = 0
            for s in ("round1_train", "round1_test", "round2_train", "round2_test"):
                c = summary[v].get(s, {}).get(lang, {}).get("by_category", {}).get(cat)
                if c and "term_acc" in c:
                    hits += c["term_acc"] * c["n"]; n += c["n"]
            vals.append(pct(hits / n) if n else None)
        if any(vals):
            rows.append([cat] + [x or "–" for x in vals])
    return rows


def md_table(rows):
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def html_table(rows):
    h = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in rows[0]) + "</tr>"
    b = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f"<table>{h}{b}</table>"


# ---------------------------------------------------------------- examples
def stt_examples(res: dict, per_lang: int = 6) -> list[dict]:
    rows = res["rows"]
    versions = list(rows)
    final = versions[-1]
    idx = {v: {(r["item_id"], r["voice"]): r for r in rows[v]} for v in versions}
    out = []
    for lang in res["languages"]:
        picked, seen_cat = [], {}
        for r in rows["base"]:
            if r["lang"] != lang or not r["set"].endswith("_test") or r["voice"] != 1:
                continue
            f = idx[final][(r["item_id"], r["voice"])]
            if r["term"] and not term_hit(r["hyp"], r["term"]) and term_hit(f["hyp"], r["term"]):
                if seen_cat.get(r["category"], 0) < 2:
                    seen_cat[r["category"]] = seen_cat.get(r["category"], 0) + 1
                    picked.append({"lang": lang, "set": r["set"], "category": r["category"], "ref": r["ref"],
                                   "audio": r["audio"],
                                   **{v: idx[v][(r["item_id"], r["voice"])]["hyp"] for v in versions}})
            if len(picked) >= per_lang:
                break
        out += picked
    return out


def tts_examples(res: dict, per_lang: int = 5) -> list[dict]:
    out = []
    for lang, d in res["per_lang"].items():
        versions = list(d["rows"])
        final = versions[-1]
        by_id = {v: {r["item_id"]: r for r in d["rows"][v]} for v in versions}
        cand = []
        for r in d["rows"]["base"]:
            if r["set"] == "general":
                continue
            f = by_id[final][r["item_id"]]
            gain = _cer(r["ref"], r["hyp"]) - _cer(f["ref"], f["hyp"])
            cand.append((gain, r["set"].endswith("_test"), r["item_id"]))
        cand.sort(key=lambda x: (-x[1], -x[0]))
        chosen = [c for c in cand if c[1]][:3] + [c for c in cand if not c[1]][:per_lang - 3]
        for _, _, iid in chosen:
            base = by_id["base"][iid]
            out.append({"lang": lang, "set": base["set"], "text": base["text"], "spoken": base["spoken"],
                        "target_asr": base["ref"],
                        "audio_target": f"audio/tts/{lang}/target/{iid}.wav",
                        **{v: by_id[v][iid]["hyp"] for v in versions},
                        **{f"audio_{v}": (by_id[v][iid]["audio"] or "").replace("results/", "", 1)
                           for v in versions}})
    return out


def _cer(ref, hyp):
    import jiwer
    r, h = norm_for_scoring(ref), norm_for_scoring(hyp) or "∅"
    return jiwer.cer(r, h) if r else 0.0


# -------------------------------------------------------------------- main
def main():
    stt, tts = load("stt_results"), load("tts_results")
    md = ["# VaakSetu – Before / After Results", "",
          "_Generated by `scripts/make_report.py` from `results/*.json` (produced by `scripts/run_pipeline.py`)._", ""]
    hp = []

    if stt:
        md += ["## Speech-to-Text (Whisper large-v3-turbo + LoRA)", "",
               f"Evaluation: {sum(len(v) for v in [stt['rows']['base']])} clips, two voices per language. "
               "Lower WER is better; *exact-term* = the target term (name, ₹ amount, date, code, symbol) "
               "appears exactly as required.", ""]
        for rd in stt["rounds"]:
            md.append(f"- **{rd['round']}**: {rd['reviewed']} utterances reviewed, {rd['corrected']} corrected "
                      f"→ trained **{rd['version']}** (parent {rd['parent']}, "
                      f"{rd['train_stats']['train_seconds']:.0f}s) → "
                      f"{'promoted' if rd['promoted'] else 'NOT promoted'}")
        md.append("")
        t = version_table(stt["summary"], "wer", with_term=True)
        md += ["### All languages", "", md_table(t), ""]
        hp += ["<h2>Speech-to-Text</h2>", html_table(t)]
        for lang in stt["languages"]:
            t = version_table(stt["summary"], "wer", lang, with_term=True)
            c = category_table(stt["summary"], lang)
            md += [f"### {LANGS[lang]['name']}", "", md_table(t), "", md_table(c), ""]
            hp += [f"<h3>{LANGS[lang]['name']}</h3>", html_table(t), html_table(c)]
        ex = stt_examples(stt)
        versions = list(stt["rows"])
        md += ["### Examples (held-out sentences, unseen voice)", ""]
        hp += ["<h3>Examples (held-out sentences, unseen voice)</h3>"]
        for e in ex:
            md += [f"- **{LANGS[e['lang']]['name']} / {e['category']}** — target: `{e['ref']}`"]
            md += [f"  - {v}: `{e[v]}`" for v in versions]
            hp.append(f"<div class='ex'><b>{LANGS[e['lang']]['name']} / {e['category']}</b> "
                      f"<audio controls src='../{e['audio']}'></audio><br>target: <code>{html.escape(e['ref'])}</code>"
                      + "".join(f"<br>{v}: <code>{html.escape(e[v])}</code>" for v in versions) + "</div>")
        md.append("")

    if tts:
        md += ["## Text-to-Speech (MMS-TTS / VITS, partial fine-tune)", "",
               "Round-trip intelligibility: the TTS output is transcribed by the *base* Whisper model and "
               "compared with the transcript of the target pronunciation (general set: with the sentence "
               "itself). Lower CER is better.", ""]
        hp += ["<h2>Text-to-Speech</h2>"]
        for lang, d in tts["per_lang"].items():
            t = version_table(d["summary"], "cer", "all")
            vs = list(d["rows"])
            drop = [["Characters silently dropped by the tokenizer (correction sets)"] +
                    [pct(sum(r["dropped_chars"] for r in d["rows"][v] if r["set"] != "general") /
                         max(1, sum(1 for r in d["rows"][v] if r["set"] != "general"))) for v in vs]]
            md += [f"### {LANGS[lang]['name']}", ""]
            for rd in d["rounds"]:
                ts = rd["train_stats"]
                md.append(f"- **{rd['round']}**: {rd['corrections']} corrections → **{rd['version']}** "
                          f"(learned chars: {' '.join(ts.get('learned_chars', [])) or '–'}; "
                          f"{ts['train_seconds']:.0f}s) → {'promoted' if rd['promoted'] else 'NOT promoted'}")
            md += ["", md_table(t), "", md_table([["", *vs]] + drop), ""]
            hp += [f"<h3>{LANGS[lang]['name']}</h3>", html_table(t), html_table([["", *vs]] + drop)]
        ex = tts_examples(tts)
        md += ["### Examples", ""]
        hp += ["<h3>Examples</h3>"]
        for e in ex:
            vs = list(tts["per_lang"][e["lang"]]["rows"])
            md += [f"- **{LANGS[e['lang']]['name']}** ({e['set']}) text: `{e['text']}` · should sound like: "
                   f"`{e['spoken']}`"]
            md += [f"  - {v} heard as: `{e[v]}`" for v in vs]
            hp.append(f"<div class='ex'><b>{LANGS[e['lang']]['name']}</b> ({e['set']})<br>text: "
                      f"<code>{html.escape(e['text'])}</code><br>should sound like: {html.escape(e['spoken'])} "
                      f"<audio controls src='{e['audio_target']}'></audio>"
                      + "".join(f"<br>{v}: <audio controls src='{e['audio_' + v]}'></audio> heard as "
                                f"<code>{html.escape(e[v])}</code>" for v in vs) + "</div>")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    style = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;"
             "background:#fff;color:#1d1d1f}table{border-collapse:collapse;margin:12px 0;font-size:14px}"
             "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}td:first-child,th:first-child"
             "{text-align:left}.ex{border-left:3px solid #5b7cfa;padding:6px 12px;margin:10px 0}"
             "audio{height:28px;vertical-align:middle}code{background:#f3f3f3;padding:1px 4px}")
    (RESULTS_DIR / "report.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>VaakSetu Results</title><style>{style}</style>"
        f"<h1>VaakSetu – Before / After</h1>" + "\n".join(hp), encoding="utf-8")
    print("wrote docs/RESULTS.md and results/report.html")


if __name__ == "__main__":
    main()
