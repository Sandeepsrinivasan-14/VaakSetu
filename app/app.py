"""VaakSetu web app - the human-in-the-loop correction UI.

  python app/app.py        ->  http://127.0.0.1:7860

Tabs
  0. About            : what the project is, how it learns, active versions, results
  1. Speech -> Text   : record/upload, transcribe, correct (optionally with a Gemma suggestion), save
  2. Text -> Speech   : synthesize, give the spoken form or a recording, save
  3. Train & Versions : dataset stats, train next version, gate/promote, rollback
  4. Compare          : same input through two versions side by side
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaaksetu.config import setup_console  # noqa: E402

setup_console()
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

import gradio as gr  # noqa: E402

from vaaksetu.assist import gemma  # noqa: E402
from vaaksetu.audio import to_mono_16k  # noqa: E402
from vaaksetu.config import LANGS  # noqa: E402
from vaaksetu.data.store import DataStore  # noqa: E402
from vaaksetu.learning import (cycle_stt, cycle_tts, free_gpu, record_stt_correction,  # noqa: E402
                               record_tts_correction)
from vaaksetu.registry import Registry  # noqa: E402

def _unknown_version_error(e: Exception, key: str, version: str) -> "gr.Error":
    known = ", ".join(["base"] + list(Registry().versions(key)))
    return gr.Error(f"Can't load {key} version '{version}' ({e}). Known versions: {known}")


LANG_CHOICES = [(v["name"], k) for k, v in LANGS.items()]
CATEGORIES = ["name", "number", "currency", "date", "alnum", "greek", "unit", "symbol", "general", "other"]
store = DataStore()
_lock = threading.Lock()
_engines: dict = {}


def stt_engine():
    if "stt" not in _engines:
        from vaaksetu.stt.engine import STTEngine
        _engines["stt"] = STTEngine()
    return _engines["stt"]


def tts_engine():
    if "tts" not in _engines:
        from vaaksetu.tts.engine import TTSEngine
        _engines["tts"] = TTSEngine()
    return _engines["tts"]


def versions(key: str) -> list[str]:
    reg = Registry()
    return ["active", "base"] + list(reg.versions(key))


# ---------------------------------------------------------------- STT tab
CHUNK_S, MAX_AUDIO_S = 28, 300      # Whisper sees 30 s windows; longer clips are split, not truncated


def transcribe_long(eng, wav, lang) -> str:
    if len(wav) < 1600:                                  # <0.1 s: nothing to transcribe
        raise gr.Error("Audio is too short / empty.")
    if float(abs(wav).max()) < 0.005:                    # Whisper hallucinates words on silence
        raise gr.Error("No speech detected - the audio is silent. Check the microphone / file.")
    if len(wav) > MAX_AUDIO_S * 16000:
        raise gr.Error(f"Audio is longer than {MAX_AUDIO_S // 60} minutes - please use a shorter clip.")
    step = CHUNK_S * 16000
    chunks = [wav[i:i + step] for i in range(0, len(wav), step)]
    chunks = [c for c in chunks if len(c) >= 1600]
    return " ".join(t for t in eng.transcribe(chunks, lang, batch_size=4) if t).strip()


def do_transcribe(audio, lang, version):
    if audio is None:
        raise gr.Error("Record or upload audio first.")
    sr, y = audio
    wav = to_mono_16k(y, sr)
    with _lock:
        eng = stt_engine()
        try:
            v = eng.use(version)
        except OSError as e:
            raise _unknown_version_error(e, "stt", version)
        text = transcribe_long(eng, wav, lang)
    return text, text, v


def do_gemma_stt(text, lang, hint):
    if not (text or "").strip():
        raise gr.Error("Transcribe first (or type a transcript) so Gemma has something to correct.")
    if not gemma.available():
        raise gr.Error(f"Ollama model '{gemma.MODEL}' not reachable at {gemma.OLLAMA_URL}.")
    try:
        return gemma.suggest_transcript(text, lang, hint)
    except (OSError, ValueError, KeyError) as e:
        raise gr.Error(f"Gemma request failed: {e}")


def do_save_stt(audio, lang, model_out, corrected, used_version, category):
    if audio is None or not corrected.strip():
        raise gr.Error("Need the audio and a corrected transcript.")
    sr, y = audio
    uid = record_stt_correction(store, audio=to_mono_16k(y, sr), lang=lang, model_output=model_out or "",
                                corrected=corrected, model_version=used_version or "unknown",
                                category=category)
    return f"Saved as training sample `{uid}`. It will be used in the next STT training run."


# ---------------------------------------------------------------- TTS tab
def do_speak(text, lang, version):
    if not text.strip():
        raise gr.Error("Enter text.")
    with _lock:
        tts = tts_engine()
        v = tts.resolve(lang, version)
        try:
            wav, sr = tts.synthesize(text, lang, v)
        except OSError as e:
            raise _unknown_version_error(e, f"tts-{lang}", v)
    return (sr, wav), v


def do_gemma_tts(text, lang):
    if not (text or "").strip():
        raise gr.Error("Enter text first.")
    if not gemma.available():
        raise gr.Error(f"Ollama model '{gemma.MODEL}' not reachable at {gemma.OLLAMA_URL}.")
    try:
        out = gemma.suggest_spoken_form(text, lang)
        if not gemma.script_ok(out, lang):
            gr.Warning(f"Gemma's suggestion is not pure {LANGS[lang]['name']} script - please edit it before saving.")
        return out
    except (OSError, ValueError, KeyError) as e:
        raise gr.Error(f"Gemma request failed: {e}")


def do_preview_spoken(spoken, lang):
    if not (spoken or "").strip():
        raise gr.Error("Write or generate the spoken form first.")
    with _lock:
        wav, sr = tts_engine().synthesize(spoken, lang, "base")
    return sr, wav


def do_save_tts(text, lang, spoken, recording, used_version, category):
    if not text.strip() or not (spoken.strip() or recording is not None):
        raise gr.Error("Need the text and either a spoken form or a recording.")
    rec = to_mono_16k(recording[1], recording[0]) if recording is not None else None
    with _lock:
        uid = record_tts_correction(store, tts_engine(), lang=lang, text=text, spoken=spoken or None,
                                    recording=rec, model_version=used_version or "unknown", category=category)
    return f"Saved as training sample `{uid}`. It will be used in the next {LANGS[lang]['name']} TTS training run."


# ---------------------------------------------------------- training tab
def stats_table():
    return [[r["modality"], r["lang"], r["role"], r["batch"], r["n"], r["errors"]] for r in store.stats()]


def versions_table():
    rows = []
    reg = Registry()
    for key, e in reg.data.items():
        for v, m in e["versions"].items():
            rows.append([key, v, m.get("parent"), "yes" if e["active"] == v else "",
                         m.get("n_corrections"), m.get("train_seconds"), "; ".join(m.get("gate", []))[:160]])
    return rows


def do_train(what, force, progress=gr.Progress()):
    logs: list[str] = []
    log = logs.append
    with _lock:
        try:
            if what == "stt":
                _engines.pop("stt", None); free_gpu()       # training needs the GPU memory
                cycle_stt(store, force=force, log=log)
            else:
                lang = what.split("-")[1]
                tts = tts_engine(); tts.drop(lang)
                cycle_tts(store, tts, stt_engine(), lang, force=force, log=log)
                tts.drop(lang)
        except RuntimeError as e:
            raise gr.Error(str(e))                          # e.g. "no STT corrections stored yet"
        finally:
            if "stt" in _engines:
                _engines.pop("stt"); free_gpu()             # reload with the new registry state
    return "\n".join(logs), stats_table(), versions_table()


def do_rollback(key, version):
    version = version.strip()
    try:
        Registry().rollback(key, version)
    except KeyError:
        known = ", ".join(["base"] + list(Registry().versions(key)))
        raise gr.Error(f"'{version}' is not a known version of {key}. Known: {known}")
    with _lock:
        _engines.pop("tts", None)
    threading.Thread(target=warmup, daemon=True).start()
    return f"{key} active -> {version}", versions_table()


# ----------------------------------------------------------- compare tab
def do_compare_stt(audio, lang, va, vb):
    if audio is None:
        raise gr.Error("Record or upload audio first.")
    sr, y = audio
    wav = to_mono_16k(y, sr)
    with _lock:
        eng = stt_engine()
        for v in (va, vb):
            try:
                eng.use(v)
            except OSError as e:
                raise _unknown_version_error(e, "stt", v)
        eng.use(va); a = transcribe_long(eng, wav, lang)
        eng.use(vb); b = transcribe_long(eng, wav, lang)
    return a, b


def do_compare_tts(text, lang, va, vb):
    if not text.strip():
        raise gr.Error("Enter text.")
    with _lock:
        tts = tts_engine()
        try:
            a, sr = tts.synthesize(text, lang, tts.resolve(lang, va))
            b, sr2 = tts.synthesize(text, lang, tts.resolve(lang, vb))
        except OSError as e:
            raise _unknown_version_error(e, f"tts-{lang}", va)
    return (sr, a), (sr2, b)


# ------------------------------------------------------------------- about
ROOT = Path(__file__).resolve().parents[1]


def _readme_section(title: str) -> str:
    """One '## title' section of README.md, so the About tab never drifts from the docs."""
    try:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        body = text.split(f"## {title}", 1)[1]
        return body.split(chr(10) + "## ", 1)[0].strip()
    except (OSError, IndexError):
        return "_(see README.md)_"


def about_md() -> str:
    reg = Registry()
    rows = []
    models = [("stt", "Speech → Text (Whisper + LoRA)")] + [(f"tts-{k}", f"Text → Speech {v['name']} (MMS-VITS)") for k, v in LANGS.items()]
    for key, label in models:
        versions = reg.versions(key)
        rows.append(f"| {label} | **{reg.active(key)}** | {len(versions)} |")
    n_corr = sum(r["n"] for r in store.stats() if r["role"] == "correction")
    return f"""
## What is VaakSetu?
*Vāk* (speech) + *setu* (bridge). VaakSetu is a speech-to-text and text-to-speech system for
**Tamil, Hindi and English** that **learns from human corrections**. When a model gets a name, a ₹ amount,
a date, a vehicle number, α/β/γ or °C wrong, a person corrects it once. The correction is stored and the
model is **fine-tuned**, so the fix lives in the model weights: no dictionaries or find-and-replace rules.

## How it learns
```
Base model → wrong output → human correction (this app / bulk import) → append-only store (SQLite + audio)
  → fine-tune next version (warm start, all corrections so far + replay of general speech)
  → automatic evaluation → promotion gate (no regressions allowed) → active version
```
| | Base model | What is trained | What stays frozen |
|---|---|---|---|
| STT | `openai/whisper-large-v3-turbo` | LoRA adapters on the decoder (4.3 M params, 0.5 %) | whole audio encoder + base weights |
| TTS | `facebook/mms-tts-tam / hin / eng` | text encoder + new symbol tokens (α, ₹, °, digits …) | duration predictor, flow, decoder (the voice) |
| Assistant | Gemma (`gemma3n:e2b`, Ollama) | – (only suggests corrections; a human confirms) | – |

Anti-forgetting: cumulative data, warm start, replay, Learning-without-Forgetting distillation (STT),
frozen acoustic modules (TTS), and a promotion gate with rollback.

## Models right now
| Model | Active version | Versions trained |
|---|---|---|
{chr(10).join(rows)}

Stored corrections: **{n_corr}**

## Results at a glance
{_readme_section("Results at a glance")}

## Learn more
- `docs/RESULTS.md` and `results/report.html`: full before/after tables and audio examples
- `docs/TECHNICAL.md`: model choice, trainable/frozen layers, hyperparameters, data format, versioning
- `docs/DEMO.md`: step-by-step demo
"""


# ------------------------------------------------------------------- UI
def build() -> gr.Blocks:
    with gr.Blocks(title="VaakSetu") as ui:
        gr.Markdown("# VaakSetu\nSpeech models that learn from your corrections. Tamil · Hindi · English")
        with gr.Tab("About"):
            about = gr.Markdown(about_md())
            gr.Button("Refresh", size="sm").click(about_md, outputs=about)
        with gr.Tab("Speech → Text"):
            with gr.Row():
                s_lang = gr.Dropdown(LANG_CHOICES, value="ta", label="Language")
                s_ver = gr.Dropdown(versions("stt"), value="active", label="Model version")
            s_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speech")
            s_btn = gr.Button("Transcribe", variant="primary")
            s_out = gr.Textbox(label="Model output", interactive=False)
            s_used = gr.Textbox(label="Version used", interactive=False)
            s_fix = gr.Textbox(label="Correct transcript (edit this)", lines=2)
            with gr.Row():
                s_hint = gr.Textbox(label="Hint for Gemma (optional, e.g. 'the name is Sundaramoorthy')")
                s_gem = gr.Button("Suggest with Gemma")
            s_cat = gr.Dropdown(CATEGORIES, value="other", label="Category")
            s_save = gr.Button("Save correction → training data", variant="primary")
            s_msg = gr.Markdown()
            s_btn.click(do_transcribe, [s_audio, s_lang, s_ver], [s_out, s_fix, s_used])
            s_gem.click(do_gemma_stt, [s_out, s_lang, s_hint], s_fix)
            s_save.click(do_save_stt, [s_audio, s_lang, s_out, s_fix, s_used, s_cat], s_msg)

        with gr.Tab("Text → Speech"):
            with gr.Row():
                t_lang = gr.Dropdown(LANG_CHOICES, value="ta", label="Language")
                t_ver = gr.Dropdown(["active", "base"], value="active", label="Model version")
            t_text = gr.Textbox(label="Text", lines=2, value="டிக்கெட் விலை ₹500.")
            t_btn = gr.Button("Speak", variant="primary")
            t_audio = gr.Audio(label="Model speech", interactive=False)
            t_used = gr.Textbox(label="Version used", interactive=False)
            gr.Markdown("**Wrong pronunciation?** Write how it should be read (spoken form) "
                        "or record the correct reading.")
            with gr.Row():
                t_spoken = gr.Textbox(label="Spoken form", lines=2)
                t_gem = gr.Button("Suggest with Gemma")
            t_prev = gr.Button("Preview target (base voice reading the spoken form)")
            t_prev_audio = gr.Audio(label="Target pronunciation", interactive=False)
            t_rec = gr.Audio(sources=["microphone", "upload"], type="numpy",
                             label="…or your own recording (optional)")
            t_cat = gr.Dropdown(CATEGORIES, value="other", label="Category")
            t_save = gr.Button("Save correction → training data", variant="primary")
            t_msg = gr.Markdown()
            t_lang.change(lambda l: gr.Dropdown(choices=versions(f"tts-{l}"), value="active"), t_lang, t_ver)
            t_btn.click(do_speak, [t_text, t_lang, t_ver], [t_audio, t_used])
            t_gem.click(do_gemma_tts, [t_text, t_lang], t_spoken)
            t_prev.click(do_preview_spoken, [t_spoken, t_lang], t_prev_audio)
            t_save.click(do_save_tts, [t_text, t_lang, t_spoken, t_rec, t_used, t_cat], t_msg)

        with gr.Tab("Train & Versions"):
            gr.Markdown("Training uses **all** stored corrections (old + new) plus replay data, "
                        "evaluates against the current model and promotes only if nothing regressed.")
            d_stats = gr.Dataframe(stats_table(), headers=["modality", "lang", "role", "batch", "n", "errors"],
                                   label="Stored training data")
            d_vers = gr.Dataframe(versions_table(), label="Model versions",
                                  headers=["model", "version", "parent", "active", "corrections", "train s", "gate"])
            with gr.Row():
                tr_what = gr.Dropdown(["stt"] + [f"tts-{l}" for l in LANGS], value="stt", label="Train")
                tr_force = gr.Checkbox(label="Promote even if the gate fails")
                tr_btn = gr.Button("Train next version", variant="primary")
            tr_log = gr.Textbox(label="Training log", lines=12)
            tr_btn.click(do_train, [tr_what, tr_force], [tr_log, d_stats, d_vers])
            with gr.Row():
                rb_key = gr.Dropdown(["stt"] + [f"tts-{l}" for l in LANGS], value="stt", label="Model")
                rb_ver = gr.Textbox(label="Version (e.g. v001 or base)")
                rb_btn = gr.Button("Set active / rollback")
            rb_msg = gr.Markdown()
            rb_btn.click(do_rollback, [rb_key, rb_ver], [rb_msg, d_vers])

        with gr.Tab("Compare"):
            with gr.Row():
                c_lang = gr.Dropdown(LANG_CHOICES, value="ta", label="Language")
                c_va = gr.Dropdown(versions("stt"), value="base", label="Version A", allow_custom_value=True)
                c_vb = gr.Dropdown(versions("stt"), value="active", label="Version B", allow_custom_value=True)
            gr.Markdown("### STT")
            c_audio = gr.Audio(sources=["microphone", "upload"], type="numpy")
            c_sbtn = gr.Button("Transcribe with A and B")
            with gr.Row():
                c_sa, c_sb = gr.Textbox(label="A"), gr.Textbox(label="B")
            c_sbtn.click(do_compare_stt, [c_audio, c_lang, c_va, c_vb], [c_sa, c_sb])
            gr.Markdown("### TTS  (versions refer to the TTS model of the chosen language)")
            c_text = gr.Textbox(label="Text", value="α கதிர்கள் பற்றி இன்றைய பாடம்.")
            c_tbtn = gr.Button("Speak with A and B")
            with gr.Row():
                c_ta, c_tb = gr.Audio(label="A"), gr.Audio(label="B")
            c_tbtn.click(do_compare_tts, [c_text, c_lang, c_va, c_vb], [c_ta, c_tb])
    return ui


def warmup() -> None:
    """Load every model once at start-up so the first click in a demo is as fast as the rest."""
    import time
    import numpy as np
    t0 = time.time()
    try:
        with _lock:
            eng = stt_engine()
            eng.use("active")
            eng.transcribe([np.zeros(16000, dtype=np.float32)], "en")
            print(f"[warmup] STT ready ({time.time() - t0:.0f}s)", flush=True)
            tts = tts_engine()
            for lang in LANGS:
                tts.synthesize("a", lang, "active")
            print(f"[warmup] TTS ready ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:                               # warm-up must never stop the app
        print(f"[warmup] STT/TTS skipped: {e}", flush=True)
    print(f"[warmup] Gemma {'ready' if gemma.warmup() else 'unavailable (Ollama not running?)'} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    threading.Thread(target=warmup, daemon=True).start()
    # no fixed port: Gradio uses GRADIO_SERVER_PORT if set, otherwise the first free port from 7860 up
    build().queue().launch(server_name="127.0.0.1")
