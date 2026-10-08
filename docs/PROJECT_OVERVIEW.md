# VaakSetu — Project Overview

*A simple guide to what the project is, how it works, and what it is built with.*

---

## 1. What is VaakSetu?

**VaakSetu** ("vāk" = speech, "setu" = bridge) is a **speech-to-text (STT)** and **text-to-speech (TTS)** system for **Tamil, Hindi and English (Indian)**.

Its main idea: **the models learn from human corrections.**

- When the STT model hears a clip and writes the wrong text (a wrong name, a ₹ amount, a date, a vehicle number, α/β/γ, °C…), a person fixes it once.
- When the TTS model mispronounces or skips a symbol or number, a person supplies the correct spoken form.
- Each correction becomes **training data**, and the model is **fine-tuned**, so the fix lives **inside the model weights**.

> Important rule: there are **no dictionaries, regexes or find-and-replace rules** at runtime. The model itself must learn.

The project was built for a technical exercise ("STT & TTS Model Training"), which required: open-source trainable models, support for Indian languages, a human-correction mechanism, real fine-tuning, persistent training data, continual learning without forgetting, and reusable saved models.

---

## 2. How it works

### The learning loop

```
Model gives a wrong output
        ↓
Human corrects it (Gradio app, or bulk JSONL import)
        ↓
Saved to the database  (data/vaaksetu.db — append-only, never deleted)
        ↓
Train next version     (python scripts/train.py stt | tts --lang ta)
   • uses ALL corrections so far (old + new)
   • starts from the previous version's weights (warm start)
   • mixes in "anchor" clips of normal speech (replay) to avoid forgetting
        ↓
Automatic evaluation vs. the currently active version
        ↓
Promotion gate: new version is promoted only if it does not get worse
        ↓
Better model becomes the active version  (old versions kept; rollback possible)
```

### The three components

| Component | Base model | What gets trained |
|---|---|---|
| **STT** | OpenAI Whisper large-v3-turbo (809M params) | Small **LoRA adapters** on the decoder only (~0.5% of weights). The audio encoder stays frozen. |
| **TTS** | Facebook MMS-TTS (VITS) — separate models for Tamil, Hindi, English | Only the **text encoder** (plus new symbol/digit embeddings). Voice, duration and vocoder parts stay frozen. |
| **Assistant** | Gemma (`gemma3n:e2b`, via local Ollama) | Not trained. Only **suggests** corrections; a human always confirms. |

### Key techniques

- **LoRA fine-tuning (STT):** cheap, small adapter files instead of retraining the whole model.
- **Vocabulary extension (TTS):** characters the base model silently drops (α, β, ₹, °, %, digits…) get new learnable tokens.
- **Anti-forgetting:** cumulative data, warm starts, replay of general-speech anchors, Learning-without-Forgetting (KL distillation), frozen modules, lower learning rate on continuing runs, and an EMA of weights.
- **Balanced batches + Tamil loss weighting:** stops a large later batch from erasing earlier learning (this fixed a real regression seen in STT v004).
- **Promotion gate with bootstrap check:** a version is rejected only if it truly regresses, not because of one noisy test clip.
- **Versioning & rollback:** every version is stored in `models/registry.json`; rolling back is just repointing the "active" version.

### Data storage

- **SQLite database** (`data/vaaksetu.db`): corrections and training history. Delete is blocked by database triggers — records are "retired", never removed.
- **Audio** stored by content hash (identical audio stored once).
- **Snapshots** of the exact training set for every run, hashed for reproducibility.

### Evaluation

- **STT:** WER / CER plus *exact-term accuracy* (is the name/amount/date/symbol exactly right?).
- **TTS:** *round-trip CER* — synthesize speech, transcribe it with the untouched base Whisper, and compare to the target.
- Test data is a seed catalogue of 372 sentences × 2 neural voices per language (744 clips), with training and held-out test templates kept separate.

### Latest additions

A mandatory Tamil "fleet alert" sentence (hyphenated date, 24-hour time, hyphenated vehicle number, decimal-paise currency, code-mixed technical terms) was added as "round3". Result: **STT v006 promoted with exact match on all clips**; TTS-ta v006 trained but was **rejected by the gate** (kept as a candidate; v005 stays active).

---

## 3. The app

`python app/app.py` → http://127.0.0.1:7860 (Gradio), with five tabs:

1. **About** — live project status and active model versions.
2. **Speech → Text** — transcribe, correct (optionally with Gemma's suggestion), save as training data.
3. **Text → Speech** — speak text, give the correct spoken form, save as training data.
4. **Train & Versions** — dataset stats, train the next version, rollback / set active.
5. **Compare** — run the same input through two versions side by side.

---

## 4. What we used

### Models
| Purpose | Model | Licence |
|---|---|---|
| STT | `openai/whisper-large-v3-turbo` | MIT |
| TTS | `facebook/mms-tts-tam`, `-hin`, `-eng` (VITS) | CC-BY-NC 4.0 |
| Correction assistant | `gemma3n:e2b` via Ollama | local |

### Languages & libraries
- **Python 3.10–3.12**
- **PyTorch** (CUDA) — training and inference
- **Transformers** — Whisper and VITS models
- **PEFT** — LoRA adapters
- **Accelerate**, **NumPy**, **Numba** (JIT for Monotonic Alignment Search in TTS training)
- **librosa**, **soundfile** — audio handling
- **jiwer** — WER / CER metrics
- **Gradio** — correction web UI
- **SQLite** — training-data store
- **PyYAML** — configs (`configs/stt.yaml`, `configs/tts.yaml`)
- **edge-tts** — generates the synthetic seed recordings (one-time, needs internet)
- **pytest** — 13 unit tests
- **Ollama** — runs Gemma locally (optional)

### Hardware
Developed on an RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM, Windows 11. STT training peaks at ~3.5 GB of GPU memory.

---

## 5. Project layout

```
vaaksetu/   core library (config, audio, text, data store, registry, learning loop,
            stt/, tts/, eval/, assist/)
scripts/    train.py (CLI) · run_pipeline.py · build_seed_audio.py · make_report.py
            · load_demo.py · round3_fleet_alert.py
app/        Gradio UI
configs/    stt.yaml, tts.yaml (all hyperparameters)
data/       database, audio, snapshots, exports, seed clips
models/     registry.json + versioned adapters/checkpoints
results/    evaluation outputs and report.html
docs/       REQUIREMENTS, TECHNICAL, RESULTS, DEMO, this overview
tests/      pytest suite
```

---

## 6. Quick start

```bash
pip install -r requirements.txt
python scripts/build_seed_audio.py     # create demo recordings (internet once)
python scripts/run_pipeline.py         # baseline → 2 correction rounds → retest
python scripts/make_report.py          # docs/RESULTS.md + results/report.html
python app/app.py                      # correction UI

python scripts/train.py status                     # see data + model versions
python scripts/train.py import file.jsonl          # add corrections in bulk
python scripts/train.py stt                        # train next STT version
python scripts/train.py tts --lang ta              # train next Tamil TTS version
python scripts/train.py rollback stt v001          # roll back
```

---

## 7. Known limitations

- Evaluation audio is synthetic (neural voices), not field recordings.
- Tamil STT error rate stays high on general speech (long agglutinative words).
- Currency corrections can make STT over-apply number formats.
- TTS generalizes only partly to *unseen* number values; symbols transfer better.
- Gemma suggestions are unreliable for Tamil — a human must always confirm.
- MMS-TTS is licensed non-commercially (CC-BY-NC).

For deeper detail see `docs/TECHNICAL.md` and `docs/RESULTS.md`.
