# VaakSetu – Technical Documentation

VaakSetu (*vāk* "speech" + *setu* "bridge") is a speech-to-text (STT) and text-to-speech (TTS) system for **Tamil, Hindi and English (Indian)**. It improves from human corrections by training the models themselves: adapters and model weights change, not lookup tables or text-replacement rules.

```
Base model → incorrect output → human correction → training data (append-only store)
          → fine-tuning (adapter / partial) → new version → evaluation + promotion gate
          → retest → improved output → (repeat with new corrections, old ones kept)
```

---

## 1. Model selection

| | STT | TTS |
|---|---|---|
| **Base model** | `openai/whisper-large-v3-turbo` (809 M params, MIT) | `facebook/mms-tts-tam`, `-hin`, `-eng` (VITS, 36.3 M params each) |
| **Why** | Strongest open multilingual ASR that still runs on a 4 GB laptop GPU in bf16 (≈1.6 GB). Supports ta/hi/en natively. Mature LoRA tooling (PEFT). | One of the few open TTS families with **Tamil and Hindi** checkpoints that can be fully fine-tuned on a 4 GB GPU. End-to-end VITS: text → waveform with no separate vocoder. |
| **Capabilities** | Transcription in 99 languages. Handles code-mixed speech. Outputs arbitrary Unicode (α, ₹, °, µ). | Character-level input. The base vocabulary has no symbols, and English has only digits 0–6, so unknown characters are **silently dropped** (the main failure mode). |
| **Licence** | MIT | CC-BY-NC 4.0 (fine for research/evaluation; for commercial use swap in AI4Bharat Indic Parler-TTS (Apache-2.0), which needs a 16 GB GPU to train) |
| **Rejected** | Whisper-small (poor Tamil). IndicConformer (NeMo, fragile on Windows). Gemma 3n/4 as ASR (≥ 5 GB, cannot be fine-tuned on 4 GB). | XTTS-v2 (non-commercial licence, too heavy). Indic Parler-TTS (0.9 B, too big to train locally). |

**Gemma** (`gemma3n:e2b` via Ollama, configurable to Gemma 4 with `VAAKSETU_GEMMA_MODEL`) is used as a *correction assistant only*:
- It suggests a corrected transcript or the spoken form of a sentence.
- The human accepts or edits the suggestion before it is saved.
- It is never used at inference time and never writes training data on its own.
- Measured with `gemma3n:e2b`: English and Hindi suggestions were correct (`alpha particles` → `α particles`; `तीन हजार रुपए` → `₹3,000`; a correct spoken form for `₹2,500 … 15/08/2024`). The Tamil spoken form for `₹2,500` was wrong, so human confirmation is required. A larger Gemma is recommended for Tamil.

## 2. Input / output coverage

| Category | STT target convention | TTS example (written → spoken) |
|---|---|---|
| Names | correct spelling in the sentence's script (ஸ்ரீநிவாசன், श्रीनिवासन, Srinivasan) | – (native script is already handled) |
| Greek / Unicode | `α particles`, `β क्षय`, `γ கதிர்கள்` | `α` → "alpha / अल्फा / ஆல்பா" |
| Currency | `₹2,500`; decimal paise kept as given (`₹5,850.50`) | `₹2,500` → "two thousand five hundred rupees"; `₹5,850.50` → "ஐயாயிரத்து எண்ணூற்று ஐம்பது ரூபாய் ஐம்பது காசு" |
| Dates | `15/08/2024` (DD/MM/YYYY) or `24-07-2026` (DD-MM-YYYY) - whichever separator the source uses | → "fifteenth august twenty twenty four" / "இருபத்து நான்கு ஜூலை இரண்டாயிரத்து இருபத்தாறு" |
| Time (24h) | `14:30`, `18:00` | → "பதினான்கு முப்பது" / "பதினெட்டு மணி" |
| Alphanumeric | `TN09AB1234`, `ABCDE1234F`, or hyphenated `TN88-AB-1234` | → "t n zero nine a b one two three four" (hyphens ignored when spelling out) |
| Units / symbols | `37°C`, `45%`, `5 µg`, `98.6°F` | → "thirty seven degrees celsius" |
| Technical / code-mixed terms | `எல்பிஜி` (LPG), `ஜிபிஎஸ்` (GPS), `டாஷ்போர்டு` (dashboard), `ஃப்ளீட் மேனேஜர்` (fleet manager) - English loanwords already transliterated into the sentence's script | acronyms spelled letter-by-letter (`எல்பிஜி` → "எல் பி ஜி"); transliterated loanword phrases read as written |

## 3. Languages

| Language | STT | TTS | Tested |
|---|---|---|---|
| Tamil (ta) | ✔ Whisper `ta` | ✔ `mms-tts-tam` | ✔ all categories |
| Hindi (hi) | ✔ Whisper `hi` | ✔ `mms-tts-hin` | ✔ all categories |
| English, Indian (en) | ✔ Whisper `en` | ✔ `mms-tts-eng` | ✔ all categories |

Other Whisper and MMS languages can be added by adding a row to `LANGS` in `vaaksetu/config.py`. They have not been tested.

## 4. Error correction and learning

**STT** (app tab *Speech → Text*, or `scripts/train.py import`):
1. The model transcribes the audio.
2. The human edits the transcript. Gemma can suggest a correction; the human confirms it.
3. The (audio, corrected text, original output, model version) record is stored.

**TTS** (app tab *Text → Speech*): the human flags a bad reading and supplies either
- a **spoken form**: the **frozen base voice** reads it to produce the target audio, so the voice stays identical, or
- **their own recording** of the correct reading.

The training pair is (**original written text** → target speech). The model learns to read `₹500` correctly itself. No substitution happens at inference time.

## 5. Model-level training

### STT – LoRA on the Whisper decoder
| | |
|---|---|
| Trainable | LoRA r=32, α=64, dropout 0.05 on `q,k,v,out_proj` of self- and cross-attention and on `fc1,fc2` of all 4 decoder layers: **4.26 M params (0.52 %)** |
| Frozen | the whole 32-layer audio encoder, embeddings, all base weights (**≈ 809 M**) |
| Why decoder-only | The errors are in *what is written* (spelling, formatting, symbols), which is the decoder's job. The frozen encoder keeps acoustic robustness intact and is run under `no_grad`, so training fits in 4 GB. |
| Loss | token cross-entropy on the corrected transcript (forced language/task tokens masked). On replay clips it adds **Learning-without-Forgetting** KL(base ‖ adapted) on the token distributions: the frozen base is obtained by disabling the adapter, weight 1.0. |
| Decoding | Whisper's default `suppress_tokens` forbids `/ @ : + # -`, so `15/08/2024` or a hyphenated `24-07-2026` / `TN88-AB-1234` could never be produced. Those tokens are un-suppressed, with the same setting for base and all versions. `max_new_tokens` is 400 (Whisper's hard decoder limit is 448 total positions) so a long multi-clause sentence is never silently truncated mid-output. |
| Precision | base in bf16, adapters in fp32, autocast |
| Hyperparameters | lr 3e-4 (×0.5 when warm-starting from a previous adapter; cosine schedule, 10 % warm-up), 6 epochs, batch 8, AdamW wd 0.01, grad-clip 1.0, replay ratio 0.5, seed 42 |
| Augmentation | on-the-fly speed perturbation 0.9–1.1×, gain 0.6–1.2, Gaussian noise at 15–30 dB SNR |
| **EMA** | the saved adapter is an exponential moving average of the weights (decay 0.999), not the last step. Decay is warmup-corrected (`d = min(0.999, step/(step+10))`) so a ~200-step run still tracks a useful average instead of sitting near the random LoRA init. Smooths out noise from training on a few hundred clips. |
| **Per-language loss weight** | Tamil corrections are weighted ×1.4 in the token-level cross-entropy (`lang_loss_weight` in `configs/stt.yaml`); Hindi/English stay at ×1.0. Anchors are *never* reweighted, so the replay/LwF anti-forgetting balance is unaffected — only the correction loss for the language that needs the most help gets a bigger gradient share. |
| **Balanced batch replay** | Corrections are grouped by their collection batch (`round1`, `round2`, an import, the live app's `ui` batch, ...). Each batch contributes the *same* number of rows per epoch (resampled with replacement if it's smaller than the largest), instead of flat concatenation. This directly targets the mechanism behind the documented Tamil round-2 regression: a large later batch would otherwise dominate the epoch's gradient and quietly erode what a smaller, earlier batch taught. `balance_batches: true` in `configs/stt.yaml`; disable to reproduce the old behaviour. |

### TTS – partial fine-tuning of VITS
| | |
|---|---|
| Trainable | `text_encoder` (incl. new symbol embeddings; base-vocabulary rows frozen by a gradient mask) |
| Frozen | `duration_predictor`, `posterior_encoder`, `flow`, `decoder` (HiFi-GAN). The voice, timbre and rhythm cannot drift. |
| Objective | VITS prior-side losses: the frozen posterior encoder and flow map the target speech to latents. **Monotonic Alignment Search** aligns them to the characters. Loss = closed-form KL(posterior‖text prior), including the posterior variance term. No discriminator is needed. |
| New symbols | Each unseen character (α β γ ₹ ° % / µ, missing digits, Latin letters inside Tamil/Hindi text) and every digit becomes **4 learnable sub-token slots** (`α, α#1, α#2, α#3`). One VITS token has only one prior, so a multi-phoneme reading ("alpha") needs several positions. This is sub-word tokenisation; what each slot sounds like is learned. |
| Learning rates | new embeddings 2e-3, text encoder 2e-5. Adam β=(0.8, 0.99). |
| Other | 60 epochs, batch 8, grad-clip 5, dropout **and layerdrop off** during fine-tuning, posterior **mean** used as the target latent (lower variance) |
| **EMA** | same warmup-corrected exponential moving average (decay 0.999) as STT, applied to the trainable text-encoder weights (including the new symbol embeddings) before saving. |

**Why these choices (Tamil ablation, 21 corrections + 10 anchors, round-trip CER; base: corrected 0.364 / held-out 0.270 / general 0.133):**

| Variant | corrected | held-out | general |
|---|---|---|---|
| KL without posterior-variance term (first attempt) | 0.625 | 0.544 | 0.477 |
| text enc. + duration predictor, train mode (dropout + layerdrop 0.1) | 0.536 | 0.534 | 0.229 |
| same, dropout off | 0.483 | 0.288 | 0.237 |
| duration predictor + new embeddings only | 0.553 | 0.382 | 0.277 |
| new embeddings only | 0.312 | 0.244 | 0.133 |
| **text encoder + new embeddings, duration frozen (chosen)** | **0.217** | **0.234** | **0.131** |

Rows 2–4 were 30-epoch runs, the others 60. Fine-tuning the stochastic duration predictor on a few dozen sentences damaged *all* speech, so it stays frozen. (Ablation scripts are kept locally in `archive/tts-ablation/`.)

## 6. Training data persistence

```
data/vaaksetu.db            SQLite (schema in vaaksetu/data/store.py)
  samples                   uid, modality, lang, category, text, spoken, audio_sha,
                            model_output, model_version, was_error, role, batch, source,
                            status (active|retired), created_at
  training_runs             model_key, version, parent, base_model, snapshot file + SHA-256,
                            n_samples, hyper-parameters, metrics, promoted
  triggers                  DELETE is blocked on both tables. Retiring is a soft flag.
data/audio/ab/<sha256>.wav  content-addressed audio. Identical audio is stored once and
                            files are never overwritten.
data/snapshots/*.jsonl      exact training-set manifest of every run (reproducible)
data/exports/*.jsonl        full dataset export (`python scripts/train.py export`)
```

Inserts are idempotent (unique `uid`), so re-running a script never duplicates data. New training runs **read** the store and never modify existing samples.

## 7. Continued learning (retaining earlier behaviour)

1. **Cumulative data.** Every run trains on *all* active corrections (old and new), not just the latest batch.
2. **Warm start.** STT continues from the parent adapter. TTS continues from the parent weights, including previously learned symbols.
3. **Replay.** STT mixes in general-speech anchor clips with correct transcripts. TTS mixes in *self-distilled* anchors: the base voice reading general sentences.
4. **Stability controls.**
   - STT: the encoder is frozen, the adapters are low-rank, *Learning-without-Forgetting* distillation to the base model runs on replay clips, and the learning rate is halved for warm-started runs.
   - TTS: acoustic and duration modules are frozen, base-vocabulary embeddings are frozen, and the context layers use a low learning rate.
5. **Promotion gate.** A new version is evaluated against the active one on general speech *and* on every earlier correction set, both corrected and held-out. A set only fails the gate when its regression both exceeds the flat tolerance (0.02 WER / 0.03 CER) on the observed eval clips *and* survives a paired bootstrap over those clips — the 5th-percentile lower bound of the resampled regression must still exceed tolerance (`vaaksetu/learning.py:gate`, 300 resamples). This means a one-off unlucky eval clip can no longer flip a promotion decision on its own, while a real, consistent regression (like STT v004's Tamil WER jump from 50% to 60%) still fails easily. Otherwise the candidate stays unpromoted and can be inspected or force-promoted.
6. **Version history is kept, never rewritten.** Superseded experiments stay in the registry with a `round` tag such as `round1@recipe1` and a note:
   - `stt v001`/`v002`: first recipe, before LwF distillation and the `/` un-suppression;
   - `tts-ta v001`/`v002`: before the KL fix and the frozen duration predictor.
7. **Rollback.** `python scripts/train.py rollback stt v001`

`docs/RESULTS.md` demonstrates this with the original two rounds, plus a third added later:
- **Round 1:** names, Greek letters, currency.
- **Round 2:** dates, alphanumerics, units.
- **Round 3 (Tamil only):** a single mandatory real-world sentence (`vaaksetu/data/seed_catalog.py`,
  `TA["round3"]["fleet_alert"]`; trained via `scripts/round3_fleet_alert.py`) combining a
  hyphenated date, 24-hour time (a new category), a hyphenated vehicle registration, decimal-paise
  currency, and code-mixed English-in-Tamil technical terms (LPG, GPS, dashboard, fleet manager).
  Per that requirement the exact same sentence is used for training, testing, and the demo, so
  round3 is intentionally excluded from `tests/test_core.py::test_catalog_splits_are_disjoint`.

The report shows every earlier round's sets still evaluated after each new round's training, so a
regression in Tamil, English or Hindi from round1/round2 (or round3's own held-out audio) would be
caught by the promotion gate before the new version becomes active.

## 8. Model storage, versioning and reusability

```
models/registry.json          {"stt": {"active": "v003", "versions": {...}}, "tts-ta": {...}}
models/stt/v003/adapter/      adapter_model.safetensors + adapter_config.json (PEFT)
models/stt/v003/processor/    feature extractor + tokenizer
models/stt/v003/metadata.json parent, snapshot hash, hyper-params, loss curve, param counts, gate
models/tts/ta/v005/           model.safetensors + config.json + vocab.json + tokenizer_config.json
                              + tokenization_vaaksetu.py (slot tokenizer, auto_map) + metadata.json
```

**Loading elsewhere** (only `transformers`, `peft` and `soundfile` needed; see `scripts/load_demo.py`):

```python
# STT
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
model = PeftModel.from_pretrained(base, "models/stt/v003/adapter")
proc = WhisperProcessor.from_pretrained("models/stt/v003/processor")

# TTS
from transformers import AutoTokenizer, VitsModel
tok = AutoTokenizer.from_pretrained("models/tts/ta/v005", trust_remote_code=True)
tts = VitsModel.from_pretrained("models/tts/ta/v005")
```

`python scripts/load_demo.py merge models/stt/v003 dist/whisper-merged` merges the LoRA into a plain Whisper checkpoint that needs no PEFT.

## 8b. TTS number-reading generalisation (optional augmentation)

Round-trip CER on *unseen* numbers/dates does not reliably improve from a
handful of human-corrected examples (documented in RESULTS.md and section
10). `vaaksetu/tts/augment.py` adds an opt-in fix for English currency:
`python scripts/train.py augment-numbers --n 30` generates N synthetic
(written, spoken) pairs — e.g. `("₹4,250.", "four thousand two hundred and
fifty rupees")` — via a hand-verified English cardinal-number converter
(cross-checked against the seed catalogue's own hand-written phrasings in
`tests/test_core.py`), synthesises the target audio with the frozen base
voice (identical mechanism to the existing self-distilled anchors — no
inference-time rule is added), and stores them as corrections tagged
`batch="augment-numbers"`, `source="synthetic-numbers"` so they stay
distinguishable and reversible.

Hindi and Tamil are intentionally **not** covered: their cardinal-number
grammar is irregular enough that a hand-written converter risks silently
teaching the model wrong pronunciations, which would be worse than doing
nothing. That needs either a native-speaker-verified converter or a
well-supported library — future work, not guessed here.

## 9. Adding new corrections later

```bash
python app/app.py                              # collect corrections in the browser, or
python scripts/train.py import my_fixes.jsonl  # bulk import (format in the script docstring)
python scripts/train.py status                 # what is stored / which versions exist
python scripts/train.py stt                    # train → evaluate → gate → promote
python scripts/train.py tts --lang ta
```

Each run creates the next version (`v003`, …) warm-started from the active one, trained on the cumulative data, snapshotted and gated.

## 10. Evaluation methodology

- **Seed data** (`vaaksetu/data/seed_catalog.py`, `scripts/build_seed_audio.py`): 372 sentences, rendered as 744 clips with two neural voices per language.
  - **Held-out** sets use different carrier sentences, and for rule-like categories different values (new amounts, dates and codes), so a retest is never the clip that was corrected.
  - The human corrector is simulated by the curated ground truth, so the run is reproducible. The app runs the identical loop with a real person.
- **STT metrics:** WER and CER after normalisation (case, punctuation and thousands separators ignored; symbols kept), plus **exact-term accuracy** (whether the required name, ₹ amount, date, code or symbol appears exactly).
- **TTS metric:** round-trip CER. The TTS output is transcribed by the **base** Whisper (independent of the STT adapters) and compared with the transcript of the target pronunciation; the general set is compared with the sentence itself. The report also lists the fraction of characters the tokenizer silently drops.
- **Limitations**
  - The seed recordings are synthetic (neural TTS voices), not field recordings.
  - Round-trip ASR is a proxy for listening tests.
  - TTS reading of *unseen* number values generalises only partially from a few examples. Symbols seen in training (α, ₹, °) transfer to new sentences much better.
  - STT v004 (round 2) was rejected by the gate. Round-1 held-out WER rose from 17.2% to 20.7%, almost entirely from Tamil (49.6% → 59.8%; Hindi 9.2% → 11.1%; English unchanged), while exact-term accuracy stayed at 81–82%.
  - Currency corrections make STT over-apply number formats: a PIN code came out as "₹6,000" with v003 and "6/04/2024" with v004. More varied number corrections (see `examples/new_corrections.jsonl`) are the remedy.
