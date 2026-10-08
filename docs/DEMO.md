# Demo: how the models keep learning from new corrections

## A. With the app (a real human in the loop)

```bash
python app/app.py          # http://127.0.0.1:7860
```

**STT (Speech → Text tab)**
1. Choose the language (Tamil / Hindi / English), then record or upload speech and click **Transcribe**.
2. If the output is wrong (a misspelt name, `alpha` instead of `α`, `2500 rupees` instead of `₹2,500`, …), edit the *Correct transcript* box. **Suggest with Gemma** proposes a fix that you can accept or edit.
3. Click **Save correction → training data**. The audio, the original output, your correction and the model version are stored permanently in `data/vaaksetu.db` and `data/audio/`.

**TTS (Text → Speech tab)**
1. Type a sentence, for example `டிக்கெட் விலை ₹500.`, and click **Speak**. The base model skips `₹500` because those characters are not in its vocabulary.
2. Write how it should sound in *Spoken form*: `டிக்கெட் விலை ஐநூறு ரூபாய்`. **Suggest with Gemma** can draft this. Click **Preview target** to hear it in the same voice, or upload your own recording instead.
3. Click **Save correction → training data**.

**Train (Train & Versions tab)**
1. Select `stt` or `tts-ta` / `tts-hi` / `tts-en` and click **Train next version**. This:
   - trains `v00N+1` from the current active version on **all** stored corrections plus replay data;
   - evaluates it against the current version on general speech and on every earlier correction set;
   - promotes it only if nothing regressed (the log shows the reasons).
2. **Compare** plays or transcribes the same input with any two versions side by side.
3. **Set active / rollback** returns to any earlier version.

## B. From the command line

`examples/new_corrections.jsonl` holds four fresh corrections:
- a Tamil name,
- an English PIN code,
- a Hindi ₹ amount,
- a Tamil °C reading.

```bash
python scripts/train.py import examples/new_corrections.jsonl --batch round3
python scripts/train.py status                 # shows the new batch next to round1/round2
python scripts/train.py stt                    # -> models/stt/v003 (warm-started from v002)
python scripts/train.py tts --lang ta          # -> models/tts/ta/v003
python scripts/train.py tts --lang hi
```

What happens to the data:

| Step | What is stored | Where |
|---|---|---|
| import / save | one row per correction (never updated or deleted) + audio by SHA-256 | `data/vaaksetu.db`, `data/audio/` |
| train | frozen manifest of *exactly* the rows used, with its SHA-256 | `data/snapshots/stt_v003.jsonl` |
| train | adapter / weights + tokenizer/processor + `metadata.json` (parent, hyper-parameters, loss curve) | `models/stt/v003/` |
| gate | metrics + promoted flag | `training_runs` table, `models/registry.json` |

Earlier corrections are always part of the next run. Training reads the store and never deletes from it (SQLite triggers block DELETE). So adding round 3 cannot silently drop round 1 or round 2, and the gate checks that their accuracy holds.

## C. Reusing a trained model somewhere else

Copy the version folder, e.g. `models/stt/v002/` or `models/tts/ta/v002/`, then:

```bash
python scripts/load_demo.py stt models/stt/v002 clip.wav --lang ta --device cpu
python scripts/load_demo.py tts models/tts/ta/v002 "டிக்கெட் விலை ₹500." out.wav --device cpu
python scripts/load_demo.py merge models/stt/v002 dist/whisper-vaaksetu   # plain Whisper, no PEFT
```
