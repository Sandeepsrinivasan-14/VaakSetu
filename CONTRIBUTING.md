# Contributing to VaakSetu

Thanks for helping. The project's one rule: **corrections must live in model weights**, never in
runtime dictionaries, regexes or find-and-replace code.

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
pytest -q                                                              # should pass offline
python app/app.py                                                      # http://127.0.0.1:7860
```

## Guidelines
- Keep the training data append-only (`vaaksetu/data/store.py`); never rewrite stored samples.
- A new model version must pass the promotion gate (`vaaksetu/eval`) or be explicitly forced.
- Don't commit model weights, audio or `data/vaaksetu.db` (see `.gitignore`); share them via releases.
- Add a test in `tests/` for any change to the store, registry, tokenizer or metrics.
- Hyper-parameters belong in `configs/*.yaml`, not in code.

## Gemma assistant
Gemma (via Ollama) only *suggests* corrections; a human confirms them. Environment variables:
`VAAKSETU_GEMMA_MODEL`, `VAAKSETU_GEMMA_GPU_LAYERS` (default `0`, i.e. CPU, so it never competes
with Whisper/VITS for a small GPU), `VAAKSETU_GEMMA_KEEP_ALIVE`, `OLLAMA_HOST`.

## Pull requests
Small, focused PRs with a short description of what changed and how you tested it.
