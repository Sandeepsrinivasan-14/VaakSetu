"""Standalone loader - proves saved models are reusable outside this project.

Uses ONLY transformers + peft + soundfile/librosa (no `vaaksetu` import), so the
same code runs on Kaggle/Colab/another PC after copying a model folder.

  python scripts/load_demo.py stt models/stt/v002 clip.wav --lang ta [--device cpu]
  python scripts/load_demo.py tts models/tts/ta/v002 "டிக்கெட் விலை ₹500." out.wav [--device cpu]
  python scripts/load_demo.py merge models/stt/v002 dist/whisper-vaaksetu-merged
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_stt(folder: str, device: str):
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    folder = Path(folder)
    base = json.loads((folder / "adapter" / "adapter_config.json").read_text())["base_model_name_or_path"]
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(base, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(folder / "adapter")).to(device).eval()
    proc = WhisperProcessor.from_pretrained(str(folder / "processor"))
    # same decoding setting as training/eval: allow the symbols our conventions use
    gc = model.generation_config
    gc.suppress_tokens = [t for t in (gc.suppress_tokens or [])
                          if not any(s in proc.tokenizer.decode([t]) for s in "/@:+#")]
    return model, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stt", "tts", "merge"])
    ap.add_argument("model")
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--lang", default="ta")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252 (₹, α, Tamil)

    import torch
    if a.mode == "stt":
        import librosa
        model, proc = load_stt(a.model, a.device)
        wav, _ = librosa.load(a.inp, sr=16000)
        feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(a.device, model.dtype)
        with torch.inference_mode():
            ids = model.generate(input_features=feats, language=a.lang, task="transcribe", max_new_tokens=128)
        print(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())

    elif a.mode == "tts":
        import soundfile as sf
        from transformers import AutoTokenizer, VitsModel
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        model = VitsModel.from_pretrained(a.model).to(a.device).eval()
        torch.manual_seed(0)
        with torch.inference_mode():
            wav = model(**tok(a.inp, return_tensors="pt").to(a.device)).waveform[0].cpu().numpy()
        sf.write(a.out or "out.wav", wav, model.config.sampling_rate)
        print(f"wrote {a.out or 'out.wav'} ({len(wav) / model.config.sampling_rate:.2f}s)")

    else:  # merge LoRA into a plain Whisper checkpoint (no peft needed to use it)
        model, proc = load_stt(a.model, "cpu")
        merged = model.merge_and_unload()
        merged.save_pretrained(a.inp)
        proc.save_pretrained(a.inp)
        print(f"merged full model saved to {a.inp}")


if __name__ == "__main__":
    main()
