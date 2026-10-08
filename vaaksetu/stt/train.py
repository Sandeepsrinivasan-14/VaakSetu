"""LoRA fine-tuning of Whisper on human-corrected transcripts.

Trainable : LoRA adapters (r=32) on every decoder layer's self-attention,
            cross-attention and feed-forward projections.
Frozen    : the entire audio encoder, token embeddings, all base weights.

Continual learning
  * warm start from the parent version's adapter (keeps what it learned),
  * train on the *cumulative* set of active corrections (old + new),
  * mix in replay/anchor clips of general speech with correct transcripts,
  * the caller then evaluates and only promotes if nothing regressed.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_scheduler

from ..audio import augment, load_audio
from ..config import LANGS, SAMPLE_RATE, load_config
from .engine import pick_device_dtype


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def balanced_batch_rows(by_batch: dict, target_per_batch: int, rng: np.random.Generator) -> list[dict]:
    """One epoch's worth of correction rows, resampled so every correction
    batch (round1, round2, ...) contributes exactly `target_per_batch` rows
    regardless of its natural size."""
    rows: list[dict] = []
    for grp in by_batch.values():
        idx = rng.choice(len(grp), target_per_batch, replace=target_per_batch > len(grp))
        rows += [grp[i] for i in idx]
    return rows


def _count(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class _LabelBuilder:
    def __init__(self, tok):
        self.tok = tok
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.task = tok.convert_tokens_to_ids("<|transcribe|>")
        self.nots = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")

    def __call__(self, text: str, lang: str) -> list[int]:
        lang_id = self.tok.convert_tokens_to_ids(f"<|{LANGS[lang]['whisper']}|>")
        body = self.tok.encode(" " + text.strip(), add_special_tokens=False)
        return [self.sot, lang_id, self.task, self.nots] + body + [self.eot]


def train_stt(train_rows: list[dict], anchor_rows: list[dict], out_dir: Path,
              parent_adapter: Path | None = None, cfg: dict | None = None, log=print) -> dict:
    """Train an adapter.  Each row: {"audio_path", "text", "lang"}.  Returns stats."""
    cfg = cfg or load_config("stt")
    tc, lc = cfg["train"], cfg["lora"]
    _seed_all(tc["seed"])
    rng = np.random.default_rng(tc["seed"])
    device, dtype = pick_device_dtype()
    t0 = time.time()

    processor = WhisperProcessor.from_pretrained(cfg["base_model"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"], dtype=dtype)
    model.config.use_cache = False
    if parent_adapter:
        log(f"  warm-start from {parent_adapter}")
        model = PeftModel.from_pretrained(model, str(parent_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], bias="none"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()           # adapters train in fp32, base stays bf16
    model.to(device)
    trainable, total = _count(model)
    log(f"  trainable {trainable/1e6:.2f}M / total {total/1e6:.1f}M params "
        f"({100*trainable/total:.2f}%)")

    encoder = model.get_base_model().model.encoder.eval()
    labels_for = _LabelBuilder(processor.tokenizer)
    eot = labels_for.eot

    audio_cache: dict[str, np.ndarray] = {}

    def audio(path: str) -> np.ndarray:
        if path not in audio_cache:
            audio_cache[path] = load_audio(path)
        return audio_cache[path]

    anchor_rows = [{**r, "_anchor": True} for r in anchor_rows]
    n_anchor = min(len(anchor_rows) * 4, math.ceil(tc["replay_ratio"] * len(train_rows)))

    # Balanced replay across correction batches (round1, round2, ...).  Without
    # this, a later batch that happens to be larger than an earlier one
    # dominates each epoch's gradient and can quietly regress what the earlier
    # batch taught - the mechanism behind the documented Tamil round-2
    # regression.  Each batch contributes `target_per_batch` rows per epoch
    # (resampled with replacement if it's smaller), so no batch's influence
    # scales with how many sentences happen to be in it.
    by_batch: dict[str | None, list[dict]] = {}
    for r in train_rows:
        by_batch.setdefault(r.get("batch"), []).append(r)
    balance = tc.get("balance_batches", True) and len(by_batch) > 1
    target_per_batch = max(len(v) for v in by_batch.values()) if balance else None

    steps_per_epoch = math.ceil(((target_per_batch * len(by_batch) if balance else len(train_rows))
                                 + n_anchor) / tc["batch_size"])
    total_steps = steps_per_epoch * tc["epochs"] // tc["grad_accum"]
    lr = tc["learning_rate"] * (tc.get("continue_lr_scale", 1.0) if parent_adapter else 1.0)
    distill_w = float(tc.get("distill_weight", 0.0))
    lang_weight = tc.get("lang_loss_weight") or {}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=tc["weight_decay"])
    sched = get_scheduler("cosine", opt, num_warmup_steps=int(tc["warmup_ratio"] * total_steps),
                          num_training_steps=total_steps)

    # EMA of the adapter weights, applied before saving.  Decay is warmup-
    # corrected (grows towards `ema_decay` over the first ~10 steps) so short
    # runs on a few hundred corrections still benefit instead of the average
    # staying pinned near the random LoRA init.
    ema_decay = float(tc.get("ema_decay", 0.0) or 0.0)
    ema_state = {n: p.detach().clone() for n, p in model.named_parameters()
                if p.requires_grad} if ema_decay > 0 else None

    history = []
    step = 0
    model.train()
    for epoch in range(tc["epochs"]):
        anchors = [anchor_rows[i] for i in rng.choice(len(anchor_rows), n_anchor,
                                                        replace=n_anchor > len(anchor_rows))] if anchor_rows else []
        corrections = balanced_batch_rows(by_batch, target_per_batch, rng) if balance else list(train_rows)
        rows = corrections + anchors
        random.shuffle(rows)
        ep_loss, n = 0.0, 0
        for b in range(0, len(rows), tc["batch_size"]):
            batch = rows[b:b + tc["batch_size"]]
            wavs = [augment(audio(r["audio_path"]), rng) if tc["augment"] else audio(r["audio_path"])
                    for r in batch]
            feats = processor.feature_extractor(wavs, sampling_rate=SAMPLE_RATE,
                                                return_tensors="pt").input_features.to(device, dtype)
            seqs = [labels_for(r["text"], r["lang"]) for r in batch]
            L = max(len(s) for s in seqs) - 1
            dec = torch.full((len(seqs), L), eot, dtype=torch.long)
            lab = torch.full((len(seqs), L), -100, dtype=torch.long)
            for i, s in enumerate(seqs):
                dec[i, :len(s) - 1] = torch.tensor(s[:-1])
                lab[i, :len(s) - 1] = torch.tensor(s[1:])
                lab[i, :3] = -100             # forced lang/task/notimestamps tokens
            dec, lab = dec.to(device), lab.to(device)

            with torch.no_grad():
                enc = encoder(feats).last_hidden_state
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                logits = model(encoder_outputs=(enc,), decoder_input_ids=dec).logits
            tok_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), lab.reshape(-1),
                                       ignore_index=-100, reduction="none").reshape(lab.shape)
            valid = (lab != -100).float()
            # Per-language loss weight (anchors excluded: replay/anti-forgetting
            # balance must not shift just because a language is upweighted).
            w = torch.tensor([1.0 if r.get("_anchor") else float(lang_weight.get(r["lang"], 1.0))
                              for r in batch], device=device).unsqueeze(1)
            wmask = valid * w
            loss = (tok_loss * wmask).sum() / wmask.sum().clamp(min=1.0)
            is_anchor = torch.tensor([bool(r.get("_anchor")) for r in batch], device=device)
            if distill_w > 0 and bool(is_anchor.any()):
                # Learning-without-Forgetting: match the frozen base model on general speech
                with torch.no_grad(), model.disable_adapter(), \
                        torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                    t_logits = model(encoder_outputs=(enc,), decoder_input_ids=dec).logits
                m = (lab != -100) & is_anchor[:, None]
                kl = F.kl_div(F.log_softmax(logits.float()[m], -1), F.log_softmax(t_logits.float()[m], -1),
                              log_target=True, reduction="batchmean")
                loss = loss + distill_w * kl
            loss = loss / tc["grad_accum"]
            loss.backward()
            if (b // tc["batch_size"] + 1) % tc["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               tc["max_grad_norm"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if ema_state is not None:
                    d = min(ema_decay, step / (step + 10))
                    with torch.no_grad():
                        for pname, p in model.named_parameters():
                            if p.requires_grad:
                                ema_state[pname].mul_(d).add_(p.detach(), alpha=1 - d)
            ep_loss += loss.item() * tc["grad_accum"]; n += 1
        history.append(round(ep_loss / max(n, 1), 4))
        log(f"  epoch {epoch + 1}/{tc['epochs']}  loss {history[-1]:.4f}")

    if ema_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema_state:
                    p.data.copy_(ema_state[n].to(p.dtype))

    out_dir = Path(out_dir)
    model.save_pretrained(str(out_dir / "adapter"))
    processor.save_pretrained(str(out_dir / "processor"))
    stats = {
        "base_model": cfg["base_model"], "method": "LoRA (decoder)", "lora": lc,
        "hyperparameters": tc, "trainable_params": trainable, "total_params": total,
        "frozen_params": total - trainable, "n_corrections": len(train_rows),
        "n_replay_per_epoch": n_anchor, "effective_lr": lr, "distill_weight": distill_w,
        "ema_decay": ema_decay, "lang_loss_weight": lang_weight,
        "balance_batches": balance, "batches": {str(k): len(v) for k, v in by_batch.items()},
        "steps": step, "loss_per_epoch": history,
        "train_seconds": round(time.time() - t0, 1),
        "parent_adapter": str(parent_adapter) if parent_adapter else None,
        "device": device, "dtype": str(dtype),
    }
    del model, encoder, opt
    torch.cuda.empty_cache()
    return stats
