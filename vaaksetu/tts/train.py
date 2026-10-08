"""Partial fine-tuning of MMS-TTS (VITS) from pronunciation corrections.

What is trained
  text_encoder        (embeddings incl. newly added symbol tokens, transformer, prior projection)
  duration_predictor  (stochastic duration predictor)
What is frozen
  posterior_encoder, flow, decoder (HiFi-GAN)  -> the voice/timbre cannot drift.

Objective (the VITS prior-side losses, no discriminator needed):
  1. The frozen posterior encoder + flow map the *target* speech to latents z_p.
  2. Monotonic Alignment Search (MAS) aligns text tokens to z_p frames.
  3. KL(posterior || text prior) trains the text encoder to predict the right
     acoustic latents for each character *in context*.
  4. The duration predictor learns the aligned durations (NLL).

Targets come from the correction: either the human's recording or - to keep
the voice identical - the frozen base model speaking the human-supplied
spoken form.  The model learns written form -> correct speech, so at inference
the raw text goes straight in: no substitution rules are applied.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numba
import numpy as np
import torch

from ..audio import load_audio
from ..config import LANGS, load_config
from ..text import nfc
from .engine import load_tts, save_tts
from .vocab import extend_vocab, unknown_chars


# --------------------------------------------------------------------------- MAS
@numba.njit(cache=True)
def _mas_single(value: np.ndarray, t_y: int, t_x: int) -> np.ndarray:
    """Viterbi-style monotonic alignment (from the VITS reference implementation).
    value: [t_y (frames), t_x (tokens)] log-likelihoods."""
    path = np.zeros((t_y, t_x), dtype=np.float32)
    v = value.copy()
    neg = -1e9
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = neg if x == y else v[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else neg
            else:
                v_prev = v[y - 1, x - 1]
            v[y, x] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1.0
        if index != 0 and (index == y or v[y - 1, index] < v[y - 1, index - 1]):
            index -= 1
    return path


def maximum_path(neg_cent: torch.Tensor, frame_lens: torch.Tensor, text_lens: torch.Tensor) -> torch.Tensor:
    """neg_cent: [B, T_frames, T_text] -> hard alignment of same shape."""
    nc = neg_cent.detach().float().cpu().numpy()
    out = np.zeros_like(nc)
    for b in range(nc.shape[0]):
        ty, tx = int(frame_lens[b]), int(text_lens[b])
        out[b, :ty, :tx] = _mas_single(np.ascontiguousarray(nc[b, :ty, :tx]), ty, tx)
    return torch.from_numpy(out).to(neg_cent.device)


# ------------------------------------------------------------------ spectrogram
def linear_spectrogram(wav: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """VITS linear magnitude spectrogram. wav [B, S] -> [B, n_fft//2+1, S//hop]."""
    pad = (n_fft - hop) // 2
    wav = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(wav, n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft, device=wav.device),
                      center=False, return_complex=True)
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-6)


# ----------------------------------------------------------------------- losses
def vits_prior_losses(model, input_ids, text_mask, spec, spec_mask):
    """Returns (kl_loss, duration_loss) for a batch."""
    te = model.text_encoder(input_ids=input_ids, padding_mask=text_mask.unsqueeze(-1).float(),
                            attention_mask=text_mask)
    h = te.last_hidden_state.transpose(1, 2)                   # [B, C, Tx]
    m_p = te.prior_means.transpose(1, 2)                       # [B, C, Tx]
    logs_p = te.prior_log_variances.transpose(1, 2)            # log-std, [B, C, Tx]
    x_mask = text_mask.unsqueeze(1).float()                    # [B, 1, Tx]
    y_mask = spec_mask.unsqueeze(1).float()                    # [B, 1, Ty]

    with torch.no_grad():
        # Posterior *mean* as the target latent: the posterior is frozen, so
        # sampling would only add gradient noise on a tiny dataset.
        _, m_q, logs_q = model.posterior_encoder(spec, y_mask)
        z_p = model.flow(m_q * y_mask, y_mask, reverse=False)  # [B, C, Ty]

        s_p_sq_r = torch.exp(-2 * logs_p)
        n1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
        n2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
        n3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
        n4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
        neg_cent = n1 + n2 + n3 + n4                           # [B, Ty, Tx]
        attn = maximum_path(neg_cent, spec_mask.sum(1), text_mask.sum(1))

    durations = attn.sum(1).unsqueeze(1)                       # [B, 1, Tx]
    nll = model.duration_predictor(h, x_mask, None, durations=durations, reverse=False)
    dur_loss = torch.sum(nll) / torch.sum(x_mask)

    m_exp = torch.matmul(attn, m_p.transpose(1, 2)).transpose(1, 2)        # [B, C, Ty]
    logs_exp = torch.matmul(attn, logs_p.transpose(1, 2)).transpose(1, 2)
    # Closed-form KL(q || p) between Gaussians.  Because z_p is the posterior
    # *mean* (not a sample), the posterior variance exp(2*logs_q) must appear
    # explicitly; without it the prior std can collapse towards 0 and the loss
    # decreases without bound, wrecking the text encoder.  (The flow is
    # volume-preserving, so logs_q carries over to z_p-space as in VITS.)
    kl = (logs_exp - logs_q - 0.5
          + 0.5 * (torch.exp(2 * logs_q) + (z_p - m_exp) ** 2) * torch.exp(-2 * logs_exp))
    kl_loss = torch.sum(kl * y_mask) / torch.sum(y_mask)
    return kl_loss, dur_loss


# ----------------------------------------------------------------------- train
def _base_vocab_size(lang: str) -> int:
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(LANGS[lang]["mms"]).vocab_size


def train_tts(lang: str, train_rows: list[dict], anchor_rows: list[dict], out_dir: Path,
              parent_version: str = "base", cfg: dict | None = None, log=print) -> dict:
    """Each row: {"text": written input, "audio_path": target speech}."""
    cfg = cfg or load_config("tts")
    tc = cfg["train"]
    random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    model, tok = load_tts(lang, parent_version, device)
    texts = [nfc(r["text"]) for r in train_rows + anchor_rows]
    targets = unknown_chars(tok, texts, cfg["vocab_skip_chars"])
    targets += [d for d in cfg["slot_chars"] if any(d in t for t in texts) and d not in targets]
    added = extend_vocab(model, tok, targets, slots=cfg["slots_per_char"], seed=tc["seed"])
    if added:
        log(f"  vocab: {len(added)} chars now learnable x{cfg['slots_per_char']} slots: {' '.join(added)}")
    model.to(device)

    for name, module in model.named_children():
        module.requires_grad_(name in cfg["trainable_modules"])
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    log(f"  trainable {trainable/1e6:.2f}M / total {total/1e6:.1f}M params")

    # Embedding rows of the *original* vocabulary stay frozen (gradient mask),
    # so text made only of known characters keeps its base-model embeddings.
    emb = model.text_encoder.embed_tokens.weight
    n_base = int(tc.get("n_frozen_rows") or 0) or _base_vocab_size(lang)
    row_mask = torch.zeros(emb.size(0), 1, device=emb.device)
    row_mask[n_base:] = 1.0
    emb.register_hook(lambda g: g * row_mask)
    groups = [
        {"params": [emb], "lr": tc["lr_new_embeddings"], "weight_decay": 0.0},
        {"params": [p for n, p in model.text_encoder.named_parameters()
                    if p.requires_grad and not n.startswith("embed_tokens")],
         "lr": tc["lr_text_encoder"], "weight_decay": 0.0},
    ]
    if "duration_predictor" in cfg["trainable_modules"] and tc.get("lr_duration", 0) > 0:
        groups.append({"params": list(model.duration_predictor.parameters()),
                       "lr": tc["lr_duration"], "weight_decay": 0.0})
    model.train()
    for name in cfg["frozen_modules"]:
        getattr(model, name).eval()
    if not tc.get("dropout", True):
        # Fine-tuning on a few dozen sentences: disable dropout *and* layerdrop
        # (MMS checkpoints ship layerdrop=0.1, which skips whole transformer
        # layers at random in train mode).  Gradients still flow in eval mode.
        model.eval()

    n_anchor = min(len(anchor_rows), math.ceil(tc["replay_ratio"] * len(train_rows)))
    def encode(rows):
        return [(tok(nfc(r["text"]), return_tensors="pt")["input_ids"][0],
                 torch.from_numpy(load_audio(r["audio_path"], sr=model.config.sampling_rate)), None)
                for r in rows]
    corr, anch = encode(train_rows), encode(anchor_rows)

    opt = torch.optim.AdamW(groups, betas=tuple(tc["betas"]), eps=1e-9)

    # EMA of the trainable weights, applied before saving.  Warmup-corrected
    # decay (see stt/train.py for the same idea) so a 60-epoch run on a few
    # dozen sentences still benefits instead of tracking the random init.
    ema_decay = float(tc.get("ema_decay", 0.0) or 0.0)
    ema_state = {n: p.detach().clone() for n, p in model.named_parameters()
                if p.requires_grad} if ema_decay > 0 else None

    history = []
    step = 0
    for epoch in range(tc["epochs"]):
        rows = corr + random.sample(anch, n_anchor) if anch else list(corr)
        random.shuffle(rows)
        ep = [0.0, 0.0]; n = 0
        for b in range(0, len(rows), tc["batch_size"]):
            batch = rows[b:b + tc["batch_size"]]
            tx = max(len(x[0]) for x in batch)
            ids = torch.zeros(len(batch), tx, dtype=torch.long)
            tmask = torch.zeros(len(batch), tx, dtype=torch.long)
            S = max(len(x[1]) for x in batch)
            wav = torch.zeros(len(batch), S)
            flen = []
            for i, (t, w, _) in enumerate(batch):
                ids[i, :len(t)] = t; tmask[i, :len(t)] = 1; wav[i, :len(w)] = w
                flen.append(len(w) // 256)
            spec = linear_spectrogram(wav.to(device))[:, :, :max(flen)]
            smask = (torch.arange(spec.size(2))[None, :] < torch.tensor(flen)[:, None]).long().to(device)
            spec = spec * smask.unsqueeze(1)
            kl, dur = vits_prior_losses(model, ids.to(device), tmask.to(device), spec, smask)
            loss = kl + dur
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
            opt.step()
            step += 1
            if ema_state is not None:
                d = min(ema_decay, step / (step + 10))
                with torch.no_grad():
                    for pname, p in model.named_parameters():
                        if p.requires_grad:
                            ema_state[pname].mul_(d).add_(p.detach(), alpha=1 - d)
            ep[0] += kl.item(); ep[1] += dur.item(); n += 1
        history.append({"kl": round(ep[0] / n, 4), "dur": round(ep[1] / n, 4)})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"  epoch {epoch + 1}/{tc['epochs']}  kl {history[-1]['kl']:.4f}  dur {history[-1]['dur']:.4f}")

    if ema_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema_state:
                    p.data.copy_(ema_state[n].to(p.dtype))

    model.eval()
    save_tts(model, tok, out_dir)
    stats = {
        "base_model": LANGS[lang]["mms"], "parent_version": parent_version,
        "method": "partial fine-tune (text encoder incl. new symbol embeddings), prior KL with MAS",
        "trainable_modules": cfg["trainable_modules"], "frozen_modules": cfg["frozen_modules"],
        "hyperparameters": tc, "learned_chars": added, "char_slots": tok.char_slots,
        "vocab_size": len(tok.encoder), "ema_decay": ema_decay,
        "trainable_params": trainable, "total_params": total, "frozen_params": total - trainable,
        "n_corrections": len(train_rows), "n_replay_per_epoch": n_anchor,
        "loss_per_epoch": history, "train_seconds": round(time.time() - t0, 1), "device": device,
    }
    del model, opt
    torch.cuda.empty_cache()
    return stats
