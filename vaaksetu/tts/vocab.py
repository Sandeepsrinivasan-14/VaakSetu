"""Vocabulary extension for MMS-TTS (VITS) character tokenizers.

The base tokenizer silently *drops* any character outside its vocabulary, so
"α", "₹", "°", missing digits or Latin letters inside Tamil/Hindi text are
never spoken.  We add each such character (and digits, which the base models
read poorly) as K new sub-token slots with new, trainable embedding rows;
fine-tuning then teaches the model how to pronounce them in context.
Nothing is replaced in the text at inference time.
"""
from __future__ import annotations

import torch

from ..text import nfc


def _norm(tokenizer, text: str) -> str:
    text = nfc(text)
    return tokenizer.normalize_text(text) if tokenizer.normalize else text


def unknown_chars(tokenizer, texts: list[str], skip: str) -> list[str]:
    found: list[str] = []
    for t in texts:
        for ch in _norm(tokenizer, t):
            if ch not in tokenizer.encoder and ch not in skip and not ch.isspace() and ch not in found:
                found.append(ch)
    return found


def dropped_fraction(tokenizer, text: str, skip: str) -> float:
    """Fraction of content characters the tokenizer would silently drop."""
    content = [c for c in _norm(tokenizer, text) if not c.isspace() and c not in skip]
    if not content:
        return 0.0
    return sum(c not in tokenizer.encoder for c in content) / len(content)


def extend_vocab(model, tokenizer, chars: list[str], slots: int, seed: int = 0) -> list[str]:
    """Give every char in `chars` `slots` learnable sub-tokens (c, c#1, ...).

    Existing base tokens keep their embedding (slot 0); only missing tokens are
    appended.  Returns the list of chars whose slot layout changed."""
    changed, new_tokens = [], []
    for c in chars:
        if tokenizer.char_slots.get(c, 1) >= slots:
            continue
        changed.append(c)
        tokenizer.char_slots[c] = slots
        for tok in [c] + [f"{c}#{j}" for j in range(1, slots)]:
            if tok not in tokenizer.encoder:
                new_tokens.append(tok)
    if not new_tokens:
        return changed
    for tok in new_tokens:
        idx = len(tokenizer.encoder)
        tokenizer.encoder[tok] = idx
        tokenizer.decoder[idx] = tok
    tokenizer.init_kwargs["char_slots"] = tokenizer.char_slots

    emb: torch.nn.Embedding = model.text_encoder.embed_tokens
    old = emb.weight.data
    g = torch.Generator().manual_seed(seed)
    mean, std = old.mean(0, keepdim=True).cpu(), old.std(0, keepdim=True).cpu()
    extra = (mean + 0.5 * std * torch.randn(len(new_tokens), old.size(1), generator=g)).to(old)
    new = torch.nn.Embedding(old.size(0) + len(new_tokens), old.size(1)).to(old.device, old.dtype)
    new.weight.data = torch.cat([old, extra], 0)
    model.text_encoder.embed_tokens = new
    model.config.vocab_size = new.num_embeddings
    return changed
