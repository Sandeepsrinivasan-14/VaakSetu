"""VITS character tokenizer with multi-slot tokens for learned symbols.

In VITS every input token has a single (constant) prior over the frames it is
aligned to, so one token cannot express a multi-phoneme word such as "alpha"
for "α".  Characters listed in `char_slots` are therefore emitted as K
sub-tokens  c, c#1, ..., c#(K-1),  each with its own learned embedding - the
same idea as sub-word tokenisation.  What each slot sounds like is learned by
the model; the tokenizer contains no pronunciation knowledge.

This file is copied next to every saved model so it can be loaded elsewhere
with `AutoTokenizer.from_pretrained(path, trust_remote_code=True)`.
"""
from transformers import VitsTokenizer


class VaakSetuVitsTokenizer(VitsTokenizer):
    def __init__(self, vocab_file, char_slots=None, **kwargs):
        self.char_slots = dict(char_slots or {})
        super().__init__(vocab_file, char_slots=self.char_slots, **kwargs)

    def _tokenize(self, text):
        tokens = []
        for c in text:
            tokens.append(c)
            tokens.extend(f"{c}#{j}" for j in range(1, self.char_slots.get(c, 1)))
        if self.add_blank:
            interspersed = [self._convert_id_to_token(0)] * (len(tokens) * 2 + 1)
            interspersed[1::2] = tokens
            tokens = interspersed
        return tokens

    def convert_tokens_to_string(self, tokens):
        if self.add_blank and len(tokens) > 1:
            tokens = tokens[1::2]
        return "".join(t for t in tokens if "#" not in t or len(t) == 1)
