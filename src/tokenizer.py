"""Deterministic character-level tokenizer.

Policy
------
* Vocabulary = sorted set of unique characters in the training corpus
  (65 for Tiny Shakespeare). Sorting makes the id assignment deterministic.
* No special tokens. The corpus is one continuous stream that we cut into
  fixed-length windows, so we never need <pad>, <bos> or <eos>.
* Characters outside the vocabulary raise a ValueError (the API turns this
  into an HTTP 400) instead of being silently dropped.
"""


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        unknown = sorted({c for c in s if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown!r}")
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)
