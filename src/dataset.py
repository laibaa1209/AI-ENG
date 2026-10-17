"""Tiny Shakespeare loading + next-token batches.

Source: the raw file behind the Hugging Face `tiny_shakespeare` dataset card
(Karpathy's char-rnn input.txt, 1,115,394 characters, ~40k lines).
"""
import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")


def load_text(path=DATA_PATH):
    if not os.path.exists(path):
        print(f"downloading Tiny Shakespeare -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def train_val_split(ids, val_frac=0.1):
    """Contiguous split: the last 10% of the corpus is validation (never seen in training)."""
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


def get_batch(data, batch_size, context_length, device="cpu", generator=None):
    """Sample random contiguous windows.

    x = tokens[i : i+T]      (input)
    y = tokens[i+1 : i+T+1]  (target = the next token at every position)
    """
    ix = torch.randint(len(data) - context_length - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)
