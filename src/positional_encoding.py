"""Position information added to token embeddings."""
import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len, d_model):
    """PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
       PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))"""
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float) / d_model)  # (d/2,)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-trainable) encoding from Vaswani et al. (2017)."""

    def __init__(self, max_len, d_model):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.pe[: x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """One trainable vector per position (GPT-2 style); used for the ablation."""

    def __init__(self, max_len, d_model):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        return x + self.emb(torch.arange(x.size(1), device=x.device))
