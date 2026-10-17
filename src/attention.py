"""Scaled dot-product attention and multi-head attention, written out explicitly.

Mask convention used everywhere in this repo: True = "may attend", False = blocked.
"""
import math

import torch
import torch.nn as nn


def causal_mask(T, device=None):
    """Lower-triangular (T, T) boolean mask. Row i may attend to columns 0..i only."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q, k, v, mask=None):
    """q, k: (..., T, d_k), v: (..., T, d_v) -> output (..., T, d_v), weights (..., T, T).

    scores  = Q K^T / sqrt(d_k)
    scores  = -inf where mask is False   (future tokens for a causal mask)
    weights = softmax(scores, dim=-1)    (each row sums to 1)
    output  = weights V
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, causal=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.last_weights = None  # (B, h, T, T), kept for visualisation

    def split_heads(self, x):
        """(B, T, d_model) -> (B, h, T, d_head)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x):
        """(B, h, T, d_head) -> (B, T, d_model)  (concatenation of the heads)"""
        B, h, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d)

    def forward(self, x, trace=False):
        B, T, C = x.shape
        log = print if trace else (lambda *a: None)
        log(f"input x                     {tuple(x.shape)}  (B, T, d_model)")

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        log(f"Q, K, V projections         {tuple(q.shape)} each")

        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        log(f"split heads                 {tuple(q.shape)}  (B, h, T, d_head)")

        mask = causal_mask(T, x.device) if self.causal else None
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        log(f"scores / weights per head   {tuple(weights.shape)}  (B, h, T, T)")
        log(f"per-head output             {tuple(out.shape)}  (B, h, T, d_head)")
        self.last_weights = weights.detach()

        # ponytail: dropout on the per-head outputs rather than on the weights, so the pure
        # attention function above stays untouched; both are just regularisers.
        out = self.attn_dropout(out)
        out = self.merge_heads(out)
        log(f"merge (concat) heads        {tuple(out.shape)}  (B, T, h*d_head)")

        out = self.resid_dropout(self.out_proj(out))
        log(f"output projection W_O       {tuple(out.shape)}  (B, T, d_model)")
        return out
