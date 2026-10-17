"""Decoder-style (GPT) Transformer block, pre-LayerNorm variant.

    x -> x + MHA(LN1(x))      attention sublayer with residual
      -> x + FFN(LN2(x))      feed-forward sublayer with residual

Pre-LN (GPT-2) instead of the original post-LN: the residual stream is never
normalised, so gradients flow straight through the identity path and training
is stable without a long warm-up.
"""
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.0, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, trace=False):
        x = x + self.attn(self.ln1(x), trace=trace)
        x = x + self.ffn(self.ln2(x))
        return x
