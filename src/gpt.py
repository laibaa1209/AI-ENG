"""MiniGPT: token embedding + position + N causal Transformer blocks + LM head."""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding
from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int = 65
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"  # or "learned"
    causal: bool = True  # False only for the MLM-vs-CLM leakage experiment

    def to_dict(self):
        return asdict(self)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # nn.Embedding default init is N(0, 1): same scale as the sinusoids (|PE| <= 1),
        # so neither the token identity nor the position drowns the other.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        pos_cls = SinusoidalPositionalEncoding if cfg.pos_encoding == "sinusoidal" else LearnedPositionalEncoding
        self.pos = pos_cls(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout, cfg.causal) for _ in range(cfg.n_layers)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, trace=False):
        """idx: (B, T) token ids -> logits (B, T, vocab). If targets given, also mean CE loss."""
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"sequence length {T} > context_length {self.cfg.context_length}"
        x = self.drop(self.pos(self.tok_emb(idx)))  # (B, T, d_model)
        for i, block in enumerate(self.blocks):
            x = block(x, trace=trace and i == 0)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab)
        loss = None
        if targets is not None:
            # targets are already shifted by one (see dataset.get_batch): position t predicts token t+1
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, greedy=False):
        """Autoregressive sampling: feed the last <= context_length tokens, sample one, append, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # only the last position's prediction is needed
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
