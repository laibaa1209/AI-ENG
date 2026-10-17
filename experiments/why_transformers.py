"""Lesson 01: sequential depth of an RNN vs self-attention, measured.

    python -m experiments.why_transformers

An RNN must compute h_t = f(h_{t-1}, x_t) one step at a time -> T dependent steps (sequential depth O(T)).
Self-attention computes all T outputs with a few big matrix multiplies -> sequential depth O(1) per layer,
but materialises a (T x T) score matrix -> memory O(T^2).
We time one forward+backward pass of each layer type (same d_model, same batch) for growing T.
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.utils import ROOT

D, B = 128, 8


class LoopRNN(nn.Module):
    """Plain Elman RNN written as an explicit Python loop so the sequential dependency is visible."""

    def __init__(self, d):
        super().__init__()
        self.wx = nn.Linear(d, d)
        self.wh = nn.Linear(d, d, bias=False)

    def forward(self, x):
        h = torch.zeros(x.size(0), x.size(2))
        outs = []
        for t in range(x.size(1)):  # T dependent steps: step t cannot start before step t-1 finishes
            h = torch.tanh(self.wx(x[:, t]) + self.wh(h))
            outs.append(h)
        return torch.stack(outs, 1)


def bench(layer, T, reps=5):
    x = torch.randn(B, T, D, requires_grad=True)
    layer(x).sum().backward()  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        layer(x).sum().backward()
    return (time.perf_counter() - t) / reps * 1000


def main():
    torch.manual_seed(0)
    rnn, attn = LoopRNN(D), MultiHeadAttention(D, 4)
    rows = []
    print(f"{'T':>6}{'RNN seq. steps':>16}{'attn seq. steps':>17}{'RNN ms':>10}{'attn ms':>10}{'attn score MB':>15}")
    for T in (32, 64, 128, 256, 512, 1024):
        r, a = bench(rnn, T), bench(attn, T)
        mem = B * 4 * T * T * 4 / 2**20  # B * heads * T * T float32 scores
        rows.append((T, r, a, mem))
        print(f"{T:>6}{T:>16}{1:>17}{r:>10.1f}{a:>10.1f}{mem:>15.1f}")

    with open(os.path.join(ROOT, "outputs", "logs", "why_transformers.txt"), "w") as f:
        f.write("T, rnn_ms, attn_ms, attn_score_MB\n" + "\n".join(",".join(f"{v:.2f}" for v in row) for row in rows) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    Ts = [r[0] for r in rows]
    axes[0].plot(Ts, [r[1] for r in rows], "o-", label="RNN (loop over T)")
    axes[0].plot(Ts, [r[2] for r in rows], "s-", label="multi-head self-attention")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sequence length T")
    axes[0].set_ylabel("forward+backward time (ms)")
    axes[0].set_title(f"Wall-clock, CPU, B={B}, d_model={D}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(Ts, [r[3] for r in rows], "s-", color="#e8590c")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sequence length T")
    axes[1].set_ylabel("attention score memory (MB)")
    axes[1].set_title("Price of parallelism: O(T^2) memory")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "outputs", "figures", "why_transformers.png"), dpi=150)


if __name__ == "__main__":
    main()
