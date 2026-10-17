"""Figure 1 + Figure 2: the 3-token hand-calculation, step by step, checked against src/attention.py.

    python -m experiments.hand_attention
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention import causal_mask, scaled_dot_product_attention
from src.utils import ROOT, save_text_figure

Q = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 4.0]])


def fmt(m):
    return "\n".join("   [" + "  ".join(f"{v:8.4f}" if v != float("-inf") else "    -inf" for v in row) + " ]" for row in m.tolist())


def main():
    torch.set_printoptions(precision=4)
    out = []
    p = out.append
    d_k = Q.shape[1]
    p(f"3 tokens, d_k = d_v = {d_k}")
    p(f"Q =\n{fmt(Q)}\nK =\n{fmt(K)}\nV =\n{fmt(V)}")

    s = Q @ K.T
    p(f"\nStep 2: Q K^T =\n{fmt(s)}")
    s = s / math.sqrt(d_k)
    p(f"\nStep 3: divide by sqrt(d_k) = {math.sqrt(d_k):.4f}\n{fmt(s)}")
    w = torch.softmax(s, dim=-1)
    p(f"\nStep 4: row-wise softmax (rows sum to {w.sum(-1).tolist()})\n{fmt(w)}")
    o = w @ V
    p(f"\nStep 5: output = weights @ V\n{fmt(o)}")

    mask = causal_mask(3)
    sm = s.masked_fill(~mask, float("-inf"))
    p(f"\nStep 6: causal mask (1 = may attend)\n{fmt(mask.float())}\nmasked scores\n{fmt(sm)}")
    wm = torch.softmax(sm, dim=-1)
    p(f"masked weights\n{fmt(wm)}")
    om = wm @ V
    p(f"masked output\n{fmt(om)}")

    ref_o, ref_w = scaled_dot_product_attention(Q, K, V)
    ref_om, _ = scaled_dot_product_attention(Q, K, V, mask)
    ok = torch.allclose(o, ref_o) and torch.allclose(om, ref_om) and torch.allclose(w, ref_w)
    p(f"\nsrc.attention.scaled_dot_product_attention agrees with step-by-step result: {ok}")
    text = "\n".join(out)
    print(text)

    with open(os.path.join(ROOT, "outputs", "logs", "hand_attention.txt"), "w") as f:
        f.write(text + "\n")
    save_text_figure(text, os.path.join(ROOT, "outputs", "figures", "fig1_hand_attention.png"), "Figure 1 - hand-calculated attention (python -m experiments.hand_attention)")

    # Figure 2: mask + weights heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = ["tok1", "tok2", "tok3"]
    for ax, m, title, cmap in [
        (axes[0], mask.float(), "Causal mask (1 = allowed)", "Greys"),
        (axes[1], w, "Weights, no mask", "Blues"),
        (axes[2], wm, "Weights, causal mask", "Blues"),
    ]:
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(m[i, j])}" if cmap == "Greys" else f"{m[i, j]:.3f}",
                        ha="center", va="center", color="white" if m[i, j] > 0.6 else "black")
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("key (attended to)")
        ax.set_ylabel("query (attending)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig2_causal_mask_hand.png"), dpi=150)


if __name__ == "__main__":
    main()
