"""Figure 4: sinusoidal positional encoding.

    python -m experiments.positional_encoding_viz

(a) the PE matrix itself, (b) a few dimensions as waves of different frequency,
(c) cosine similarity between positions: depends only on the distance |i - j|,
    which is what lets attention recover relative order.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.positional_encoding import sinusoidal_table
from src.utils import ROOT


def main():
    pe = sinusoidal_table(128, 128)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im = axes[0].imshow(pe, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_title("(a) PE[pos, dim]")
    axes[0].set_xlabel("embedding dimension")
    axes[0].set_ylabel("position")
    fig.colorbar(im, ax=axes[0])

    for d in (16, 32, 48, 64, 96):
        axes[1].plot(pe[:, d], label=f"dim {d}")
    axes[1].set_title("(b) each dimension is a sinusoid of a different wavelength")
    axes[1].set_xlabel("position")
    axes[1].legend(fontsize=8)

    unit = pe / pe.norm(dim=1, keepdim=True)
    sim = unit @ unit.T
    im = axes[2].imshow(sim, cmap="viridis")
    axes[2].set_title("(c) cosine similarity between positions")
    axes[2].set_xlabel("position j")
    axes[2].set_ylabel("position i")
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    path = os.path.join(ROOT, "outputs", "figures", "fig4_positional_encoding.png")
    fig.savefig(path, dpi=150)
    # numeric evidence: similarity falls with distance and is translation invariant
    print("cos-sim(10, 10+k) for k=0,1,2,5,20:", [round(sim[10, 10 + k].item(), 3) for k in (0, 1, 2, 5, 20)])
    print("cos-sim(50, 50+k) for k=0,1,2,5,20:", [round(sim[50, 50 + k].item(), 3) for k in (0, 1, 2, 5, 20)])
    print(f"saved {path}")


if __name__ == "__main__":
    main()
