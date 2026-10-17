"""Figure 7: attention maps of the TRAINED model on a real prompt.

    python -m experiments.attention_visualization --prompt "ROMEO:\nWhat light"
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils import DEFAULT_CHECKPOINT, ROOT, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--prompt", default="ROMEO:\nWhat light through yonder window breaks?")
    args = ap.parse_args()

    model, tok, _ = load_checkpoint(args.checkpoint)
    idx = torch.tensor([tok.encode(args.prompt)])
    with torch.no_grad():
        model(idx)
    labels = [c if c != "\n" else "\\n" for c in args.prompt]
    T = len(labels)

    layers = [0, len(model.blocks) - 1]
    H = model.cfg.n_heads
    fig, axes = plt.subplots(len(layers), H, figsize=(4.2 * H, 4.4 * len(layers)))
    for r, li in enumerate(layers):
        w = model.blocks[li].attn.last_weights[0]  # (h, T, T)
        for h in range(H):
            ax = axes[r, h]
            ax.imshow(w[h], cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"layer {li} head {h}")
            ax.set_xticks(range(T), labels, fontsize=6)
            ax.set_yticks(range(T), labels, fontsize=6)
            if h == 0:
                ax.set_ylabel("query position")
            if r == len(layers) - 1:
                ax.set_xlabel("key position")
    fig.suptitle(f"Attention weights of the trained MiniGPT on {args.prompt!r} (upper triangle = 0: causal)")
    fig.tight_layout(h_pad=2.5)
    path = os.path.join(ROOT, "outputs", "figures", "fig7_attention_trained.png")
    fig.savefig(path, dpi=150)

    # simple quantitative summary per head: mean attention distance and weight on the previous token
    print(f"{'layer':>5} {'head':>4} {'mean distance':>14} {'w(prev tok)':>12} {'w(self)':>8} {'w(first tok)':>12}")
    pos = torch.arange(T)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
    for li, block in enumerate(model.blocks):
        w = block.attn.last_weights[0]
        for h in range(H):
            md = (w[h] * dist).sum(-1).mean().item()
            prev = w[h].diagonal(-1).mean().item()
            self_ = w[h].diagonal().mean().item()
            first = w[h][1:, 0].mean().item()
            print(f"{li:>5} {h:>4} {md:>14.2f} {prev:>12.3f} {self_:>8.3f} {first:>12.3f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
