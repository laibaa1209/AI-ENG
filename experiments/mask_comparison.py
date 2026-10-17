"""Lesson 06: why BERT (bidirectional, masked LM) and GPT (causal LM) need different attention constraints.

    python -m experiments.mask_comparison

Three tiny models, identical architecture/budget, only the mask + objective differ:
  A. GPT  : causal mask,        predict token t+1 at every position          (honest generator)
  B. LEAK : bidirectional mask, predict token t+1 at every position          (the bug BERT-style attention causes in a LM)
  C. BERT : bidirectional mask, 15% of inputs -> [MASK], predict those only  (MLM)

Measurements
  * training loss on its own objective
  * "last-position" next-token loss: the prediction at the final position, where no future token
    exists in the input -- exactly the situation during generation.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention import causal_mask
from src.dataset import get_batch, load_text, train_val_split
from src.gpt import GPT, GPTConfig
from src.tokenizer import CharTokenizer
from src.utils import ROOT, save_text_figure, set_seed

T, B, STEPS = 64, 32, 600
SMALL = dict(context_length=T, d_model=64, n_heads=4, n_layers=2, ffn_dim=256, dropout=0.0)


def mlm_batch(x, mask_id, p=0.15):
    """BERT-style: choose 15% of positions; input becomes [MASK] there, targets are -100 elsewhere (ignored by CE)."""
    chosen = torch.rand(x.shape) < p
    inp = x.masked_fill(chosen, mask_id)
    tgt = x.masked_fill(~chosen, -100)
    return inp, tgt


def train_model(kind, data, vocab):
    set_seed(0)
    model = GPT(GPTConfig(vocab_size=vocab + (kind == "BERT"), causal=(kind == "GPT"), **SMALL))  # BERT: +1 slot for [MASK]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    hist = []
    for step in range(STEPS):
        x, y = get_batch(data, B, T)
        if kind == "BERT":
            x, y = mlm_batch(x, mask_id=vocab)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return model.eval(), hist


@torch.no_grad()
def last_position_loss(model, data, n=40):
    g = torch.Generator().manual_seed(7)
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(data, B, T, generator=g)
        logits, _ = model(x)
        tot += torch.nn.functional.cross_entropy(logits[:, -1], y[:, -1]).item()
    return tot / n


def main():
    text = load_text()
    tok = CharTokenizer.from_text(text)
    train, val = train_val_split(tok.encode(text))

    results, hists, models = {}, {}, {}
    for kind in ("GPT", "LEAK", "BERT"):
        print(f"training {kind} ...", flush=True)
        model, hist = train_model(kind, train, tok.vocab_size)
        models[kind], hists[kind] = model, hist
        results[kind] = {"final_train_loss": sum(hist[-50:]) / 50, "last_pos_val_loss": last_position_loss(model, val)}

    set_seed(0)
    prompt = torch.tensor([tok.encode("ROMEO:")])
    samples = {k: tok.decode(models[k].generate(prompt, 120, temperature=0.8)[0].tolist()) for k in ("GPT", "LEAK")}

    lines = ["Same tiny Transformer (2 layers, d_model 64, 600 steps); only mask + objective differ.", "",
             f"{'model':<6}{'mask':<15}{'objective':<24}{'train loss (own obj.)':>22}{'next-token loss @ last pos (val)':>36}"]
    desc = {"GPT": ("causal", "next token, all pos."), "LEAK": ("bidirectional", "next token, all pos."), "BERT": ("bidirectional", "15% masked tokens")}
    for k, r in results.items():
        lp = f"{r['last_pos_val_loss']:.3f}" if k != "BERT" else "n/a (not a next-token model)"
        lines.append(f"{k:<6}{desc[k][0]:<15}{desc[k][1]:<24}{r['final_train_loss']:>22.3f}{lp:>36}")
    lines += ["", f"uniform-guess loss = ln(65) = {math.log(65):.3f}", "",
              "GPT  sample: " + repr(samples["GPT"]), "LEAK sample: " + repr(samples["LEAK"])]
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(ROOT, "outputs", "logs", "mask_comparison.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    save_text_figure(out, os.path.join(ROOT, "outputs", "figures", "mlm_vs_clm_results.png"), "MLM vs CLM comparison (python -m experiments.mask_comparison)")

    # figure: the two mask matrices + an MLM example + loss curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    n = 8
    word = "To be or"
    axes[0].imshow(causal_mask(n).float(), cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("GPT: causal mask\nrow t sees columns <= t")
    axes[1].imshow(torch.ones(n, n), cmap="Greys", vmin=0, vmax=1)
    masked = [c if i not in (3, 6) else "[M]" for i, c in enumerate(word)]
    axes[1].set_title("BERT: bidirectional (no mask)\ninput positions 3,6 replaced by [MASK]")
    for ax, xl in ((axes[0], list(word)), (axes[1], masked)):
        ax.set_xticks(range(n), xl)
        ax.set_yticks(range(n), xl)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
    for k, h in hists.items():
        sm = [sum(h[max(0, i - 20) : i + 1]) / len(h[max(0, i - 20) : i + 1]) for i in range(len(h))]
        axes[2].plot(sm, label=f"{k} ({desc[k][0]}, {desc[k][1]})")
    axes[2].axhline(math.log(65), color="gray", ls=":", label="uniform guess")
    axes[2].set_title("training loss on each model's own objective")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("cross-entropy")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "outputs", "figures", "mlm_vs_clm_masks.png"), dpi=150)


if __name__ == "__main__":
    main()
