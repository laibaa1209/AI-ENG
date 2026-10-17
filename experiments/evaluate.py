"""Quantitative + qualitative evaluation of the final checkpoint.

    python -m experiments.evaluate

* validation loss / perplexity over the WHOLE validation split (non-overlapping windows), not a random estimate
* 5 fixed prompts, temperature 0.8 (seeded)                         -> Figure 8
* greedy vs temperature sampling: repetition + diversity statistics (optional experiment)
"""
import math
import os

import torch
import torch.nn.functional as F

from src.dataset import load_text, train_val_split
from src.generate import generate_text
from src.utils import DEFAULT_CHECKPOINT, ROOT, load_checkpoint, save_text_figure, set_seed

PROMPTS = ["ROMEO:", "JULIET:\nO ", "KING HENRY:\nWhat news", "First Citizen:\n", "To be, or not to be"]


@torch.no_grad()
def full_val_loss(model, data, T):
    n = (len(data) - 1) // T
    x = data[: n * T].view(n, T)
    y = data[1 : n * T + 1].view(n, T)
    total = 0.0
    for i in range(0, n, 64):
        logits, _ = model(x[i : i + 64])
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i : i + 64].reshape(-1), reduction="sum").item()
    return total / (n * T)


def distinct_ngrams(text, n=4):
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def longest_repeat(text, min_len=8):
    """Fraction of the text covered by a substring that already occurred earlier (crude loop detector)."""
    covered = 0
    for i in range(min_len, len(text)):
        if text[i - min_len : i] in text[: i - min_len]:
            covered += 1
    return covered / max(1, len(text))


def main(checkpoint=DEFAULT_CHECKPOINT):
    model, tok, extra = load_checkpoint(checkpoint)
    _, val = train_val_split(tok.encode(load_text()))
    vl = full_val_loss(model, val, model.cfg.context_length)
    lines = [f"checkpoint: {os.path.relpath(checkpoint, ROOT)}  ({model.num_params():,} params)",
             f"full validation set: {len(val):,} chars, loss {vl:.4f} nats/char, perplexity {math.exp(vl):.3f}, "
             f"{vl / math.log(2):.3f} bits/char", ""]

    samples = []
    for i, p in enumerate(PROMPTS):
        set_seed(i)
        s = generate_text(model, tok, p, 250, temperature=0.8)
        samples.append(f"### Prompt {i + 1}: {p!r}  (temperature 0.8, seed {i})\n{s}\n")
    lines += samples

    lines.append("### Greedy vs temperature sampling (prompt 'ROMEO:', 400 new chars, mean over 3 seeds where stochastic)")
    lines.append(f"{'strategy':<22}{'distinct 4-grams':>18}{'repeated-text frac':>20}")
    strategies = [("greedy", dict(greedy=True)), ("temperature 0.5", dict(temperature=0.5)),
                  ("temperature 0.8", dict(temperature=0.8)), ("temperature 1.0", dict(temperature=1.0)),
                  ("temperature 1.5", dict(temperature=1.5))]
    examples = []
    for name, kw in strategies:
        d, r = [], []
        for seed in range(1 if kw.get("greedy") else 3):
            set_seed(seed)
            s = generate_text(model, tok, "ROMEO:", 400, **kw)[len("ROMEO:"):]
            d.append(distinct_ngrams(s))
            r.append(longest_repeat(s))
            if seed == 0:
                examples.append(f"--- {name} ---\n{s[:300]}\n")
        lines.append(f"{name:<22}{sum(d) / len(d):>18.3f}{sum(r) / len(r):>20.3f}")
    lines.append("")
    lines += examples

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(ROOT, "outputs", "samples", "evaluation.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    save_text_figure("\n".join([lines[1], ""] + samples), os.path.join(ROOT, "outputs", "figures", "fig8_generation_samples.png"),
                     "Figure 8 - generation samples from fixed prompts (python -m experiments.evaluate)")


if __name__ == "__main__":
    main()
