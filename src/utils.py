import os
import random

import numpy as np
import torch
import yaml

from .gpt import GPT, GPTConfig
from .tokenizer import CharTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CHECKPOINT = os.path.join(ROOT, "checkpoints", "tiny_gpt.pt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(path, model, tokenizer, extra=None):
    torch.save(
        {"model": model.state_dict(), "config": model.cfg.to_dict(), "chars": tokenizer.chars, "extra": extra or {}},
        path,
    )


def load_checkpoint(path=DEFAULT_CHECKPOINT, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer(ckpt["chars"]), ckpt.get("extra", {})


def model_summary(model):
    c = model.cfg
    lines = [
        "MiniGPT architecture",
        f"  vocab_size      {c.vocab_size}",
        f"  context_length  {c.context_length}",
        f"  d_model         {c.d_model}",
        f"  n_heads         {c.n_heads}  (d_head = {c.d_model // c.n_heads})",
        f"  n_layers        {c.n_layers}",
        f"  ffn_dim         {c.ffn_dim}",
        f"  dropout         {c.dropout}",
        f"  pos_encoding    {c.pos_encoding}",
        "",
        f"  {'module':<28}{'params':>12}",
    ]
    groups = {"tok_emb": 0, "pos (learned only)": 0, "blocks": 0, "ln_f": 0, "lm_head": 0}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        key = "pos (learned only)" if key == "pos" else key
        groups[key] += p.numel()
    per_block = groups["blocks"] // c.n_layers
    for k, v in groups.items():
        lines.append(f"  {k:<28}{v:>12,}")
        if k == "blocks":
            lines.append(f"    {'per block':<26}{per_block:>12,}")
    lines.append(f"  {'TOTAL':<28}{model.num_params():>12,}  ({model.num_params() / 1e6:.2f}M)")
    return "\n".join(lines)


def save_text_figure(text, path, title=None):
    """Render terminal-style text into a PNG so it can be dropped into the report as evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = text.splitlines()
    width = max(len(l) for l in lines + [title or ""])
    fig = plt.figure(figsize=(max(6, width * 0.1), 0.6 + 0.2 * (len(lines) + (2 if title else 0))))
    fig.patch.set_facecolor("#1e1e1e")
    body = (f"{title}\n\n" if title else "") + text
    fig.text(0.02, 0.98, body, family="monospace", fontsize=10, color="#d4d4d4", va="top")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
