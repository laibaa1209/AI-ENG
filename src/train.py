"""Train MiniGPT on Tiny Shakespeare.

    python -m src.train --config configs/tiny_gpt.yaml
    python -m src.train --config configs/tiny_gpt.yaml --set n_layers=2 max_steps=1500 run_name=depth2
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import psutil
import torch
import yaml

from .dataset import get_batch, load_text, train_val_split
from .gpt import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .utils import ROOT, get_device, load_config, model_summary, save_checkpoint, set_seed

MODEL_KEYS = set(GPTConfig.__dataclass_fields__)


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        val = yaml.safe_load(v)  # "2" -> 2, "true" -> True
        if isinstance(val, str):  # YAML 1.1 reads "3e-05" (no dot) as a string
            try:
                val = float(val)
            except ValueError:
                pass
        out[k] = val
    return out


def lr_at(step, cfg):
    """Linear warm-up then cosine decay to min_lr."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def peak_rss_mb():
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):  # Windows
        return mi.peak_wset / 2**20
    import resource  # Linux / macOS

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


@torch.no_grad()
def estimate_loss(model, splits, cfg, device, iters):
    model.eval()
    g = torch.Generator().manual_seed(1234)  # same eval batches every time -> comparable numbers
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, cfg["batch_size"], cfg["context_length"], device, g))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(cfg):
    set_seed(cfg["seed"])
    device = cfg.get("device") or get_device()
    run = cfg["run_name"]
    log_path = os.path.join(ROOT, "outputs", "logs", f"{run}.jsonl")
    ckpt_path = os.path.join(ROOT, cfg.get("checkpoint_path") or f"checkpoints/{run}.pt")

    text = load_text()
    tok = CharTokenizer.from_text(text)
    train_data, val_data = train_val_split(tok.encode(text), cfg["val_frac"])
    if cfg.get("overfit_chars"):  # debugging gate: tiny subset, train == val
        train_data = val_data = train_data[: cfg["overfit_chars"]]

    mcfg = GPTConfig(vocab_size=tok.vocab_size, **{k: v for k, v in cfg.items() if k in MODEL_KEYS and k != "vocab_size"})
    model = GPT(mcfg).to(device)
    print(model_summary(model))
    print(f"device={device}  train_tokens={len(train_data):,}  val_tokens={len(val_data):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    open(log_path, "w").close()
    splits = {"train": train_data, "val": val_data}
    tokens_per_step = cfg["batch_size"] * cfg["context_length"]
    t0 = time.time()
    for step in range(cfg["max_steps"] + 1):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
            losses = estimate_loss(model, splits, cfg, device, cfg["eval_iters"])
            elapsed = time.time() - t0
            prompt = torch.tensor([tok.encode(cfg["sample_prompt"])], device=device)
            model.eval()
            sample = tok.decode(model.generate(prompt, cfg["sample_tokens"], temperature=0.8)[0].tolist())
            model.train()
            rec = {
                "step": step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
                "tokens_per_s": round(step * tokens_per_step / elapsed) if step else 0,
                "sample": sample,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:5d} | train {rec['train_loss']:.4f} | val {rec['val_loss']:.4f} | lr {lr:.2e} | {elapsed:6.1f}s")
            print(f"  sample: {sample[:120]!r}")

        if step == cfg["max_steps"]:
            break
        x, y = get_batch(train_data, cfg["batch_size"], cfg["context_length"], device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

    wall = time.time() - t0
    metrics = {
        "run_name": run,
        "params": model.num_params(),
        "final_train_loss": rec["train_loss"],
        "final_val_loss": rec["val_loss"],
        "val_perplexity": round(math.exp(rec["val_loss"]), 3),
        "train_time_s": round(wall, 1),
        "tokens_per_s": round(cfg["max_steps"] * tokens_per_step / wall),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else peak_rss_mb(), 1),
        "mem_kind": "cuda_allocated" if device == "cuda" else "process_peak_rss",
        "attn_scores_mb": round(cfg["n_layers"] * cfg["batch_size"] * cfg["n_heads"] * cfg["context_length"] ** 2 * 4 / 2**20, 2),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else f"CPU {platform.processor()} ({os.cpu_count()} threads)",
        "config": cfg,
    }
    save_checkpoint(ckpt_path, model, tok, extra={k: v for k, v in metrics.items() if k != "config"})
    with open(os.path.join(ROOT, "outputs", "logs", f"{run}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_losses(log_path, os.path.join(ROOT, "outputs", "figures", f"loss_curve_{run}.png"), run)
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2))
    print(f"checkpoint -> {ckpt_path}")
    return metrics


def plot_losses(log_path, fig_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    steps = [r["step"] for r in recs]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, [r["train_loss"] for r in recs], label="train loss", color="#2a6fdb")
    ax1.plot(steps, [r["val_loss"] for r in recs], label="validation loss", color="#e8590c")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cross-entropy (nats / char)")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(steps, [r["lr"] for r in recs], color="gray", ls=":", label="learning rate")
    ax2.set_ylabel("learning rate")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title(f"Training vs validation loss — {title}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "tiny_gpt.yaml"))
    ap.add_argument("--set", nargs="*", help="override config keys, e.g. --set n_layers=2 max_steps=500")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.update(parse_overrides(args.set))
    train(cfg)


if __name__ == "__main__":
    main()
