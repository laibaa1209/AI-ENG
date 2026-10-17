"""Controlled ablations: change ONE factor at a time against a shared baseline.

    python -m experiments.ablation            # run everything, then tabulate
    python -m experiments.ablation --report  # only rebuild table/plots from existing logs

Every run is a separate process so peak-memory numbers are not polluted by earlier runs.
Shared settings: batch 16, 1000 steps, lr 1e-3 cosine, seed 1337, dropout 0.1.
"""
import argparse
import json
import os
import subprocess
import sys

from src.utils import ROOT

COMMON = ["batch_size=16", "max_steps=1000", "eval_interval=100", "eval_iters=20", "sample_tokens=200", "warmup_steps=100"]
BASE = {"n_layers": 4, "n_heads": 4, "context_length": 128, "pos_encoding": "sinusoidal", "lr": 1e-3, "min_lr": 1e-4}

RUNS = {
    "abl_base": {},
    "abl_depth2": {"n_layers": 2},
    "abl_depth6": {"n_layers": 6},
    "abl_heads2": {"n_heads": 2},
    "abl_heads8": {"n_heads": 8},
    "abl_ctx64": {"context_length": 64},
    "abl_ctx256": {"context_length": 256},
    "abl_pos_learned": {"pos_encoding": "learned"},
    "abl_lr3e-4": {"lr": 3e-4, "min_lr": 3e-5},
    "abl_base_seed2": {"seed": 2},  # same config, different seed -> run-to-run noise floor
}

EXPERIMENTS = {
    "Depth (n_layers)": ("n_layers", ["abl_depth2", "abl_base", "abl_depth6"]),
    "Heads (n_heads)": ("n_heads", ["abl_heads2", "abl_base", "abl_heads8"]),
    "Context length": ("context_length", ["abl_ctx64", "abl_base", "abl_ctx256"]),
    "Position method": ("pos_encoding", ["abl_base", "abl_pos_learned"]),
    "Learning rate": ("lr", ["abl_lr3e-4", "abl_base"]),
    "Seed (noise floor)": ("seed", ["abl_base", "abl_base_seed2"]),
}


def run_all():
    for name, change in RUNS.items():
        if os.path.exists(os.path.join(ROOT, "outputs", "logs", f"{name}_metrics.json")):
            print(f"skip {name} (already done)")
            continue
        overrides = [f"{k}={v}" for k, v in {**BASE, **change}.items()]
        cmd = [sys.executable, "-m", "src.train", "--set", f"run_name={name}", f"checkpoint_path=checkpoints/{name}.pt", *COMMON, *overrides]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def load(name):
    with open(os.path.join(ROOT, "outputs", "logs", f"{name}_metrics.json")) as f:
        return json.load(f)


def report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["| Experiment | Setting | Params | Train loss | Val loss | Val PPL | Train time (s) | Tokens/s | Peak RSS (MB) | Attn-score mem (MB) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 3, figsize=(4.2 * 3, 3.6 * 2))
    for ax, (title, (key, names)) in zip(axes.flat, EXPERIMENTS.items()):
        for n in names:
            m = load(n)
            setting = m["config"][key]
            rows.append(f"| {title} | {key}={setting} | {m['params']:,} | {m['final_train_loss']:.4f} | {m['final_val_loss']:.4f} | "
                        f"{m['val_perplexity']:.2f} | {m['train_time_s']:.0f} | {m['tokens_per_s']:,} | {m['peak_mem_mb']:.0f} | {m['attn_scores_mb']:.1f} |")
            recs = [json.loads(l) for l in open(os.path.join(ROOT, "outputs", "logs", f"{n}.jsonl"), encoding="utf-8")]
            ax.plot([r["step"] for r in recs], [r["val_loss"] for r in recs], label=f"{key}={setting}")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("validation loss")
        ax.set_ylim(2.0, 2.8)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "outputs", "figures", "ablation_val_curves.png"), dpi=150)

    table = "\n".join(rows)
    with open(os.path.join(ROOT, "outputs", "logs", "ablation_results.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    with open(os.path.join(ROOT, "outputs", "samples", "ablation_samples.md"), "w", encoding="utf-8") as f:
        for n in RUNS:
            last = [json.loads(l) for l in open(os.path.join(ROOT, "outputs", "logs", f"{n}.jsonl"), encoding="utf-8")][-1]
            f.write(f"## {n}\n\n```\n{last['sample']}\n```\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        run_all()
    report()
