"""Figure 3 + Figure 5: multi-head tensor-shape trace and model summary.

    python -m experiments.shape_trace
"""
import contextlib
import io
import os

import torch

from src.gpt import GPT, GPTConfig
from src.utils import ROOT, load_config, model_summary, save_text_figure


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "tiny_gpt.yaml"))
    model = GPT(GPTConfig(**{k: cfg[k] for k in GPTConfig.__dataclass_fields__ if k in cfg})).eval()
    x = torch.randint(65, (2, 16))  # B=2, T=16 so the trace is readable

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"token ids                   {tuple(x.shape)}  (B, T)")
        emb = model.tok_emb(x)
        print(f"token embedding             {tuple(emb.shape)}  (B, T, d_model)")
        print(f"+ sinusoidal position       {tuple(model.pos(emb).shape)}")
        print("--- block 0 / multi-head attention ---")
        logits, _ = model(x, trace=True)
        print("--- after all blocks ---")
        print(f"logits                      {tuple(logits.shape)}  (B, T, vocab)")
        mha = model.blocks[0].attn
        print(f"\nn_heads={mha.n_heads}, d_head={mha.d_head}: each head attends in its own {mha.d_head}-dim subspace,")
        print(f"then {mha.n_heads} x {mha.d_head} = {mha.n_heads * mha.d_head} features are concatenated and mixed by W_O.")
    trace = buf.getvalue()
    print(trace)
    save_text_figure(trace, os.path.join(ROOT, "outputs", "figures", "fig3_shape_trace.png"), "Figure 3 - multi-head attention shape trace (python -m experiments.shape_trace)")

    summary = model_summary(model)
    print(summary)
    save_text_figure(summary, os.path.join(ROOT, "outputs", "figures", "fig5_model_summary.png"), "Figure 5 - model summary / parameter count")
    with open(os.path.join(ROOT, "outputs", "logs", "shape_trace.txt"), "w") as f:
        f.write(trace + "\n" + summary + "\n")


if __name__ == "__main__":
    main()
