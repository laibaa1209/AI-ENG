"""Command-line inference.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 200
    python -m src.generate --prompt "ROMEO:" --greedy
"""
import argparse

import torch

from .utils import DEFAULT_CHECKPOINT, get_device, load_checkpoint, set_seed


def generate_text(model, tok, prompt, max_new_tokens=200, temperature=0.8, top_k=None, greedy=False, device="cpu"):
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, greedy=greedy)
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    model, tok, extra = load_checkpoint(args.checkpoint, device)
    text = generate_text(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.greedy, device)

    print(f"Prompt: {args.prompt}")
    print(f"Generated: {text}")
    print()
    print(f"model parameters: {model.num_params() / 1e6:.2f}M ({model.num_params():,})")
    print(f"context length: {model.cfg.context_length}")
    print(f"layers/heads/d_model: {model.cfg.n_layers}/{model.cfg.n_heads}/{model.cfg.d_model}")
    print(f"decoding: {'greedy' if args.greedy else f'temperature={args.temperature} top_k={args.top_k}'}  seed={args.seed}")
    if "final_val_loss" in extra:
        print(f"checkpoint val loss: {extra['final_val_loss']}  (perplexity {extra['val_perplexity']})")


if __name__ == "__main__":
    main()
