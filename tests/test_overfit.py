"""Debugging gate (Milestone 4): the model must be able to memorise one tiny repeated batch.

If target shifting, masking or gradients were broken, the loss would plateau.
"""
import torch

from src.dataset import get_batch
from src.gpt import GPT, GPTConfig
from src.tokenizer import CharTokenizer


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    text = "To be, or not to be, that is the question. " * 4
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text))
    x, y = get_batch(data, batch_size=8, context_length=32)  # one fixed batch, reused every step

    model = GPT(GPTConfig(vocab_size=tok.vocab_size, context_length=32, d_model=64, n_heads=4, n_layers=2, ffn_dim=256, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    print(f"overfit: loss {first:.3f} -> {loss.item():.3f}")
    assert first > 2.5  # ~ln(vocab) at init
    assert loss.item() < 0.1
