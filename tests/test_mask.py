import torch

from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from src.gpt import GPT, GPTConfig


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    assert m.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_future_positions_get_zero_weight():
    q = k = v = torch.randn(1, 6, 4)
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    future = ~causal_mask(6)
    assert (w[0][future] == 0).all()


def test_mha_weights_are_causal_in_every_head():
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    mha(torch.randn(2, 5, 16))
    assert (mha.last_weights[..., ~causal_mask(5)] == 0).all()


def test_model_output_at_t_ignores_tokens_after_t():
    """Acceptance test: a token at position t cannot use information from positions > t."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, context_length=16, d_model=32, n_heads=4, n_layers=2, ffn_dim=64)).eval()
    x = torch.randint(20, (1, 16))
    t = 7
    x2 = x.clone()
    x2[0, t + 1 :] = torch.randint(20, (16 - t - 1,))  # change only the future
    l1, _ = model(x)
    l2, _ = model(x2)
    assert torch.allclose(l1[0, : t + 1], l2[0, : t + 1], atol=1e-6)
    assert not torch.allclose(l1[0, t + 1 :], l2[0, t + 1 :])


def test_bidirectional_model_does_leak_future():
    """Sanity check of the test above: with causal=False the same perturbation DOES change position t."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, context_length=16, d_model=32, n_heads=4, n_layers=2, ffn_dim=64, causal=False)).eval()
    x = torch.randint(20, (1, 16))
    x2 = x.clone()
    x2[0, 8:] = (x2[0, 8:] + 1) % 20
    assert not torch.allclose(model(x)[0][0, :8], model(x2)[0][0, :8])
