import math

import torch
import torch.nn.functional as F

from src.attention import causal_mask, scaled_dot_product_attention

# The 3-token, d_k = d_v = 2 example worked by hand in the report (Section 6).
Q = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 4.0]])


def test_hand_example_matches_report():
    out, w = scaled_dot_product_attention(Q, K, V)
    expected_w = torch.tensor([[0.4011, 0.1978, 0.4011], [0.2483, 0.2483, 0.5035], [0.1978, 0.4011, 0.4011]])
    assert torch.allclose(w, expected_w, atol=1e-4)
    expected_out = torch.tensor([[0.9945, 2.6044], [0.9932, 2.7587], [1.4011, 2.4011]])  # hand-computed
    assert torch.allclose(out, expected_out, atol=1e-3)


def test_hand_example_causal_matches_report():
    out, w = scaled_dot_product_attention(Q, K, V, causal_mask(3))
    expected_w = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.1978, 0.4011, 0.4011]])
    assert torch.allclose(w, expected_w, atol=1e-4)
    assert torch.allclose(out[0], V[0])  # token 1 can only copy its own value


def test_scaling_uses_sqrt_dk():
    q, k, v = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 3)
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w, torch.softmax(q @ k.T / math.sqrt(8), dim=-1), atol=1e-6)


def test_weights_are_distributions_and_shapes():
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 6)
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
    assert out.shape == (2, 3, 5, 6) and w.shape == (2, 3, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5), atol=1e-6)
    assert (w >= 0).all()


def test_matches_pytorch_reference():
    q, k, v = torch.randn(2, 4, 7, 16), torch.randn(2, 4, 7, 16), torch.randn(2, 4, 7, 16)
    for causal in (False, True):
        ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(7) if causal else None)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert torch.allclose(ours, ref, atol=1e-5)
