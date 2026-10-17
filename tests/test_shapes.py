import torch

from src.attention import MultiHeadAttention
from src.dataset import get_batch
from src.gpt import GPT, GPTConfig
from src.positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding, sinusoidal_table
from src.tokenizer import CharTokenizer
from src.transformer_block import TransformerBlock


def test_split_merge_roundtrip():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 7, 32)
    h = mha.split_heads(x)
    assert h.shape == (2, 4, 7, 8)
    assert torch.equal(h[:, 1], x[:, :, 8:16])  # head 1 owns features 8..15
    assert torch.equal(mha.merge_heads(h), x)


def test_mha_and_block_preserve_shape():
    x = torch.randn(3, 10, 64)
    for heads in (1, 2, 4, 8):
        assert MultiHeadAttention(64, heads)(x).shape == x.shape
    block = TransformerBlock(64, 4, 256)
    assert block(x).shape == x.shape
    assert block.attn.last_weights.shape == (3, 4, 10, 10)


def test_positional_encodings():
    pe = sinusoidal_table(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))
    x = torch.zeros(2, 10, 16)
    assert SinusoidalPositionalEncoding(50, 16)(x).shape == (2, 10, 16)
    assert LearnedPositionalEncoding(50, 16)(x).shape == (2, 10, 16)


def test_gpt_logits_and_loss():
    cfg = GPTConfig(vocab_size=30, context_length=12, d_model=32, n_heads=4, n_layers=2, ffn_dim=128)
    model = GPT(cfg)
    idx = torch.randint(30, (4, 12))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 12, 30)
    assert loss.ndim == 0
    out = model.generate(idx[:1, :3], max_new_tokens=20)  # longer than context -> cropping path
    assert out.shape == (1, 23)


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=5, context_length=8)
    assert x.shape == y.shape == (5, 8)
    assert torch.equal(y, x + 1)  # with data = 0..99, the next token is always +1


def test_tokenizer_roundtrip_and_determinism():
    tok = CharTokenizer.from_text("hello world")
    assert tok.chars == sorted(set("hello world"))
    assert tok.decode(tok.encode("low hold")) == "low hold"
    try:
        tok.encode("xyz")
        assert False, "unknown chars must raise"
    except ValueError:
        pass
