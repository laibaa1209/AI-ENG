# MiniGPT — a small GPT-style language model built from scratch

AI Engineering, Assignment 1: *Build a small LLM from Scratch (Transformer Deep Dive)*

**Team:** `<<MEMBER 1 FULL NAME — ROLL NO>>`, `<<MEMBER 2 FULL NAME — ROLL NO>>`, `<<MEMBER 3 FULL NAME — ROLL NO>>`

We built a character-level, decoder-only Transformer in PyTorch and trained it from scratch on Tiny Shakespeare.
Every Transformer component is our own code: attention, multi-head split/merge, sinusoidal positions, the block,
the causal mask, the LM head, the loss, and sampling. **No pretrained checkpoint and no `transformers` model classes are used.**

| | |
|---|---|
| Model | 4 layers · 4 heads · d_model 128 · FFN 512 · context 128 · **809,984 parameters (0.81M)** |
| Data | Tiny Shakespeare, 1,115,394 chars, 65-char vocabulary, 90/10 contiguous split |
| Result | validation loss **1.638** nats/char (perplexity **5.14**) after 5,000 steps, 71 min on a laptop CPU |
| Report | [`report/report.md`](report/report.md) (Medium article source) · [`report/report.pdf`](report/report.pdf) |
| Colab | [`colab/MiniGPT_Colab.ipynb`](colab/MiniGPT_Colab.ipynb) |
| Live API | _add your Render URL here after deploying_ (see [Deployment](#deployment)) |

## Where things live

```
src/                       the model and the training pipeline
  tokenizer.py             deterministic char tokenizer (65 symbols, no special tokens)
  dataset.py               download, train/val split, shifted (x, y) batches
  attention.py             scaled dot-product attention, causal mask, MultiHeadAttention (explicit split/merge)
  positional_encoding.py   sinusoidal (default) and learned position encodings
  transformer_block.py     pre-LN block: x + MHA(LN(x)), x + FFN(LN(x)), GELU FFN
  gpt.py                   MiniGPT: embedding + position + blocks + LM head, loss, generate()
  train.py                 AdamW, warm-up + cosine LR, eval/logging/sampling, checkpointing, loss plot
  generate.py              command-line inference
  utils.py                 seeding, config, checkpoint save/load, model summary
api/app.py                 FastAPI service (/generate, /health, / playground, /docs)
tests/                     test_attention, test_mask, test_shapes, test_overfit, test_api
experiments/               hand_attention, shape_trace, positional_encoding_viz, why_transformers,
                           mask_comparison (MLM vs CLM), attention_visualization, ablation, evaluate
configs/tiny_gpt.yaml      the exact configuration of the shipped checkpoint
checkpoints/tiny_gpt.pt    trained weights (3.3 MB, committed)
outputs/figures|logs|samples   every figure, log and sample used in the report
report/                    technical report (Medium article) + PDF
colab/                     Google Colab notebook
```

## Setup

Python 3.10+.

```bash
git clone https://github.com/laibaa1209/AI-ENG.git && cd AI-ENG
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset downloads itself on first use into `data/tinyshakespeare.txt` (see [`data/README.md`](data/README.md)).

## Run the tests

```bash
python -m pytest -q
```

Five test files: the hand-worked attention example (weights and outputs), the sqrt(d_k) scaling, agreement with
PyTorch's reference attention, the causal mask (future weights are exactly 0; changing tokens after position t
leaves logits at positions ≤ t unchanged), the multi-head split/merge round trip and every tensor shape,
target shifting, the **tiny overfit gate** (loss 2.79 → <0.1 on one repeated batch), and the API with a fresh
checkpoint reload. Expected: `18 passed` (about 15 s on CPU).

## Inference (CLI)

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200            # temperature 0.8
python -m src.generate --prompt "ROMEO:" --greedy
python -m src.generate --prompt "JULIET:" --temperature 1.0 --top-k 20 --seed 7
```

Expected output (`--max-new-tokens 150`, seed 42):

```
Prompt: ROMEO:
Generated: ROMEO:
The world: now go will to this ralief,
It they holy 'Was from the common and with your lifes
That I would have my death on Marcius there,
The most of

model parameters: 0.81M (809,984)
context length: 128
layers/heads/d_model: 4/4/128
decoding: temperature=0.8 top_k=None  seed=42
checkpoint val loss: 1.6376  (perplexity 5.143)
```

Prompts may only use the 65 characters of the corpus (letters, digits `3`, punctuation `!$&',-.:;?`, space, newline).
Anything else is rejected with a clear error.

## Training

```bash
python -m src.train --config configs/tiny_gpt.yaml                      # reproduces checkpoints/tiny_gpt.pt (seed 1337)
python -m src.train --set n_layers=2 max_steps=500 run_name=quick checkpoint_path=checkpoints/quick.pt   # any key can be overridden
```

Writes `outputs/logs/<run>.jsonl` (step, train/val loss, lr, throughput, a sample every 250 steps),
`outputs/logs/<run>_metrics.json`, `outputs/figures/loss_curve_<run>.png` and the checkpoint.
The shipped model: batch 32, 5,000 steps, AdamW (β=0.9/0.95, wd 0.1), lr 1e-3 with 200 warm-up steps then
cosine decay to 1e-4, dropout 0.1, gradient clipping 1.0. That took 71 min on an Intel Core i7-8665U laptop CPU
(4 cores / 8 threads, 16 GB RAM, no GPU) and takes a few minutes on a Colab T4.

## Reproduce every figure and table in the report

```bash
python -m experiments.hand_attention            # Fig 1, 2  hand calculation + causal mask
python -m experiments.shape_trace               # Fig 3, 5  multi-head shape trace + parameter count
python -m experiments.positional_encoding_viz   # Fig 4
python -m experiments.why_transformers          # RNN vs attention: sequential steps, time, memory
python -m experiments.mask_comparison           # MLM (BERT) vs CLM (GPT) masking experiment
python -m experiments.attention_visualization   # Fig 7  attention of the trained model
python -m experiments.evaluate                  # Fig 8  full-val perplexity, 5 fixed prompts, greedy vs sampling
python -m experiments.ablation                  # Fig 9  depth / heads / context / position ablations (~1 h CPU)
python -m experiments.ablation --report         #        rebuild the table from existing logs only
```

## REST API

```bash
uvicorn api.app:app --port 8000
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 150, "temperature": 0.8, "seed": 0}'
```

| Endpoint | |
|---|---|
| `GET /` | small HTML playground |
| `GET /health` | model configuration, parameter count, validation loss |
| `POST /generate` | `{prompt, max_new_tokens (1–1000), temperature (0–2], top_k?, greedy?, seed?}` → generated text + latency |
| `GET /docs` | Swagger UI |

## Deployment

The repo contains a Render blueprint (`render.yaml`). It installs the CPU-only PyTorch wheel and starts uvicorn.

1. Push this repository to GitHub.
2. On [render.com](https://render.com): **New + → Blueprint →** select the repo → **Apply**.
3. Wait for the build (about 5 min), then open `https://<service-name>.onrender.com/`.

The free plan sleeps after 15 min idle. The first request after that takes about 30–60 s.

## Known limitations

* **Character-level, 0.81M parameters.** Output looks like Shakespeare (speaker tags, verse lines, archaic
  words), but sentences are not coherent across more than a line or two. Some invented words appear.
* **Context of 128 characters** (about 20–25 words). The model cannot keep a character or topic consistent beyond that.
* **Closed vocabulary**: 65 characters. No digits except `3`, no accented letters, no emoji.
* **Generalisation gap**: train 1.445 vs validation 1.638 at the end. Validation was still slowly improving, but the model has started memorising the 1 MB corpus.
* **Greedy decoding loops** ("the shall be the see…"). Use temperature sampling (default 0.8).
* **Generation recomputes the whole window for every character** (no KV cache): about 25 ms per character on one CPU thread.
* Generated text is an experimental artifact, not factual content.

## References

Vaswani et al., *Attention Is All You Need* (2017) · Radford et al., *Language Models are Unsupervised Multitask
Learners* (GPT-2, 2019) · Devlin et al., *BERT* (2019) · Karpathy, *char-rnn* (Tiny Shakespeare data) and
*nanoGPT* (consulted for the training-loop conventions: AdamW betas, cosine schedule, 0.02 init; no code copied) ·
Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020, pre-LN).
