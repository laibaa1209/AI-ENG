"""REST API for MiniGPT.

    uvicorn api.app:app --host 0.0.0.0 --port 8000
    curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "ROMEO:", "max_new_tokens": 150}'

Interactive docs at /docs, a tiny HTML playground at /.
"""
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.generate import generate_text
from src.utils import DEFAULT_CHECKPOINT, load_checkpoint

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))  # small free-tier instances have ~1 vCPU
MODEL, TOK, EXTRA = load_checkpoint(os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT))

app = FastAPI(title="MiniGPT (Tiny Shakespeare)", description="Character-level GPT trained from scratch.", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", min_length=1, max_length=500)
    max_new_tokens: int = Field(200, ge=1, le=1000)
    temperature: float = Field(0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1, le=65)
    greedy: bool = False
    seed: int | None = None


@app.get("/health")
def health():
    c = MODEL.cfg
    return {"status": "ok", "parameters": MODEL.num_params(), "context_length": c.context_length,
            "n_layers": c.n_layers, "n_heads": c.n_heads, "d_model": c.d_model, "vocab_size": c.vocab_size,
            "val_loss": EXTRA.get("final_val_loss")}


@app.post("/generate")
def generate(req: GenerateRequest):
    if req.seed is not None:
        torch.manual_seed(req.seed)
    t = time.perf_counter()
    try:
        text = generate_text(MODEL, TOK, req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.greedy)
    except ValueError as e:  # character outside the 65-char Shakespeare vocabulary
        raise HTTPException(status_code=400, detail=str(e))
    return {"prompt": req.prompt, "generated": text, "completion": text[len(req.prompt):],
            "max_new_tokens": req.max_new_tokens, "temperature": req.temperature, "top_k": req.top_k,
            "greedy": req.greedy, "latency_ms": round((time.perf_counter() - t) * 1000)}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniGPT</title><style>body{font-family:system-ui;max-width:760px;margin:2rem auto;padding:0 16px}
textarea,input{width:100%;font:inherit;box-sizing:border-box}pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}</style></head>
<body><h1>MiniGPT &mdash; Tiny Shakespeare</h1><p>A character-level GPT trained from scratch. <a href="/docs">API docs</a></p>
<label>Prompt<textarea id="p" rows="3">ROMEO:</textarea></label>
<label>Max new tokens <input id="n" type="number" value="200" min="1" max="1000"></label>
<label>Temperature <input id="t" type="number" value="0.8" step="0.1" min="0.1" max="2"></label>
<p><button onclick="go()">Generate</button></p><pre id="o"></pre>
<script>async function go(){o.textContent="generating...";const r=await fetch("/generate",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:p.value,max_new_tokens:+n.value,temperature:+t.value})});
const j=await r.json();o.textContent=r.ok?j.generated+"\\n\\n["+j.latency_ms+" ms]":JSON.stringify(j.detail)}</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
