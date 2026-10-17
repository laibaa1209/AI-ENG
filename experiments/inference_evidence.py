"""Figure 10: a real CLI call and a real HTTP call to the running API, captured as text + PNG.

    python -m experiments.inference_evidence
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from src.utils import ROOT, save_text_figure


def main():
    cmd = [sys.executable, "-m", "src.generate", "--prompt", "ROMEO:", "--max-new-tokens", "150"]
    cli = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    cli_block = "$ python -m src.generate --prompt \"ROMEO:\" --max-new-tokens 150\n" + cli

    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.app:app", "--port", "8765"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                health = json.load(urllib.request.urlopen("http://127.0.0.1:8765/health"))
                break
            except OSError:
                time.sleep(0.5)
        payload = {"prompt": "JULIET:", "max_new_tokens": 150, "temperature": 0.8, "seed": 0}
        req = urllib.request.Request("http://127.0.0.1:8765/generate", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req))
    finally:
        server.terminate()

    api_block = (
        "$ uvicorn api.app:app --port 8765 &\n"
        "$ curl http://127.0.0.1:8765/health\n" + json.dumps(health) + "\n\n"
        "$ curl -X POST http://127.0.0.1:8765/generate -H 'Content-Type: application/json' \\\n"
        f"       -d '{json.dumps(payload)}'\n" + json.dumps(resp, indent=2)
    )
    text = cli_block + "\n" + api_block
    print(text)
    with open(os.path.join(ROOT, "outputs", "logs", "inference_evidence.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    # JSON strings show newlines as \n; that's what a real terminal shows too
    save_text_figure(text, os.path.join(ROOT, "outputs", "figures", "fig10_api_call.png"), "Figure 10 - CLI and REST API inference")


if __name__ == "__main__":
    main()
