import os

import pytest

from src.utils import DEFAULT_CHECKPOINT

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_CHECKPOINT), reason="train or download checkpoints/tiny_gpt.pt first")


def test_generate_endpoint_and_checkpoint_reload():
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/generate", json={"prompt": "ROMEO:", "max_new_tokens": 30, "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["generated"].startswith("ROMEO:") and len(body["generated"]) == len("ROMEO:") + 30
    assert client.post("/generate", json={"prompt": "日本"}).status_code == 400  # outside vocabulary
