"""Small FastAPI application exposing adapter hot-swapping."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from transformers.modeling_transformer import HotSwapLoRAAdapter


class _DummyModel:
    """Minimal model with attention and FFN weights.

    This is only intended for demos and unit tests; real deployments should
    provide their own model instance with attributes matching the adapter's
    expectations.
    """

    def __init__(self, dim: int = 32) -> None:
        self.attention = np.zeros((dim, dim), dtype=np.float32)
        self.ffn = np.zeros((dim, dim), dtype=np.float32)


app = FastAPI()
model = _DummyModel()
adapter = HotSwapLoRAAdapter(model)


class LoadRequest(BaseModel):
    bucket: str
    key: str


@app.post("/adapters/load")
async def load_adapter(req: LoadRequest) -> dict[str, str]:
    """Load LoRA weights from S3 and hot swap them into the model."""
    adapter.load_from_s3(req.bucket, req.key)
    return {"status": "loaded"}
