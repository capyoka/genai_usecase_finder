from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

from ..config import get_settings


class EmbeddingClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._use_openai = bool(self.settings.openai_api_key and OpenAI is not None)
        self._client = OpenAI(api_key=self.settings.openai_api_key) if self._use_openai else None
        self.dimension = (
            self.settings.embedding_dim if self._use_openai else self.settings.fallback_embedding_dim
        )

    def embed_texts(self, texts: Iterable[str]) -> List[np.ndarray]:
        items = list(texts)
        if not items:
            return []
        if self._use_openai and self._client is not None:
            response = self._client.embeddings.create(model=self.settings.embedding_model, input=items)
            vectors = [np.array(data.embedding, dtype=np.float32) for data in response.data]
        else:
            vectors = [self._fallback_embedding(text) for text in items]
        return [self._normalize(vec) for vec in vectors]

    def _fallback_embedding(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        return vec

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return (vec / norm).astype(np.float32)
