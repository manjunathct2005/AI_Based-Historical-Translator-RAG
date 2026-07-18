"""
rag/embeddings.py

Wraps a sentence-transformers embedding model (default: BAAI/bge-small-en-v1.5,
or the multilingual BAAI/bge-m3 for Indic-script text) behind a simple,
cached interface.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    def __init__(self, model_name: str = None, multilingual: bool = False):
        self.model_name = model_name or (
            settings.EMBEDDING_MODEL_NAME_MULTILINGUAL
            if multilingual
            else settings.EMBEDDING_MODEL_NAME
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        # BGE models are trained expecting a specific instruction prefix on
        # queries (but not on documents) for best retrieval quality.
        prefixed = _BGE_QUERY_PREFIX + text if "bge" in self.model_name.lower() else text
        vector = self.model.encode([prefixed], normalize_embeddings=True)
        return np.asarray(vector, dtype="float32")[0]

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


@lru_cache(maxsize=2)
def get_embedding_model(multilingual: bool = False) -> EmbeddingModel:
    """Cached accessor so we don't reload the model on every Streamlit rerun."""
    return EmbeddingModel(multilingual=multilingual)
