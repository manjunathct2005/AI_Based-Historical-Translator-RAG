"""
rag/vector_store.py

An ephemeral (per-session/query) vector index over freshly web-fetched
passages. Because this project deliberately has no static local corpus,
the index is built on-the-fly from search results for each query rather
than persisted as a permanent knowledge base -- though it CAN be persisted
to disk within a session to avoid re-embedding the same pages twice.

Two backends are supported:
  - FAISS (default): fast, in-memory, zero external services.
  - ChromaDB: persistent, supports metadata filtering.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    id: str
    text: str
    source_url: str
    source_title: str
    score: float = 0.0


class FaissVectorStore:
    """Simple flat inner-product index (vectors are pre-normalized -> cosine sim)."""

    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._passages: List[Passage] = []

    def add(self, vectors: np.ndarray, passages: List[Passage]) -> None:
        assert len(passages) == vectors.shape[0]
        self.index.add(vectors)
        self._passages.extend(passages)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Passage]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        results: List[Passage] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            p = self._passages[idx]
            results.append(Passage(id=p.id, text=p.text, source_url=p.source_url,
                                    source_title=p.source_title, score=float(score)))
        return results

    def __len__(self):
        return len(self._passages)


class ChromaVectorStore:
    """Chroma-backed store; persists to settings.VECTOR_INDEX_PATH."""

    def __init__(self, collection_name: str = None):
        import chromadb

        self.client = chromadb.PersistentClient(path=settings.VECTOR_INDEX_PATH)
        self.collection = self.client.get_or_create_collection(
            collection_name or settings.CHROMA_COLLECTION
        )

    def add(self, vectors: np.ndarray, passages: List[Passage]) -> None:
        self.collection.add(
            ids=[p.id for p in passages],
            embeddings=[v.tolist() for v in vectors],
            documents=[p.text for p in passages],
            metadatas=[{"source_url": p.source_url, "source_title": p.source_title}
                       for p in passages],
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Passage]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
        results: List[Passage] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append(Passage(
                id=i, text=doc,
                source_url=meta.get("source_url", ""),
                source_title=meta.get("source_title", ""),
                score=1 - dist,  # convert distance to a similarity-like score
            ))
        return results

    def __len__(self):
        return self.collection.count()


def build_vector_store(dim: int, backend: str = None):
    backend = (backend or settings.VECTOR_BACKEND).lower()
    if backend == "faiss":
        return FaissVectorStore(dim)
    elif backend == "chroma":
        return ChromaVectorStore()
    raise ValueError(f"Unknown vector backend: {backend}")


def make_passage(text: str, source_url: str, source_title: str) -> Passage:
    return Passage(id=str(uuid.uuid4()), text=text, source_url=source_url,
                    source_title=source_title)
