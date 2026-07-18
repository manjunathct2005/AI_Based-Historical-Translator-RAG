"""
rag/pipeline.py

The core "web-grounded RAG" pipeline:

  1. Take the user's question / passage-to-research.
  2. Search the live web (web/search.py) -- NOT a local document folder.
  3. Fetch + clean the top pages (web/fetch.py).
  4. Chunk + embed those pages (rag/embeddings.py) into an ephemeral
     vector index (rag/vector_store.py) built fresh for this query.
  5. Retrieve the most relevant chunks for the question.
  6. Hand the question + retrieved, cited context to the LLM (llm/llm_engine.py)
     to produce a grounded answer with source citations.

This intentionally avoids any bundled knowledge base: every answer is
sourced live, "like Google", with citations back to the URLs used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config import settings
from rag.embeddings import get_embedding_model
from rag.vector_store import build_vector_store, make_passage, Passage
from utils.text_utils import chunk_text
from web.search import web_search, multi_query_search, SearchHit
from web.fetch import fetch_many

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    query: str
    answer: str
    passages: List[Passage] = field(default_factory=list)
    sources: List[SearchHit] = field(default_factory=list)


def _search_queries(query: str) -> List[str]:
    """
    Expand a single user query into a few targeted web searches, biased
    towards historical / archaeological / Indological sourcing since
    that's this app's domain.
    """
    return [
        query,
        f"{query} history archaeology",
        f"{query} inscription translation meaning",
    ]


def retrieve(query: str, top_k: int = 5, multilingual: bool = False) -> tuple[List[Passage], List[SearchHit]]:
    """Run the web-search -> fetch -> embed -> retrieve leg of the pipeline."""
    hits = multi_query_search(_search_queries(query), max_results_each=settings.WEB_SEARCH_MAX_RESULTS // 2 or 3)
    if not hits:
        logger.warning("No web search hits for query: %s", query)
        return [], []

    pages = fetch_many([h.url for h in hits[:settings.WEB_SEARCH_MAX_RESULTS]])
    url_to_hit = {h.url: h for h in hits}

    embed_model = get_embedding_model(multilingual=multilingual)
    store = build_vector_store(dim=embed_model.dim)

    used_hits: List[SearchHit] = []
    for page in pages:
        if not page.success or not page.text:
            continue
        chunks = chunk_text(page.text, chunk_size=800, overlap=100)
        if not chunks:
            continue
        vectors = embed_model.embed_documents(chunks)
        passages = [
            make_passage(c, source_url=page.url, source_title=page.title or page.url)
            for c in chunks
        ]
        store.add(vectors, passages)
        if page.url in url_to_hit:
            used_hits.append(url_to_hit[page.url])

    if len(store) == 0:
        return [], used_hits

    query_vector = embed_model.embed_query(query)
    retrieved = store.search(query_vector, top_k=top_k)
    return retrieved, used_hits


def build_context_block(passages: List[Passage]) -> str:
    """Format retrieved passages into a citation-friendly context block for the LLM prompt."""
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] Source: {p.source_title} ({p.source_url})\n{p.text}")
    return "\n\n".join(lines)


def answer_with_rag(query: str, top_k: int = 5, multilingual: bool = False) -> RAGAnswer:
    """Full pipeline: retrieve live web context, then generate a grounded answer."""
    from llm.llm_engine import get_llm_engine

    passages, hits = retrieve(query, top_k=top_k, multilingual=multilingual)

    if not passages:
        return RAGAnswer(
            query=query,
            answer=(
                "I couldn't find usable web sources for this query right now. "
                "Try rephrasing, or check your internet connection / search settings."
            ),
            passages=[], sources=hits,
        )

    context = build_context_block(passages)
    system_prompt = (
        "You are a careful research assistant specializing in Indian history, "
        "epigraphy, and historical translation. Answer the user's question using "
        "ONLY the numbered sources below. Cite sources inline like [1], [2]. "
        "If the sources don't fully answer the question, say so explicitly rather "
        "than guessing. Be concise and factual."
    )
    user_prompt = f"Sources:\n{context}\n\nQuestion: {query}"

    llm = get_llm_engine()
    answer = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    return RAGAnswer(query=query, answer=answer, passages=passages, sources=hits)
