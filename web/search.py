"""
web/search.py

Free, open-source web search (no API key required) using the
`duckduckgo_search` package. This is the entry point for the "browser
instead of local documents" retrieval strategy: instead of maintaining a
static corpus, every query is answered by searching the live web first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    domain: str
    trusted: bool


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def _is_trusted(domain: str) -> bool:
    return any(domain.endswith(t) for t in settings.TRUSTED_DOMAINS)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _ddg_search(query: str, max_results: int) -> List[dict]:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def web_search(query: str, max_results: int = None, prefer_trusted: bool = True) -> List[SearchHit]:
    """
    Search the live web for `query` and return ranked hits.

    If `prefer_trusted` is True, results from settings.TRUSTED_DOMAINS
    (Wikipedia, ASI, UNESCO, Sanskrit corpora, etc.) are sorted first --
    useful for historical/inscription queries where source quality matters
    a lot -- but untrusted-domain results are still included (and usable)
    unless settings.ALLOW_UNTRUSTED_DOMAINS is False.
    """
    max_results = max_results or settings.WEB_SEARCH_MAX_RESULTS
    raw = _ddg_search(query, max_results)

    hits: List[SearchHit] = []
    for r in raw:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        domain = _domain_of(url)
        trusted = _is_trusted(domain)
        if not trusted and not settings.ALLOW_UNTRUSTED_DOMAINS:
            continue
        hits.append(
            SearchHit(
                title=r.get("title", "").strip(),
                url=url,
                snippet=r.get("body", "").strip(),
                domain=domain,
                trusted=trusted,
            )
        )

    if prefer_trusted:
        hits.sort(key=lambda h: (not h.trusted,))

    return hits


def multi_query_search(queries: List[str], max_results_each: int = 5) -> List[SearchHit]:
    """Run several search queries and de-duplicate results by URL."""
    seen = set()
    combined: List[SearchHit] = []
    for q in queries:
        try:
            for hit in web_search(q, max_results=max_results_each):
                if hit.url not in seen:
                    seen.add(hit.url)
                    combined.append(hit)
        except Exception:
            logger.exception("Search failed for query: %s", q)
    return combined
