"""
web/fetch.py

Fetches and extracts clean article text from a URL, "browser-style" --
this is what turns a raw search hit into usable RAG context. Uses
trafilatura first (best boilerplate removal), falling back to
readability-lxml + BeautifulSoup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from utils.text_utils import clean_whitespace, truncate

logger = logging.getLogger(__name__)


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    success: bool
    error: Optional[str] = None


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def _get(url: str) -> requests.Response:
    headers = {"User-Agent": settings.WEB_USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=settings.WEB_FETCH_TIMEOUT)
    resp.raise_for_status()
    return resp


def fetch_page(url: str) -> FetchedPage:
    """Fetch a URL and return cleaned article text, like a headless browser reader-mode."""
    try:
        resp = _get(url)
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return FetchedPage(url=url, title="", text="", success=False, error=str(exc))

    html = resp.text

    text, title = _extract_with_trafilatura(html, url)
    if not text:
        text, title = _extract_with_readability(html)

    text = clean_whitespace(text)
    text = truncate(text, settings.WEB_FETCH_MAX_CHARS)

    if not text:
        return FetchedPage(url=url, title=title, text="", success=False,
                            error="No extractable content")

    return FetchedPage(url=url, title=title, text=text, success=True)


def _extract_with_trafilatura(html: str, url: str):
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True,
            favor_recall=True,
        )
        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if metadata and metadata.title else ""
        return (extracted or "", title)
    except Exception:
        logger.debug("trafilatura extraction failed for %s", url, exc_info=True)
        return "", ""


def _extract_with_readability(html: str):
    try:
        from readability import Document
        from bs4 import BeautifulSoup

        doc = Document(html)
        title = doc.short_title()
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = soup.get_text(separator="\n")
        return text, title
    except Exception:
        logger.debug("readability extraction failed", exc_info=True)
        return "", ""


def fetch_many(urls, max_workers: int = 6):
    """Fetch multiple URLs concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_page, u): u for u in urls}
        for future in as_completed(futures):
            results.append(future.result())
    return results
