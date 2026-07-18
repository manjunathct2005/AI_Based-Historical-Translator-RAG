"""utils/text_utils.py -- small shared text helpers used across modules."""

from __future__ import annotations

import re
from typing import List


def clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Simple sliding-window chunker over whitespace-normalized text.
    Good enough for web articles / OCR output; not tokenizer-exact but
    keeps chunk boundaries away from mid-word cuts where possible.
    """
    text = clean_whitespace(text)
    if not text:
        return []

    words = text.split(" ")
    chunks: List[str] = []
    start = 0
    approx_words_per_chunk = max(chunk_size // 6, 20)  # ~6 chars/word heuristic
    approx_overlap_words = max(overlap // 6, 5)

    while start < len(words):
        end = min(start + approx_words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(words):
            break
        start = end - approx_overlap_words

    return chunks


def detect_language(text: str) -> str:
    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return "unknown"


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"
