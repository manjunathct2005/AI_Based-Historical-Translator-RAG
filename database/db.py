"""
database/db.py

Engine/session setup + small repository-style helper functions used by
the Streamlit UI to log and browse history.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from database.models import Base, TranslationHistory, RAGQueryHistory, OCRHistory

_engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False}
                         if settings.DATABASE_URL.startswith("sqlite") else {})
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=_engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------- #
# Translation history
# ---------------------------------------------------------------------- #
def log_translation(source_text: str, translated_text: str, source_lang: str,
                     target_lang: str, model_used: str) -> None:
    with get_session() as s:
        s.add(TranslationHistory(
            source_text=source_text, translated_text=translated_text,
            source_lang=source_lang, target_lang=target_lang, model_used=model_used,
        ))


def recent_translations(limit: int = 20) -> List[TranslationHistory]:
    with get_session() as s:
        return (
            s.query(TranslationHistory)
            .order_by(TranslationHistory.created_at.desc())
            .limit(limit)
            .all()
        )


# ---------------------------------------------------------------------- #
# RAG query history
# ---------------------------------------------------------------------- #
def log_rag_query(query: str, answer: str, sources: List[dict]) -> None:
    with get_session() as s:
        s.add(RAGQueryHistory(query=query, answer=answer, sources_json=json.dumps(sources)))


def recent_rag_queries(limit: int = 20) -> List[RAGQueryHistory]:
    with get_session() as s:
        return (
            s.query(RAGQueryHistory)
            .order_by(RAGQueryHistory.created_at.desc())
            .limit(limit)
            .all()
        )


# ---------------------------------------------------------------------- #
# OCR history
# ---------------------------------------------------------------------- #
def log_ocr(file_name: Optional[str], extracted_text: str, engine: str,
            confidence: Optional[float]) -> None:
    with get_session() as s:
        s.add(OCRHistory(
            file_name=file_name, extracted_text=extracted_text, engine=engine,
            confidence=str(confidence) if confidence is not None else None,
        ))


def recent_ocr(limit: int = 20) -> List[OCRHistory]:
    with get_session() as s:
        return (
            s.query(OCRHistory)
            .order_by(OCRHistory.created_at.desc())
            .limit(limit)
            .all()
        )
