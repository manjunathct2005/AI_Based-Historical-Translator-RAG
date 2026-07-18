"""
database/models.py

SQLAlchemy models for lightweight app persistence: translation history and
RAG query history (including the source URLs used, for auditability).
This is NOT a document knowledge base -- retrieval always happens live
against the web (see rag/pipeline.py); this table only logs past
activity so users can revisit previous sessions.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_lang = Column(String(64), nullable=False)
    target_lang = Column(String(64), nullable=False)
    model_used = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class RAGQueryHistory(Base):
    __tablename__ = "rag_query_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # JSON list of {title, url}
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class OCRHistory(Base):
    __tablename__ = "ocr_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=True)
    extracted_text = Column(Text, nullable=False)
    engine = Column(String(32), nullable=False)
    confidence = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
