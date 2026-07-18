"""
app.py

Main Streamlit entry point for AI-Historical-Translator-RAG.

This app deliberately ships with NO local knowledge base -- the "data/"
folder is only used for caches and ephemeral vector indices. All RAG
context is retrieved live from the web (see rag/pipeline.py and web/),
similar to using a search engine + browser, with source citations.
"""

from __future__ import annotations

import logging

import streamlit as st

from config import settings
from database.db import init_db
from ui import history_page, ocr_page, rag_page, translate_page

logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG)

st.set_page_config(
    page_title=settings.UI_TITLE,
    layout=settings.UI_LAYOUT,
    page_icon="🏺",
)

init_db()


def sidebar() -> str:
    with st.sidebar:
        st.title(settings.UI_TITLE)
        st.caption("Open-source LLM · live web RAG · NLLB-200 / IndicTrans2 · Tesseract / EasyOCR")

        page = st.radio(
            "Navigate",
            ["Translate", "Research (RAG)", "OCR & Upload", "History & Health"],
            index=0,
        )

        st.divider()
        st.markdown("**Config snapshot**")
        st.code(
            f"LLM: {settings.LLM_MODEL_NAME}\n"
            f"Embeddings: {settings.EMBEDDING_MODEL_NAME}\n"
            f"Vector backend: {settings.VECTOR_BACKEND}\n"
            f"OCR engine: {settings.OCR_ENGINE}\n"
            f"Web search: {settings.WEB_SEARCH_PROVIDER}",
            language="text",
        )
        st.caption(
            "⚠️ Ancient/damaged scripts (Brahmi, Grantha, Kharosthi, Sharada) are "
            "not reliably handled by current open-source OCR/LLMs without "
            "specialized fine-tuning."
        )
        return page


def main() -> None:
    page = sidebar()

    if page == "Translate":
        translate_page.render()
    elif page == "Research (RAG)":
        rag_page.render()
    elif page == "OCR & Upload":
        ocr_page.render()
    elif page == "History & Health":
        history_page.render()


if __name__ == "__main__":
    main()
