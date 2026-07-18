"""ui/history_page.py -- Streamlit page: past activity + system health checks."""

from __future__ import annotations

import streamlit as st

from database.db import recent_ocr, recent_rag_queries, recent_translations
from health.healthcheck import run_all_checks


def render() -> None:
    st.header("🕘 History & System Health")

    tab1, tab2, tab3, tab4 = st.tabs(["Translations", "RAG queries", "OCR", "System health"])

    with tab1:
        rows = recent_translations()
        if not rows:
            st.info("No translations logged yet.")
        for r in rows:
            with st.expander(f"{r.source_lang} → {r.target_lang} · {r.created_at}"):
                st.write("**Source:**", r.source_text)
                st.write("**Translated:**", r.translated_text)
                st.caption(f"Model: {r.model_used}")

    with tab2:
        rows = recent_rag_queries()
        if not rows:
            st.info("No RAG queries logged yet.")
        for r in rows:
            with st.expander(f"{r.query[:80]} · {r.created_at}"):
                st.write(r.answer)
                if r.sources_json:
                    st.caption(f"Sources JSON: {r.sources_json}")

    with tab3:
        rows = recent_ocr()
        if not rows:
            st.info("No OCR runs logged yet.")
        for r in rows:
            with st.expander(f"{r.file_name or 'unnamed'} · {r.created_at}"):
                st.write(r.extracted_text)
                st.caption(f"Engine: {r.engine}, confidence: {r.confidence}")

    with tab4:
        if st.button("Run health checks"):
            checks = run_all_checks()
            for c in checks:
                icon = "✅" if c.ok else "❌"
                st.write(f"{icon} **{c.name}** — {c.detail}")
