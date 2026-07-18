"""ui/rag_page.py -- Streamlit page: web-grounded RAG research assistant."""

from __future__ import annotations

import streamlit as st

from database.db import log_rag_query
from rag.pipeline import answer_with_rag


def render() -> None:
    st.header("🌐 Web-Grounded Research (RAG)")
    st.caption(
        "This assistant does NOT use a local document folder. Every answer is "
        "sourced live from the web (search → fetch → embed → retrieve → answer), "
        "similar to how a search engine + browser would work, with citations."
    )

    query = st.text_input(
        "Ask about an inscription, script, dynasty, monument, manuscript, etc.",
        placeholder="e.g. What does the Allahabad Pillar inscription of Samudragupta say?",
    )
    top_k = st.slider("Number of passages to retrieve", 3, 10, 5)
    multilingual = st.checkbox("Use multilingual embeddings (better for non-English sources)", value=False)

    if st.button("Research", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching the web, fetching pages, and reasoning over them..."):
            try:
                result = answer_with_rag(query, top_k=top_k, multilingual=multilingual)
            except Exception as exc:
                st.error(f"RAG pipeline failed: {exc}")
                return

        st.subheader("Answer")
        st.write(result.answer)

        if result.sources:
            st.subheader("Sources consulted")
            for hit in result.sources:
                trust_badge = "✅ trusted" if hit.trusted else "🌐 general web"
                st.markdown(f"- [{hit.title or hit.url}]({hit.url}) — {trust_badge}")

        if result.passages:
            with st.expander("Retrieved passages (used as LLM context)"):
                for i, p in enumerate(result.passages, start=1):
                    st.markdown(f"**[{i}] {p.source_title}** — score {p.score:.3f}")
                    st.caption(p.source_url)
                    st.write(p.text[:800] + ("…" if len(p.text) > 800 else ""))
                    st.divider()

        log_rag_query(
            query, result.answer,
            [{"title": h.title, "url": h.url} for h in result.sources],
        )
