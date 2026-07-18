"""ui/translate_page.py -- Streamlit page: text translation."""

from __future__ import annotations

import streamlit as st

from database.db import log_translation
from translation.lang_codes import supported_language_labels
from translation.translator import translate
from utils.doc_parser import write_docx


def render() -> None:
    st.header("🔤 Historical / Indic Text Translation")
    st.caption(
        "Powered by NLLB-200 (general, 200 languages) and IndicTrans2 "
        "(specialized Indic ⇄ English). Both run locally, open-source, no API key."
    )

    langs = supported_language_labels()
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Source language", langs,
                                    index=langs.index("sanskrit") if "sanskrit" in langs else 0)
    with col2:
        target_lang = st.selectbox("Target language", langs,
                                    index=langs.index("english") if "english" in langs else 0)

    text = st.text_area("Text to translate", height=200,
                         placeholder="Paste transliterated or Unicode source text here...")

    if st.button("Translate", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text first.")
            return
        with st.spinner(f"Translating {source_lang} → {target_lang}..."):
            try:
                result = translate(text, source_lang, target_lang)
            except Exception as exc:
                st.error(f"Translation failed: {exc}")
                return

        st.success("Translation complete")
        st.text_area("Translated text", value=result.translated_text, height=200)
        st.caption(f"Model used: `{result.model_used}`")
        if result.warning:
            st.info(result.warning)

        log_translation(text, result.translated_text, source_lang, target_lang, result.model_used)

        docx_bytes = write_docx(
            f"Source ({source_lang}):\n{text}\n\nTranslation ({target_lang}):\n{result.translated_text}",
            title="Translation Output",
        )
        st.download_button(
            "⬇️ Download as Word document", data=docx_bytes,
            file_name="translation.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
