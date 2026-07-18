"""ui/ocr_page.py -- Streamlit page: OCR + document upload (PDF/DOCX/image) -> text -> translate."""

from __future__ import annotations

import streamlit as st

from config import settings
from database.db import log_ocr
from utils.doc_parser import parse_docx, parse_pdf, render_pdf_page_as_image
from utils.ocr import OCREngine


def render() -> None:
    st.header("📜 OCR & Document Upload")
    st.caption(
        "Extract text from photographed inscriptions, scanned manuscripts, PDFs, "
        "or Word documents using Tesseract / EasyOCR (open-source, offline)."
    )

    if any(s in settings.LOW_CONFIDENCE_SCRIPTS for s in settings.SUPPORTED_SCRIPTS):
        st.warning(
            "Heads up: for very old scripts such as **Brahmi, Grantha, Kharosthi, "
            "or Sharada**, current free/open-source OCR and LLMs are **not** "
            "reliably accurate. Results for these scripts should be treated as a "
            "rough starting point, not a verified reading, unless you have a "
            "specialized fine-tuned model."
        )

    uploaded = st.file_uploader(
        "Upload an image, PDF, or Word document",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf", "docx"],
    )
    engine_choice = st.selectbox("OCR engine", ["easyocr", "tesseract", "both"],
                                  index=["easyocr", "tesseract", "both"].index(settings.OCR_ENGINE)
                                  if settings.OCR_ENGINE in ("easyocr", "tesseract", "both") else 0)

    if uploaded is None:
        return

    file_bytes = uploaded.read()
    suffix = uploaded.name.lower().split(".")[-1]

    extracted_text = ""

    if suffix in ("png", "jpg", "jpeg", "tiff", "bmp"):
        st.image(file_bytes, caption=uploaded.name, use_column_width=True)
        if st.button("Run OCR", type="primary"):
            with st.spinner("Running OCR..."):
                ocr = OCREngine(engine=engine_choice)
                result = ocr.extract(file_bytes)
            extracted_text = result.text
            st.text_area("Extracted text", value=extracted_text, height=250)
            if result.confidence is not None:
                st.caption(f"Average confidence: {result.confidence:.1f}%")
            if result.warning:
                st.info(result.warning)
            log_ocr(uploaded.name, extracted_text, result.engine, result.confidence)

    elif suffix == "pdf":
        parsed = parse_pdf(file_bytes)
        st.write(f"Parsed **{len(parsed.pages)}** page(s).")
        for page in parsed.pages:
            with st.expander(f"Page {page.page_number}"):
                if page.text.strip():
                    st.write(page.text)
                else:
                    st.info("No embedded text detected -- likely a scanned page.")
                    if st.button(f"Run OCR on page {page.page_number}", key=f"ocr_{page.page_number}"):
                        with st.spinner("Rendering page and running OCR..."):
                            img_bytes = render_pdf_page_as_image(file_bytes, page.page_number - 1)
                            ocr = OCREngine(engine=engine_choice)
                            result = ocr.extract(img_bytes)
                        st.write(result.text)
                        if result.warning:
                            st.info(result.warning)
                        log_ocr(f"{uploaded.name} p{page.page_number}", result.text,
                                 result.engine, result.confidence)
        extracted_text = parsed.full_text

    elif suffix == "docx":
        parsed = parse_docx(file_bytes)
        st.text_area("Extracted text", value=parsed.full_text, height=250)
        extracted_text = parsed.full_text

    if extracted_text.strip():
        st.session_state["ocr_extracted_text"] = extracted_text
        st.success("Text captured. Head to the **Translate** tab to translate it, "
                   "or **Research (RAG)** to ask questions about it.")
