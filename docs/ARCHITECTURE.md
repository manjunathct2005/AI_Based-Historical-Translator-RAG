# Architecture

## Data flow: Research (RAG) tab

```
User query
   │
   ▼
web/search.py  ──►  DuckDuckGo search (multi-query expansion)
   │                (history/archaeology/inscription-biased queries)
   ▼
web/fetch.py   ──►  Fetch top N URLs, extract clean article text
   │                (trafilatura, readability-lxml fallback)
   ▼
utils/text_utils.chunk_text ──► sliding-window chunks
   │
   ▼
rag/embeddings.py ──► BAAI/bge-small-en-v1.5 (or bge-m3) embeddings
   │
   ▼
rag/vector_store.py ──► ephemeral FAISS/Chroma index for THIS query
   │
   ▼
Top-k passage retrieval (cosine similarity)
   │
   ▼
llm/llm_engine.py ──► open-source HF chat model, prompted to answer
   │                  ONLY from the numbered, cited sources
   ▼
Answer + inline citations + source list  ──►  Streamlit UI
   │
   ▼
database/db.py ──► log query/answer/sources for the History tab
```

## Data flow: Translate tab

```
User text + source/target language labels
   │
   ▼
translation/lang_codes.py ──► map labels to FLORES-200 / IndicTrans2 codes
   │
   ▼
translation/translator.py:
   if either language is Indic ──► try IndicTrans2 first
   else / on failure           ──► NLLB-200 general model
   │
   ▼
Translated text ──► Streamlit UI (+ .docx download via utils/doc_parser.write_docx)
   │
   ▼
database/db.py ──► log translation
```

## Data flow: OCR & Upload tab

```
Uploaded image / PDF / DOCX
   │
   ├─ image ─────────────► utils/ocr.py (Tesseract / EasyOCR) ──► text
   ├─ pdf ───────────────► utils/doc_parser.parse_pdf()
   │                          │ has embedded text? ──► use directly
   │                          └ scanned page? ────────► render_pdf_page_as_image()
   │                                                     + utils/ocr.py
   └─ docx ──────────────► utils/doc_parser.parse_docx()
   │
   ▼
Extracted text ──► session_state, ready to feed into Translate or RAG tabs
   │
   ▼
database/db.py ──► log OCR run (file name, engine, confidence)
```

## Why no local knowledge base?

Per project requirements, the RAG system does not ship or read from a
static local document folder. `data/` exists only for:
- ephemeral vector index scratch space during a single query, and
- (optionally) a persistent Chroma store if you enable that backend to
  avoid re-embedding frequently-revisited pages within a session.

All *content* used for grounding comes from live web search + fetch at
query time, so the system stays current and source-cited rather than
relying on a fixed, potentially stale corpus.

## Known accuracy limitations

- **Ancient/damaged scripts** (Brahmi, Grantha, Kharosthi, Sharada): no
  reliable open-source OCR/LLM support without specialized fine-tuning.
  The pipeline still runs but attaches low-confidence warnings.
- **Web retrieval quality** depends on what's publicly indexed; obscure or
  paywalled scholarly sources (many JSTOR articles, for instance) may only
  surface as abstracts/snippets rather than full text.
- **Translation quality** for low-resource Indic languages varies; NLLB-200
  and IndicTrans2 are strong for high/medium-resource languages (Hindi,
  Tamil, Telugu, Bengali, etc.) but weaker for very low-resource ones.
