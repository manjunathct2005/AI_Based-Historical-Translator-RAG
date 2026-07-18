# 🏺 AI Historical Translator & RAG

An open-source Streamlit app for translating historical / Indic-language
text and researching historical topics (inscriptions, dynasties, scripts,
monuments) using a **web-grounded RAG pipeline** — there is intentionally
**no bundled local knowledge base**. Every research answer is sourced live
from the internet (search → fetch → embed → retrieve → answer), like using
a search engine and a browser, with citations back to the source URLs.

## Stack (all open-source)

| Concern        | Component |
|----------------|-----------|
| Frontend       | Streamlit |
| LLM            | Configurable Hugging Face model (default: `Qwen/Qwen2.5-1.5B-Instruct`) |
| Embeddings     | `BAAI/bge-small-en-v1.5` (English) / `BAAI/bge-m3` (multilingual) |
| Vector DB      | FAISS (default, in-memory) or ChromaDB (persistent) |
| Translation    | `facebook/nllb-200-distilled-600M` (general, 200 languages) + `ai4bharat/indictrans2-indic-en-1B` (Indic⇄English) |
| OCR            | Tesseract + EasyOCR |
| PDF parsing    | PyMuPDF + pdfplumber |
| Word docs      | python-docx |
| Web retrieval  | `duckduckgo-search` + `trafilatura`/`readability-lxml` (no API key needed) |
| Database       | SQLite via SQLAlchemy (history/logs only, not a knowledge base) |
| Deployment     | Docker / Hugging Face Spaces |

## ⚠️ Known limitation: ancient/damaged scripts

For scripts like **Brahmi, Grantha, Kharosthi, and Sharada**, or badly
damaged inscriptions, current free open-source OCR and LLMs are **not**
reliably accurate. The app still runs them through the pipeline (so
nothing breaks) but surfaces explicit low-confidence warnings. Treat that
output as a rough draft, not a verified reading, unless you bring in a
specialized fine-tuned model.

## Project layout

```
AI-Historical-Translator-RAG/
├── app.py              # Streamlit entry point
├── config.py           # Central, env-driven configuration
├── rag/                 # Web-grounded RAG pipeline (embeddings, vector store, orchestration)
├── web/                  # Live web search + page fetch/extraction ("the browser")
├── translation/         # NLLB-200 / IndicTrans2 wrappers + language code mapping
├── llm/                  # Open-source HF LLM wrapper
├── health/               # System health checks (internet, GPU, disk, deps)
├── database/            # SQLite models + helpers for history logging
├── utils/                # OCR, PDF/DOCX parsing, text chunking helpers
├── ui/                   # Streamlit page renderers
├── tests/                # pytest unit tests
├── docs/                 # Additional documentation
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Quick start (local)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # adjust if needed -- defaults work out of the box

# Optional: install system Tesseract + language data
#   Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-san
#   macOS:   brew install tesseract tesseract-lang

streamlit run app.py
```

First run will download the embedding, LLM, and translation model weights
from the Hugging Face Hub — this can take a while depending on your
connection; subsequent runs use the local HF cache.

## Quick start (Docker)

```bash
docker build -t historical-translator-rag .
docker run -p 8501:8501 historical-translator-rag
```

Then open http://localhost:8501.

## Deploying to Hugging Face Spaces

1. Create a new Space, SDK = Docker.
2. Push this repository's contents (the `Dockerfile` here works as-is).
3. Add any `.env` overrides as Space secrets/variables if you want to swap
   model sizes or vector backends.

## Running tests

```bash
pytest -q
```

## Notes on the "no local knowledge base, use the web" design

- `rag/pipeline.py` takes a query, expands it into a few targeted web
  searches (`web/search.py`), fetches and cleans the resulting pages
  (`web/fetch.py`), chunks + embeds them on the fly (`rag/embeddings.py`),
  builds an **ephemeral** vector index per query (`rag/vector_store.py`),
  and retrieves the top passages to ground the LLM's answer with citations.
- `database/` only stores **history/logs** of past translations, RAG
  queries, and OCR runs — it is not a source of retrieval context.
- To bias retrieval toward higher-quality historical sources, edit
  `TRUSTED_DOMAINS` in `.env` / `config.py` (Wikipedia, ASI, UNESCO,
  Sanskrit/Indological text repositories are included by default).
