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

## Deploying to Streamlit Community Cloud

This repo includes the three files Streamlit Cloud looks for automatically:

- `requirements.txt` — version-pinned, lightest packages first, CPU-only
  `torch` wheel (via `--extra-index-url`), and **no ChromaDB** (FAISS is the
  default vector backend; Chroma pulls a very large dependency tree that
  commonly fails to build on free-tier runners — install it yourself if you
  want it: `pip install chromadb`).
- `packages.txt` — system `apt` packages Tesseract/OpenCV need
  (`tesseract-ocr` + language packs, `libgl1`, etc.). Streamlit Cloud reads
  this automatically; **without it, OCR will fail even if Python deps
  install fine.**
- `runtime.txt` — pins Python 3.11 so prebuilt wheels resolve instead of
  slow/failing source builds. (Some Streamlit Cloud accounts instead expose
  a Python-version dropdown in the app's "Advanced settings" at deploy
  time — set it to 3.11 there too if present.)

### Troubleshooting a failed deploy

If you see `ModuleNotFoundError` pointing at an import that looks
unrelated to the actual missing package (e.g. it blames
`from database.db import init_db` but the real problem is `sqlalchemy`
never got installed), it almost always means **the requirements.txt
install itself failed partway through** — usually on a heavy ML package
(`torch`, `faiss-cpu`, `easyocr`) — and pip stopped before installing
whatever was listed after it. Fix:

1. Open **Manage app → logs** on Streamlit Cloud and scroll up to the
   actual `pip install` output — the real failing package/line is there,
   above the generic error shown in the app UI.
2. Make sure you're using the pinned `requirements.txt` from this repo
   (not a hand-edited version with unpinned `>=` ranges), plus
   `packages.txt` and `runtime.txt` alongside it at the repo root.
3. Use **Reboot app** (not just refresh) after fixing dependencies, so
   Streamlit Cloud rebuilds the environment from scratch instead of
   reusing a half-installed cache.
4. If it still fails on `torch`/`transformers`/`easyocr`, your account's
   free-tier resources may simply be too small; either drop to a smaller
   `LLM_MODEL_NAME` (already defaulted to `Qwen/Qwen2.5-0.5B-Instruct` for
   this reason) or deploy via Docker/HF Spaces instead, where you can pick
   a larger instance.

### Memory note

Streamlit Community Cloud's free tier is roughly 1 CPU / ~1GB RAM. Loading
an LLM + embedding model + EasyOCR simultaneously can be tight. This repo
defaults to a small 0.5B LLM for that reason; bump `LLM_MODEL_NAME` in your
`.env`/Space secrets only if you've confirmed you have more headroom (e.g.
Docker on your own server, or a paid Streamlit/Spaces tier).

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
