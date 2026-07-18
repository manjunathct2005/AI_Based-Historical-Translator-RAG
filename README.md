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

Streamlit Community Cloud now provisions **Python 3.14** by default and
installs dependencies with **`uv`** (not plain `pip`), which changes how
some things resolve. Issues actually hit while building this project, and
the fixes now baked into this repo:

1. **`packages.txt` conflict on `libglib2.0-0`** — this package name has
   moved/renamed on newer Debian base images (multiarch "t64" transition)
   and isn't installable as pinned. **Fix:** removed it from `packages.txt`
   and the `Dockerfile` — `libgl1`/`libsm6`/`libxext6`/`libxrender1` cover
   what OpenCV/Tesseract actually need.
2. **`uv` resolution failed because of `--extra-index-url .../whl/cpu`** —
   `uv` treats multiple package indexes very differently from `pip` and was
   picking bad/nonexistent versions of unrelated pinned packages (like
   `requests==2.32.3`) from the PyTorch index, failing the whole install.
   **Fix:** removed the extra index; `torch` now installs from plain PyPI
   (bigger download, but it actually resolves under `uv`).
3. **`lxml[html_clean]==5.2.2` failed to build** — no prebuilt wheel exists
   yet for bleeding-edge Python versions, so it fell back to compiling from
   source, which needs `libxml2`/`libxslt` *development* headers (not just
   the runtime libs). **Fix:** dropped the `[html_clean]` extra (not needed
   by `trafilatura`/`readability-lxml` here) and added `libxml2-dev` /
   `libxslt1-dev` to `packages.txt` as a safety net regardless.
4. **Python 3.14 compatibility risk** — `torch`, `transformers`, and other
   ML packages don't reliably ship wheels for Python 3.14 yet. `runtime.txt`
   in this repo requests 3.11, but **some Streamlit Cloud accounts only
   honor the Python version chosen in the app's "Advanced settings" dropdown
   at deploy time, not the file.** If your app was already deployed on
   3.14, you generally need to **delete and re-create the app** on Streamlit
   Cloud, explicitly picking Python 3.11 in Advanced settings before the
   first deploy (you can't change an existing app's Python version after
   the fact).
5. **Heavy dependency stack risk** — `torch` + `transformers` +
   `sentence-transformers` + `faiss-cpu` is inherently a lot for a free
   tier. **Fix:** `easyocr` (which pulls in a second matching `torchvision`
   build on top of all that) is no longer installed by default — Tesseract
   (via the tiny `pytesseract` binding) is the default and only bundled OCR
   engine for cloud deploys now. EasyOCR still works in the code if you
   install it yourself in a roomier environment (Docker/local/bigger HF
   Space) — see the comments at the top of `requirements.txt` and the
   `Dockerfile`, which installs it there since Docker builds have more
   headroom.

General steps after any fix:
1. Check **Manage app → logs** for the actual failing line — the redacted
   in-app error only shows a symptom, not the cause.
2. Push the corrected `requirements.txt` / `packages.txt` / `runtime.txt`.
3. Use **Reboot app**, or if the Python version itself needs to change,
   delete and redeploy the app fresh so Advanced settings can be set before
   the first build.

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
