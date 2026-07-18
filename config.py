"""
config.py
Central configuration for AI-Historical-Translator-RAG.

All values can be overridden via environment variables (see .env.example).
Nothing here requires a paid/proprietary API key by default -- everything
is wired to open-source models and free web search, but you can swap in
hosted providers by changing the relevant *_PROVIDER value.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is a soft dependency; if it's missing we just rely on
    # whatever is already in os.environ.
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
DB_DIR = BASE_DIR / "database"

for d in (DATA_DIR, MODELS_DIR, ASSETS_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [v.strip() for v in val.split(",") if v.strip()]


@dataclass
class Settings:
    # ---- General ----
    APP_NAME: str = "AI Historical Translator & RAG"
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = _bool("DEBUG", True)

    # ---- LLM ----
    # Any causal / seq2seq model on the Hugging Face Hub. Small default so
    # the app is runnable on CPU out of the box; swap for something bigger
    # (e.g. Qwen2.5-7B-Instruct, Mistral-7B-Instruct) if you have a GPU.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "huggingface")
    # Small on purpose: Streamlit Community Cloud's free tier is ~1GB RAM /
    # 1 CPU, and a 1.5B model in fp32 alone can need 5-6GB. If you're
    # running locally or on a bigger box (Docker/your own server/HF Spaces
    # with more RAM), bump this via .env, e.g. Qwen/Qwen2.5-1.5B-Instruct
    # or Qwen/Qwen2.5-7B-Instruct.
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_DEVICE: str = os.getenv("LLM_DEVICE", "auto")  # auto | cpu | cuda

    # ---- Embeddings ----
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
    )
    EMBEDDING_MODEL_NAME_MULTILINGUAL: str = os.getenv(
        "EMBEDDING_MODEL_NAME_MULTILINGUAL", "BAAI/bge-m3"
    )
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # ---- Vector store ----
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "faiss")  # faiss | chroma
    VECTOR_INDEX_PATH: str = os.getenv(
        "VECTOR_INDEX_PATH", str(DATA_DIR / "vector_index")
    )
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "historical_rag")

    # ---- Translation ----
    TRANSLATION_MODEL_GENERAL: str = os.getenv(
        "TRANSLATION_MODEL_GENERAL", "facebook/nllb-200-distilled-600M"
    )
    TRANSLATION_MODEL_INDIC: str = os.getenv(
        "TRANSLATION_MODEL_INDIC", "ai4bharat/indictrans2-indic-en-1B"
    )
    DEFAULT_SRC_LANG: str = os.getenv("DEFAULT_SRC_LANG", "auto")
    DEFAULT_TGT_LANG: str = os.getenv("DEFAULT_TGT_LANG", "eng_Latn")

    # ---- OCR ----
    # "tesseract" is the default because it's the only OCR engine bundled in
    # requirements.txt for cloud deploys (EasyOCR pulls in torchvision on
    # top of an already heavy torch/transformers stack). If you `pip
    # install easyocr torchvision` yourself (e.g. Docker/local/bigger HF
    # Space), you can switch this to "easyocr" or "both" via .env.
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "tesseract")  # tesseract | easyocr | both
    OCR_LANGUAGES: List[str] = field(
        default_factory=lambda: _list("OCR_LANGUAGES", ["en", "hi", "sa"])
    )
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

    # ---- Web retrieval (this project intentionally has NO local knowledge
    # base -- the RAG pipeline sources context live from the web) ----
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "8"))
    WEB_FETCH_TIMEOUT: int = int(os.getenv("WEB_FETCH_TIMEOUT", "12"))
    WEB_FETCH_MAX_CHARS: int = int(os.getenv("WEB_FETCH_MAX_CHARS", "6000"))
    WEB_USER_AGENT: str = os.getenv(
        "WEB_USER_AGENT",
        "Mozilla/5.0 (compatible; HistoricalTranslatorBot/1.0; "
        "+https://github.com/your-org/ai-historical-translator-rag)",
    )
    TRUSTED_DOMAINS: List[str] = field(
        default_factory=lambda: _list(
            "TRUSTED_DOMAINS",
            [
                "wikipedia.org",
                "wikisource.org",
                "archive.org",
                "britannica.com",
                "jstor.org",
                "asi.nic.in",          # Archaeological Survey of India
                "whc.unesco.org",
                "sanskritdocuments.org",
                "gretil.sub.uni-goettingen.de",
            ],
        )
    )
    ALLOW_UNTRUSTED_DOMAINS: bool = _bool("ALLOW_UNTRUSTED_DOMAINS", True)

    # ---- Database ----
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{DB_DIR / 'app.db'}"
    )

    # ---- Health checks ----
    HEALTHCHECK_INTERVAL_SECONDS: int = int(
        os.getenv("HEALTHCHECK_INTERVAL_SECONDS", "300")
    )

    # ---- Streamlit / UI ----
    UI_TITLE: str = os.getenv("UI_TITLE", "🏺 AI Historical Translator & RAG")
    UI_LAYOUT: str = os.getenv("UI_LAYOUT", "wide")

    # ---- Supported historical / Indic scripts ----
    SUPPORTED_SCRIPTS: List[str] = field(
        default_factory=lambda: [
            "Devanagari",
            "Grantha",
            "Tamil-Brahmi",
            "Brahmi",
            "Kharosthi",
            "Sharada",
            "Modi",
            "Latin",
        ]
    )
    # Scripts we explicitly warn users about -- current open-source OCR/LLM
    # stacks are NOT reliably accurate on these without fine-tuning.
    LOW_CONFIDENCE_SCRIPTS: List[str] = field(
        default_factory=lambda: ["Brahmi", "Grantha", "Kharosthi", "Sharada"]
    )


settings = Settings()
