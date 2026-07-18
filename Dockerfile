# AI-Historical-Translator-RAG
# Open-source stack: Streamlit + HF Transformers + FAISS/Chroma + NLLB/IndicTrans2
# + Tesseract/EasyOCR + live web retrieval (no bundled knowledge base).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models/hf_cache

WORKDIR /app

# System deps: Tesseract (+ common Indic language data), OpenCV runtime libs,
# and build tools needed by some Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-san \
    tesseract-ocr-tam \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models database assets

EXPOSE 8501

HEALTHCHECK --interval=60s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
