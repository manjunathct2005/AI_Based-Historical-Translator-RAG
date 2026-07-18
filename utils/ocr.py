"""
utils/ocr.py

OCR extraction for scanned historical documents / inscriptions using
Tesseract and/or EasyOCR. Both are open-source and run fully offline
once their language data / model weights are downloaded.

NOTE on scripts like Brahmi, Grantha, Kharosthi, Sharada: neither engine
ships reliable models for these out of the box. We still run OCR (so the
pipeline doesn't break), but we surface a confidence warning so users
don't mistake garbage output for a real reading. See config.LOW_CONFIDENCE_SCRIPTS.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    engine: str
    confidence: Optional[float]  # 0-100, None if engine doesn't expose one
    warning: Optional[str] = None


class OCREngine:
    """Thin wrapper that dispatches to Tesseract and/or EasyOCR."""

    def __init__(self, engine: str = None, languages: List[str] = None):
        self.engine = (engine or settings.OCR_ENGINE).lower()
        self.languages = languages or settings.OCR_LANGUAGES
        self._easyocr_reader = None  # lazy-loaded, it's slow to init

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #
    def extract(self, image_bytes: bytes) -> OCRResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.engine == "tesseract":
            return self._run_tesseract(image)
        elif self.engine == "easyocr":
            return self._run_easyocr(image)
        elif self.engine == "both":
            tess = self._run_tesseract(image)
            easy = self._run_easyocr(image)
            # Prefer whichever produced more text; a crude but effective
            # heuristic for messy historical scans.
            best = easy if len(easy.text) >= len(tess.text) else tess
            best.text = (
                f"[Tesseract]\n{tess.text}\n\n[EasyOCR]\n{easy.text}"
                if settings.DEBUG
                else best.text
            )
            return best
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine}")

    # ---------------------------------------------------------------- #
    # Engines
    # ---------------------------------------------------------------- #
    def _run_tesseract(self, image: Image.Image) -> OCRResult:
        try:
            import pytesseract

            if settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

            lang_map = {"en": "eng", "hi": "hin", "sa": "san", "ta": "tam"}
            tess_langs = "+".join(
                lang_map.get(l, l) for l in self.languages
            ) or "eng"

            data = pytesseract.image_to_data(
                image, lang=tess_langs, output_type=pytesseract.Output.DICT
            )
            words = [w for w in data["text"] if w.strip()]
            confs = [int(c) for c in data["conf"] if c not in ("-1", -1)]
            text = " ".join(words)
            avg_conf = float(np.mean(confs)) if confs else None

            warning = self._low_confidence_warning(avg_conf)
            return OCRResult(text=text, engine="tesseract",
                              confidence=avg_conf, warning=warning)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Tesseract OCR failed")
            return OCRResult(
                text="",
                engine="tesseract",
                confidence=None,
                warning=f"Tesseract OCR failed: {exc}",
            )

    def _run_easyocr(self, image: Image.Image) -> OCRResult:
        try:
            import easyocr

            if self._easyocr_reader is None:
                self._easyocr_reader = easyocr.Reader(
                    self.languages, gpu=False, verbose=False
                )

            arr = np.array(image)
            results = self._easyocr_reader.readtext(arr, detail=1)
            if not results:
                return OCRResult(text="", engine="easyocr", confidence=None,
                                  warning="No text detected.")

            texts = [r[1] for r in results]
            confs = [r[2] * 100 for r in results]
            avg_conf = float(np.mean(confs)) if confs else None

            warning = self._low_confidence_warning(avg_conf)
            return OCRResult(text=" ".join(texts), engine="easyocr",
                              confidence=avg_conf, warning=warning)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("EasyOCR failed")
            return OCRResult(
                text="",
                engine="easyocr",
                confidence=None,
                warning=f"EasyOCR failed: {exc}",
            )

    @staticmethod
    def _low_confidence_warning(avg_conf: Optional[float]) -> Optional[str]:
        if avg_conf is not None and avg_conf < 55:
            return (
                "Low OCR confidence. If this is a damaged inscription or an "
                "ancient script (Brahmi/Grantha/Kharosthi/Sharada), current "
                "open-source OCR is not reliable without specialized "
                "fine-tuning -- treat this output as a rough draft only."
            )
        return None
