"""
translation/translator.py

Routes translation requests to:
  - IndicTrans2 when both source/target are within its Indic<->English
    training scope (generally higher quality for Indian languages), or
  - NLLB-200 as the general-purpose fallback (200 languages incl. many
    Indic + historical-adjacent languages).

Both are open-source Hugging Face models, loaded lazily and cached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from config import settings
from translation.lang_codes import to_nllb_code, to_indictrans_code, is_indic

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_used: str
    warning: Optional[str] = None


class _NLLBTranslator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.TRANSLATION_MODEL_GENERAL
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading NLLB translation model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def translate(self, text: str, src_code: str, tgt_code: str) -> str:
        self._load()
        self._tokenizer.src_lang = src_code
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        generated = self._model.generate(
            **encoded, forced_bos_token_id=forced_bos_token_id, max_length=512
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


class _IndicTrans2Translator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.TRANSLATION_MODEL_INDIC
        self._tokenizer = None
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading IndicTrans2 model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)

        try:
            from IndicTransToolkit.processor import IndicProcessor

            self._processor = IndicProcessor(inference=True)
        except ImportError:
            logger.warning(
                "IndicTransToolkit not installed; falling back to plain tokenization "
                "for IndicTrans2 (quality may be reduced). "
                "Install with: pip install IndicTransToolkit"
            )
            self._processor = None

    def translate(self, text: str, src_code: str, tgt_code: str) -> str:
        self._load()

        if self._processor is not None:
            batch = self._processor.preprocess_batch([text], src_lang=src_code, tgt_lang=tgt_code)
        else:
            batch = [text]

        encoded = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        generated = self._model.generate(**encoded, max_length=512, num_beams=5)
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

        if self._processor is not None:
            decoded = self._processor.postprocess_batch(decoded, lang=tgt_code)

        return decoded[0]


@lru_cache(maxsize=1)
def _nllb() -> _NLLBTranslator:
    return _NLLBTranslator()


@lru_cache(maxsize=1)
def _indictrans() -> _IndicTrans2Translator:
    return _IndicTrans2Translator()


def translate(text: str, source_lang: str, target_lang: str) -> TranslationResult:
    """
    source_lang / target_lang are human labels, e.g. "sanskrit", "english",
    "tamil" -- see translation/lang_codes.py for the full supported list.
    """
    warning = None

    if is_indic(source_lang) or is_indic(target_lang):
        src = to_indictrans_code(source_lang)
        tgt = to_indictrans_code(target_lang)
        if src and tgt:
            try:
                translated = _indictrans().translate(text, src, tgt)
                return TranslationResult(text, translated, source_lang, target_lang,
                                          model_used=settings.TRANSLATION_MODEL_INDIC)
            except Exception as exc:
                logger.exception("IndicTrans2 failed, falling back to NLLB")
                warning = f"IndicTrans2 failed ({exc}); used NLLB-200 fallback."

    src = to_nllb_code(source_lang)
    tgt = to_nllb_code(target_lang)
    if not src or not tgt:
        raise ValueError(
            f"Unsupported language pair: {source_lang} -> {target_lang}. "
            f"See translation/lang_codes.py for supported languages."
        )

    translated = _nllb().translate(text, src, tgt)
    return TranslationResult(text, translated, source_lang, target_lang,
                              model_used=settings.TRANSLATION_MODEL_GENERAL,
                              warning=warning)
