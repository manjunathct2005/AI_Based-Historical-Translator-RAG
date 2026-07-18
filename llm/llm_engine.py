"""
llm/llm_engine.py

Configurable open-source LLM wrapper (default: Qwen2.5-1.5B-Instruct via
Hugging Face `transformers`). Swap LLM_MODEL_NAME in .env for a bigger
model if you have the hardware -- the interface stays the same.

Kept deliberately provider-agnostic: if you later want to point this at a
hosted API instead of a local HF model, add a branch in `generate()` and
flip config.LLM_PROVIDER.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()


class LLMEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self._tokenizer = None
        self._model = None
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        with _load_lock:
            if self._pipeline is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            logger.info("Loading LLM: %s", self.model_name)
            device = self._resolve_device()

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            if device != "cuda":
                self._model.to(device)

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if device == "cuda" else -1,
            )

    @staticmethod
    def _resolve_device() -> str:
        if settings.LLM_DEVICE != "auto":
            return settings.LLM_DEVICE
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        self._ensure_loaded()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models/tokenizers without a chat template.
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"

        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens or settings.LLM_MAX_NEW_TOKENS,
            temperature=temperature or settings.LLM_TEMPERATURE,
            do_sample=(temperature or settings.LLM_TEMPERATURE) > 0,
            return_full_text=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"].strip()


@lru_cache(maxsize=1)
def get_llm_engine() -> LLMEngine:
    """Cached singleton so Streamlit reruns don't reload the model each time."""
    return LLMEngine()
