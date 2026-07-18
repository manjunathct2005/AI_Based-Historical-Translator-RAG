"""
translation/lang_codes.py

Language code helpers. NLLB-200 uses FLORES-200 codes like "eng_Latn",
"hin_Deva", "san_Deva", "tam_Taml"; IndicTrans2 uses similar but not
identical codes. This module centralizes the mapping so the rest of the
app can just say "hindi" / "sanskrit" / "tamil" / "english".
"""

from __future__ import annotations

from typing import Optional

# Human label -> (NLLB / FLORES-200 code, IndicTrans2 code)
LANGUAGES = {
    "english":    ("eng_Latn", "eng_Latn"),
    "hindi":      ("hin_Deva", "hin_Deva"),
    "sanskrit":   ("san_Deva", "san_Deva"),
    "tamil":      ("tam_Taml", "tam_Taml"),
    "telugu":     ("tel_Telu", "tel_Telu"),
    "kannada":    ("kan_Knda", "kan_Knda"),
    "malayalam":  ("mal_Mlym", "mal_Mlym"),
    "bengali":    ("ben_Beng", "ben_Beng"),
    "marathi":    ("mar_Deva", "mar_Deva"),
    "gujarati":   ("guj_Gujr", "guj_Gujr"),
    "punjabi":    ("pan_Guru", "pan_Guru"),
    "odia":       ("ory_Orya", "ory_Orya"),
    "urdu":       ("urd_Arab", "urd_Arab"),
    "nepali":     ("npi_Deva", "npi_Deva"),
    "sinhala":    ("sin_Sinh", "sin_Sinh"),
}

INDIC_LANGS = set(LANGUAGES.keys()) - {"english"}


def to_nllb_code(label: str) -> Optional[str]:
    entry = LANGUAGES.get(label.strip().lower())
    return entry[0] if entry else None


def to_indictrans_code(label: str) -> Optional[str]:
    entry = LANGUAGES.get(label.strip().lower())
    return entry[1] if entry else None


def is_indic(label: str) -> bool:
    return label.strip().lower() in INDIC_LANGS


def supported_language_labels():
    return sorted(LANGUAGES.keys())
