from translation.lang_codes import is_indic, supported_language_labels, to_indictrans_code, to_nllb_code


def test_english_not_indic():
    assert is_indic("english") is False


def test_sanskrit_is_indic():
    assert is_indic("Sanskrit") is True  # case-insensitive


def test_nllb_code_lookup():
    assert to_nllb_code("hindi") == "hin_Deva"
    assert to_nllb_code("not_a_real_language") is None


def test_indictrans_code_lookup():
    assert to_indictrans_code("tamil") == "tam_Taml"


def test_supported_language_labels_sorted_and_nonempty():
    labels = supported_language_labels()
    assert labels == sorted(labels)
    assert "english" in labels
    assert "sanskrit" in labels
