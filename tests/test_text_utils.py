from utils.text_utils import chunk_text, clean_whitespace, truncate


def test_clean_whitespace_collapses_blank_lines():
    raw = "line1\n\n\n\nline2   with   spaces"
    cleaned = clean_whitespace(raw)
    assert "\n\n\n" not in cleaned
    assert "line1" in cleaned and "line2" in cleaned


def test_chunk_text_respects_overlap_and_nonempty():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_truncate_short_text_unchanged():
    text = "short text"
    assert truncate(text, 100) == text


def test_truncate_long_text_adds_ellipsis():
    text = "word " * 100
    truncated = truncate(text, 20)
    assert truncated.endswith("…")
    assert len(truncated) <= 22
