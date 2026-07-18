from unittest.mock import patch

from web.search import SearchHit, _domain_of, _is_trusted, web_search


def test_domain_of_strips_www():
    assert _domain_of("https://www.wikipedia.org/wiki/X") == "wikipedia.org"
    assert _domain_of("https://archive.org/details/y") == "archive.org"


def test_is_trusted_matches_configured_domains():
    assert _is_trusted("wikipedia.org") is True
    assert _is_trusted("some-random-blog.example") is False


@patch("web.search._ddg_search")
def test_web_search_sorts_trusted_first(mock_ddg):
    mock_ddg.return_value = [
        {"title": "Random blog", "href": "https://random-blog.example/a", "body": "..."},
        {"title": "Wikipedia article", "href": "https://en.wikipedia.org/wiki/Test", "body": "..."},
    ]
    hits = web_search("test query", max_results=5, prefer_trusted=True)
    assert isinstance(hits[0], SearchHit)
    assert hits[0].trusted is True
    assert hits[0].domain == "en.wikipedia.org" or hits[0].domain.endswith("wikipedia.org")
