from sec_qa.rag import detect_tickers, format_passages, parse_citations
from sec_qa.store import build_query


def test_detect_tickers_by_name_and_symbol():
    assert detect_tickers("What did Apple say about NVDA supply?") == ["AAPL", "NVDA"]
    assert detect_tickers("What is inflation?") == []


def test_parse_citations_ignores_out_of_range():
    assert parse_citations("Revenue grew [1][3] and [9].", 3) == [1, 3]


def test_build_query_applies_filters_to_both_legs():
    q = build_query("risk", [0.1], 5, tickers=["AAPL"], doc_type="10-K")
    assert q["knn"]["filter"] == q["query"]["bool"]["filter"]
    assert len(q["knn"]["filter"]) == 2


def test_format_passages_numbers_from_one():
    hit = dict(ticker="A", doc_type="10-K", period="FY1", section="Item 1", text="x")
    assert format_passages([hit]).startswith("[1] (A 10-K FY1 Item 1)")
