from sec_qa.evaluate import generate_questions, is_abstention, retrieval_hit, summarize


def test_generates_100_unique_questions_over_25_companies():
    qs = generate_questions()
    assert len(qs) == 100
    assert len({q["id"] for q in qs}) == 100
    assert len({q["ticker"] for q in qs}) == 25


def test_retrieval_hit_requires_ticker_and_section():
    q = {"ticker": "AAPL", "doc_type": "10-K", "section": "Item 1A"}
    ok = {"ticker": "AAPL", "doc_type": "10-K", "section": "Item 1A"}
    assert retrieval_hit(q, [ok])
    assert not retrieval_hit(q, [{**ok, "ticker": "MSFT"}])
    assert not retrieval_hit(q, [{**ok, "section": "Item 7"}])


def test_abstention_detection():
    assert is_abstention("I can’t find that in the provided filings.")
    assert not is_abstention("Revenue was $5B [1].")


def test_summarize_metrics():
    rows = [
        {"retrieval_hit": True, "abstained": False, "correct": True, "hallucinated": False},
        {"retrieval_hit": False, "abstained": False, "correct": False, "hallucinated": True},
        {"retrieval_hit": True, "abstained": True, "correct": None, "hallucinated": False},
        {"retrieval_hit": True, "abstained": False, "correct": None, "hallucinated": False},
    ]
    s = summarize(rows)
    assert s["recall_at_k"] == 0.75
    assert s["accuracy"] == 0.5 and s["accuracy_n"] == 2
    assert abs(s["hallucination_rate"] - 1 / 3) < 1e-9
