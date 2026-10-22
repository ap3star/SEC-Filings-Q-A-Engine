from sec_qa.chunking import chunk_sections
from sec_qa.parse import html_to_text, split_items, split_transcript


def test_split_items_prefers_real_section_over_toc():
    text = ("Item 1. Business\nItem 1A. Risk Factors\n"
            "Item 1. Business\n" + "We sell widgets. " * 20 +
            "\nItem 1A. Risk Factors\n" + "Supply chain risk. " * 20)
    sections = dict(split_items(text))
    assert "widgets" in sections["Item 1"]
    assert "Supply chain" in sections["Item 1A"]


def test_html_to_text_strips_scripts():
    assert "bad" not in html_to_text("<html><script>bad()</script><p>ok</p></html>")


def test_split_transcript():
    parts = dict(split_transcript("Hello.\nQuestion-and-Answer Session\nQ: hi"))
    assert set(parts) == {"Prepared Remarks", "Q&A"}


def test_chunk_metadata():
    chunks = chunk_sections([("Item 1A", "risk " * 1000)], "AAPL", "10-K", "FY2025", "u")
    assert len(chunks) > 1
    assert all(c.ticker == "AAPL" and c.section == "Item 1A" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
