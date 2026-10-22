"""HTML -> text, 10-K item segmentation, transcript loading."""
import re
from pathlib import Path

from bs4 import BeautifulSoup

# Matches headings like "Item 1A." / "ITEM 7 -" at the start of a line
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[ab]?)\s*[\.\:\-–—]", re.IGNORECASE | re.MULTILINE)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def split_items(text: str) -> list[tuple[str, str]]:
    """Split a 10-K into (section, body) pairs.

    The table of contents also matches ITEM_RE, so for each item we keep the
    longest body, which is the real section rather than the TOC line.
    """
    matches = list(ITEM_RE.finditer(text))
    best: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = f"Item {m.group(1).upper()}"
        body = text[m.end():end].strip()
        if len(body) > len(best.get(name, "")):
            best[name] = body
    return list(best.items()) or [("Full Document", text)]


TRANSCRIPT_NAME_RE = re.compile(r"^([A-Z\-]+)_(\d{4}Q[1-4])\.txt$")


def load_transcripts(directory: Path) -> list[tuple[str, str, str, str]]:
    """Return (ticker, period, text, path) for files named TICKER_YYYYQn.txt."""
    out = []
    for p in sorted(directory.glob("*.txt")):
        m = TRANSCRIPT_NAME_RE.match(p.name)
        if m:
            out.append((m.group(1), m.group(2), p.read_text(), str(p)))
    return out


def split_transcript(text: str) -> list[tuple[str, str]]:
    """Split on a Q&A marker into prepared remarks and Q&A."""
    m = re.search(r"^.*(question[- ]and[- ]answer|q&a session|questions? and answers?).*$",
                  text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return [("Transcript", text)]
    return [("Prepared Remarks", text[:m.start()]), ("Q&A", text[m.start():])]
