import json
import logging

from .chunking import chunk_sections
from .companies import COMPANIES
from .config import PROCESSED_DIR, TRANSCRIPT_DIR
from .edgar import download_10k
from .models import Chunk
from .parse import html_to_text, load_transcripts, split_items, split_transcript

log = logging.getLogger(__name__)


def ingest_10k(ticker: str) -> list[Chunk]:
    html, meta = download_10k(ticker)
    sections = split_items(html_to_text(html))
    return chunk_sections(sections, ticker, "10-K", meta["period"], meta["url"])


def ingest_transcripts() -> list[Chunk]:
    chunks = []
    for ticker, period, text, path in load_transcripts(TRANSCRIPT_DIR):
        chunks += chunk_sections(split_transcript(text), ticker, "transcript", period, path)
    return chunks


def run_ingest(tickers: list[str] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for t in tickers or COMPANIES:
        try:
            got = ingest_10k(t)
            log.info("%s: %d 10-K chunks", t, len(got))
            chunks += got
        except Exception as e:  # keep going; one bad filing shouldn't stop the run
            log.warning("%s: 10-K failed: %s", t, e)
    chunks += ingest_transcripts()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    return chunks


def load_chunks() -> list[Chunk]:
    path = PROCESSED_DIR / "chunks.jsonl"
    return [Chunk(**json.loads(line)) for line in path.read_text().splitlines()]
