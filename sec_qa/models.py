from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    doc_type: str  # "10-K" | "transcript"
    period: str  # fiscal year or quarter label, e.g. "FY2025" / "2025Q2"
    section: str  # e.g. "Item 1A" or "Prepared Remarks"
    source: str  # URL or file path
    embedding: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return asdict(self)
