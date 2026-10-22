"""LLM extraction of revenue, guidance, and risk factors into validated JSON."""
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI

from .config import CHAT_MODEL
from .store import search


class RevenueFigure(BaseModel):
    period: str = Field(description="Fiscal period, e.g. 'FY2025' or 'Q2 2025'")
    amount_usd_millions: float
    evidence: str = Field(description="Short verbatim quote supporting the figure")


class Guidance(BaseModel):
    metric: str
    statement: str
    evidence: str


class RiskFactor(BaseModel):
    title: str
    summary: str = Field(max_length=400)


class FilingExtraction(BaseModel):
    ticker: str
    revenue: list[RevenueFigure] = []
    guidance: list[Guidance] = []
    risk_factors: list[RiskFactor] = Field(default=[], max_length=10)


PROMPT = (
    "Extract structured data for {ticker} strictly from the passages below. "
    "Only include revenue figures, forward guidance, and risk factors that are explicitly "
    "stated; leave lists empty rather than guessing. Convert revenue to USD millions. "
    "Evidence must be a verbatim quote.\n\n{passages}"
)


def extract_company(ticker: str, retries: int = 2) -> FilingExtraction:
    queries = ["total revenue net sales fiscal year", "guidance outlook expect next quarter",
               "principal risk factors"]
    seen, hits = set(), []
    for q in queries:
        for h in search(f"{ticker} {q}", k=6, tickers=[ticker]):
            if h["chunk_id"] not in seen:
                seen.add(h["chunk_id"])
                hits.append(h)
    passages = "\n\n".join(f"({h['doc_type']} {h['period']} {h['section']})\n{h['text']}"
                           for h in hits)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0).with_structured_output(FilingExtraction)
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            result = llm.invoke(PROMPT.format(ticker=ticker, passages=passages))
            return FilingExtraction.model_validate(result)  # schema-validated
        except ValidationError as e:
            last = e
    raise RuntimeError(f"extraction for {ticker} failed validation") from last
