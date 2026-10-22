"""Retrieve passages and answer with inline [n] citations."""
import re

from langchain_openai import ChatOpenAI

from .companies import COMPANIES
from .config import CHAT_MODEL
from .store import search

SYSTEM = (
    "You answer questions about public companies using ONLY the numbered passages "
    "provided. Cite every factual claim with its passage number like [1] or [2][3]. "
    "If the passages do not contain the answer, reply exactly: "
    "\"I can't find that in the provided filings.\" Never use outside knowledge."
)


def detect_tickers(question: str) -> list[str]:
    q = question.lower()
    found = []
    for t, name in COMPANIES.items():
        if re.search(rf"\b{re.escape(name.lower())}\b", q) or re.search(rf"\b{t.lower()}\b", q):
            found.append(t)
    return found


def format_passages(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({h['ticker']} {h['doc_type']} {h['period']} {h['section']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )


def parse_citations(answer: str, n_passages: int) -> list[int]:
    nums = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in nums if 1 <= n <= n_passages)


def answer_question(question: str, k: int = 8) -> dict:
    hits = search(question, k=k, tickers=detect_tickers(question) or None)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    msg = llm.invoke([
        ("system", SYSTEM),
        ("human", f"Passages:\n{format_passages(hits)}\n\nQuestion: {question}"),
    ])
    answer = msg.content
    cited = parse_citations(answer, len(hits))
    return {
        "question": question,
        "answer": answer,
        "citations": [{"n": n, **{k_: hits[n - 1][k_] for k_ in
                       ("ticker", "doc_type", "period", "section", "source", "text")}}
                      for n in cited],
        "retrieved": hits,
    }
