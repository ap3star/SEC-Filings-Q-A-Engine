"""Evaluation harness: retrieval recall, answer accuracy, hallucination rate."""
import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from .companies import COMPANIES
from .config import CHAT_MODEL, ROOT
from .rag import answer_question

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
ABSTAIN = "can't find that in the provided filings"

# (question template, expected 10-K section)
TEMPLATES = [
    ("What are the main risk factors {name} discloses in its latest 10-K?", "Item 1A"),
    ("What does {name} say its business and main products or segments are?", "Item 1"),
    ("What does {name}'s MD&A say drove changes in revenue and operating results?", "Item 7"),
    ("What legal proceedings does {name} describe in its latest 10-K?", "Item 3"),
]


def generate_questions() -> list[dict]:
    """100 labeled questions: 25 companies x 4 templates.

    Labels are the expected ticker/doc_type/section (for retrieval recall).
    Add a "reference" string to a question to include it in accuracy scoring.
    """
    return [
        {"id": f"{t}-{i}", "question": tpl.format(name=name), "ticker": t,
         "doc_type": "10-K", "section": section, "reference": None}
        for t, name in COMPANIES.items()
        for i, (tpl, section) in enumerate(TEMPLATES)
    ]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retrieval_hit(q: dict, retrieved: list[dict]) -> bool:
    return any(h["ticker"] == q["ticker"] and h["doc_type"] == q["doc_type"]
               and h["section"] == q["section"] for h in retrieved)


def is_abstention(answer: str) -> bool:
    return ABSTAIN in answer.lower().replace("’", "'")


def _judge(prompt: str) -> dict:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})
    return json.loads(llm.invoke(prompt).content)


def judge_accuracy(question: str, answer: str, reference: str) -> bool:
    out = _judge(
        'Return JSON {"correct": true|false}. Is the ANSWER consistent with the REFERENCE '
        f"and does it contain the key facts?\nQUESTION: {question}\nREFERENCE: {reference}\n"
        f"ANSWER: {answer}")
    return bool(out.get("correct"))


def judge_hallucination(answer: str, passages: list[str]) -> bool:
    """True if the answer contains any claim not supported by the cited passages."""
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    out = _judge(
        'Return JSON {"unsupported": true|false}. Is ANY factual claim in the ANSWER '
        f"unsupported by the PASSAGES?\nPASSAGES:\n{ctx}\n\nANSWER: {answer}")
    return bool(out.get("unsupported"))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    graded = [r for r in rows if r["correct"] is not None]
    answered = [r for r in rows if not r["abstained"]]
    return {
        "n": n,
        "recall_at_k": sum(r["retrieval_hit"] for r in rows) / n if n else 0.0,
        "accuracy": (sum(r["correct"] for r in graded) / len(graded)) if graded else None,
        "accuracy_n": len(graded),
        "hallucination_rate": (sum(r["hallucinated"] for r in answered) / len(answered))
        if answered else 0.0,
        "abstention_rate": 1 - len(answered) / n if n else 0.0,
    }


def run_eval(path: Path = QUESTIONS_PATH, limit: int | None = None, k: int = 8) -> dict:
    rows = []
    for q in load_questions(path)[:limit]:
        res = answer_question(q["question"], k=k)
        abstained = is_abstention(res["answer"])
        cited = [c["text"] for c in res["citations"]]
        rows.append({
            "id": q["id"],
            "retrieval_hit": retrieval_hit(q, res["retrieved"]),
            "abstained": abstained,
            "correct": judge_accuracy(q["question"], res["answer"], q["reference"])
            if q.get("reference") else None,
            "hallucinated": (not abstained) and (not cited or judge_hallucination(res["answer"], cited)),
            "answer": res["answer"],
        })
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summarize(rows), "rows": rows}
    (out_dir / "latest.json").write_text(json.dumps(report, indent=2))
    return report["summary"]
