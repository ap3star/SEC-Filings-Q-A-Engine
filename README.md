# SEC Filings Q&A Engine

RAG system over **10-Ks** and **earnings call transcripts** for 25 companies. Answers carry
cited passages, key figures are extracted into schema-validated JSON, and an evaluation
harness scores retrieval recall, accuracy, and hallucination rate.

Stack: Python, OpenAI API, LangChain, Elasticsearch.

## Pipeline

```
EDGAR 10-K HTML ─┐                                   ┌─ ask:     hybrid retrieve → cited answer
                 ├─ parse → split by Item → chunk ─ embed → Elasticsearch ─┼─ extract: revenue / guidance / risks → JSON (pydantic)
transcripts .txt ┘                                   └─ eval:    recall@k, accuracy, hallucination rate
```

Retrieval is hybrid: BM25 (`match`) plus cosine kNN over `text-embedding-3-small`
vectors, filtered by ticker when a company is mentioned in the question.

## Quick start

```bash
cp .env.example .env            # add OPENAI_API_KEY and a real SEC_USER_AGENT
docker compose up -d            # Elasticsearch on :9200
python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'

sec-qa ingest                   # download latest 10-K for 25 companies + load transcripts
sec-qa index --recreate         # embed + index
sec-qa ask "What supply chain risks does NVIDIA disclose?"
sec-qa extract AAPL             # schema-validated JSON
sec-qa eval                     # writes eval/results/latest.json
pytest                          # offline unit tests
```

## Transcripts

There is no free, redistributable transcript API, so drop transcript text files into
`data/transcripts/` named `TICKER_YYYYQn.txt` (e.g. `AAPL_2025Q2.txt`). They are split into
Prepared Remarks and Q&A and indexed alongside the 10-Ks.

## Evaluation

`sec-qa gen-questions` writes `eval/questions.jsonl`: 100 questions (25 companies × 4
templates), each labeled with the expected ticker, form, and 10-K item.

- **Recall@k**: an expected-ticker, expected-section chunk is among the retrieved passages.
- **Accuracy**: LLM-judged against a `reference` answer. Generated questions have
  `reference: null` and are skipped here until you fill in references (`accuracy_n` reports how many were graded).
- **Hallucination rate**: share of non-abstaining answers where an LLM judge finds a claim
  unsupported by the cited passages (uncited answers count as hallucinated).

## Status

Minimal but functional. Known gaps: only the latest 10-K per company, no table-aware
parsing of financial statements, no reranker.
