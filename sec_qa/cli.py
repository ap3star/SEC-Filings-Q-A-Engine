import argparse
import json
import logging

from .companies import COMPANIES


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sec-qa")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="download 10-Ks, load transcripts, chunk")
    s.add_argument("--tickers", nargs="*")
    s = sub.add_parser("index", help="embed chunks and load into Elasticsearch")
    s.add_argument("--recreate", action="store_true")
    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    s = sub.add_parser("extract", help="extract revenue/guidance/risks as JSON")
    s.add_argument("ticker")
    s = sub.add_parser("gen-questions", help="write eval/questions.jsonl (100 questions)")
    s = sub.add_parser("eval", help="run the evaluation harness")
    s.add_argument("--limit", type=int)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "ingest":
        from .ingest import run_ingest
        print(f"{len(run_ingest(args.tickers))} chunks written")
    elif args.cmd == "index":
        from .ingest import load_chunks
        from .store import index_chunks
        print(f"{index_chunks(load_chunks(), recreate=args.recreate)} chunks indexed")
    elif args.cmd == "ask":
        from .rag import answer_question
        r = answer_question(args.question)
        print(r["answer"], "\n")
        for c in r["citations"]:
            print(f"[{c['n']}] {c['ticker']} {c['doc_type']} {c['period']} {c['section']} - {c['source']}")
    elif args.cmd == "extract":
        from .extract import extract_company
        t = args.ticker.upper()
        if t not in COMPANIES:
            p.error(f"{t} is not in the 25-company universe")
        print(extract_company(t).model_dump_json(indent=2))
    elif args.cmd == "gen-questions":
        from .evaluate import QUESTIONS_PATH, generate_questions
        QUESTIONS_PATH.parent.mkdir(exist_ok=True)
        QUESTIONS_PATH.write_text("\n".join(json.dumps(q) for q in generate_questions()) + "\n")
        print(f"wrote {QUESTIONS_PATH}")
    elif args.cmd == "eval":
        from .evaluate import run_eval
        print(json.dumps(run_eval(limit=args.limit), indent=2))
