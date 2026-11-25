"""Command line: `python -m medtrace.cli ingest data/sample` / `ask "..."` / `eval`."""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .pipeline import MedTrace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medtrace")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="Build the index from a folder or file")
    p_ing.add_argument("path", nargs="?", default="data/sample")
    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="Print the full JSON answer")
    p_eval = sub.add_parser("eval", help="Run the eval set")
    p_eval.add_argument("--dataset", default="eval/questions.jsonl")
    p_eval.add_argument("--out", default="eval/results.json")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        engine = MedTrace.build(args.path)
        print(f"Indexed {len(engine.index)} chunks into {get_settings().index_dir}/")
    elif args.cmd == "ask":
        ans = MedTrace.from_settings().ask(args.question)
        if args.json:
            print(ans.model_dump_json(indent=2))
        else:
            print(ans.answer)
            for c in ans.citations:
                loc = c.source + (f", p.{c.page}" if c.page else "")
                print(f"  [{c.marker}] {loc}")
            if ans.abstained:
                print(f"  (abstained: {ans.reason})")
    elif args.cmd == "eval":
        from eval.run_eval import run

        report = run(MedTrace.from_settings(), args.dataset)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
