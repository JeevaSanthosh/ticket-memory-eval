"""Command line: `tme split` and `tme baselines`, run from the repository root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tme import baselines
from tme.conditions import CONDITIONS
from tme.data import DATA, load_tickets
from tme.llm import OllamaClient
from tme.run import load_config, load_prompts, run
from tme.split import build_split

SPLIT = Path("data/split.json")
BASELINES = Path("results/baselines.json")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_split(args) -> None:
    split = build_split(load_tickets(args.data))
    write_json(args.out, split)
    for name, gold in split.items():
        duplicates = sum(bool(g["duplicate_of"]) for g in gold)
        print(f"{name}: {len(gold)} tickets ({duplicates} duplicates), "
              f"created {gold[0]['created'][:10]} to {gold[-1]['created'][:10]}")
    print(f"wrote {args.out}")


def cmd_baselines(args) -> None:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    results = baselines.run(load_tickets(args.data), split)
    write_json(args.out, results)
    columns = {"component": "component", "area": "area", "issue_type": "type", "duplicate": "duplicate",
               "duplicate_on_duplicates": "on dups", "duplicate_on_non_duplicates": "on non-dups"}
    print(f"duplicate threshold {results['duplicate_threshold']} (chosen on dev)")
    for name in split:
        r = results[name]
        print(f"\n{name} ({r['tickets']} tickets)")
        print(f"  {'':12s}" + "".join(f"{c:>12s}" for c in columns.values()))
        for baseline in ("most_common", "vote_of_5"):
            print(f"  {baseline:12s}" + "".join(f"{r[baseline][f]:12.3f}" for f in columns))
        print("  search: " + ", ".join(f"{k} {v}" for k, v in r["search"].items()))
    print(f"\nwrote {args.out}")


def cmd_run(args) -> None:
    config = load_config(args.config)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    client = OllamaClient() if args.mode == "record" else None
    print(f"condition {args.condition} on {args.set} with {config['model']} ({args.mode})", flush=True)
    results = run(args.condition, args.set, load_tickets(args.data), split, config, load_prompts(config),
                  args.mode, client=client, limit=args.limit, verbose=True)
    scores, ops = results["scores"], results["operations"]
    print("\n" + ", ".join(f"{k} {v}" for k, v in scores.items()))
    print(", ".join(f"{k} {v}" for k, v in ops.items()))
    print(f"wrote runs/{args.set}/{args.condition}/")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tme")
    parser.add_argument("--data", type=Path, default=DATA)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split", help="choose the dev and test tickets (fixed seed)")
    p.add_argument("--out", type=Path, default=SPLIT)
    p.set_defaults(func=cmd_split)
    p = sub.add_parser("baselines", help="score the answers that need no model")
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--out", type=Path, default=BASELINES)
    p.set_defaults(func=cmd_baselines)
    p = sub.add_parser("run", help="run one condition over dev or test, recording or replaying the model's answers")
    p.add_argument("condition", choices=CONDITIONS)
    p.add_argument("set", choices=["dev", "test"])
    p.add_argument("--mode", choices=["record", "replay"], default="record",
                   help="record: ask Ollama for anything not yet recorded; replay: recordings only (CI)")
    p.add_argument("--limit", type=int, help="only the first N tickets (e.g. a timing check)")
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--config", type=Path, default=Path("eval.toml"))
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
