"""The CI gate for model results. No model runs here.

First, every committed `results/notes/<set>.json` (condition C's notes) must be what replaying
its recordings writes. Then, for every committed `results/<set>/<condition>/predictions.jsonl`:
1. Replay the condition from the cassettes with this branch's code, prompts and settings, and
   check the answers are exactly the committed ones. A prompt changed without re-recording
   stops here (no recording), and so does code that would score the recordings differently.
2. If the target branch has results for the same set and condition, compare the two with the
   paired bootstrap. The gate fails on a significant drop in component, area, issue type or
   duplicate accuracy. The duplicate and non-duplicate halves are shown but don't gate: a
   change that finds more originals at the cost of a few false alarms is a trade-off, not a
   regression, and the overall duplicate accuracy already weighs both.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tme.compare import compare, load_predictions, report
from tme.conditions import CONDITIONS
from tme.llm import CassetteMiss
from tme.notes import write_notes
from tme.run import run

GATING_FIELDS = ("component", "area", "issue_type", "duplicate")


def gate(results: Path, baseline: Path | None, tickets: list[dict], split: dict, config: dict,
         prompts: dict, cassettes: Path = Path("cassettes")) -> tuple[bool, str]:
    ok, lines = True, ["## Model results gate", ""]
    for path in sorted((results / "notes").glob("*.json")):
        set_name = path.stem
        try:
            rebuilt = write_notes(tickets, set_name, config, prompts["write_notes"], "replay", cassettes=cassettes)
        except CassetteMiss as exc:
            ok = False
            lines += [f"❌ Notes for {set_name}: {exc}", ""]
            continue
        if rebuilt != json.loads(path.read_text(encoding="utf-8")):
            ok = False
            lines += [f"❌ Replaying the note recordings no longer gives `{path.as_posix()}`. "
                      f"Re-run `tme notes {set_name} --mode replay` and commit the result.", ""]
        else:
            lines += [f"Notes for {set_name} replay to the committed notes.", ""]
    committed = sorted(p for p in results.glob("*/*/predictions.jsonl")
                       if p.parts[-3] in split and p.parts[-2] in CONDITIONS)
    if not committed:
        return ok, "\n".join(lines + ["No model results committed yet: nothing to check.", ""])
    for path in committed:
        set_name, condition = path.parts[-3], path.parts[-2]
        lines.append(f"### {condition} on {set_name}")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                run(condition, set_name, tickets, split, config, prompts, "replay",
                    cassettes=cassettes, out=Path(tmp), notes_folder=results / "notes")
            except CassetteMiss as exc:
                ok = False
                lines += [f"❌ {exc}", ""]
                continue
            replayed = (Path(tmp) / "predictions.jsonl").read_text(encoding="utf-8")
        if replayed != path.read_text(encoding="utf-8"):
            ok = False
            lines += [f"❌ Replaying the recordings no longer gives the committed `{path.as_posix()}`. "
                      f"Re-run `tme run {condition} {set_name} --mode replay --out {path.parent.as_posix()}` "
                      "and commit the result.", ""]
            continue
        lines.append("Recordings replay to the committed answers.")
        target = baseline / set_name / condition / "predictions.jsonl" if baseline else None
        if not target or not target.exists():
            lines += ["No results for this condition on the target branch: nothing to compare.", ""]
            continue
        fields = compare(split[set_name], load_predictions(target), load_predictions(path))
        failed = [r.field for r in fields if r.verdict == "fail" and r.field in GATING_FIELDS]
        ok = ok and not failed
        lines += ["", report(fields, "target branch", "this branch", len(split[set_name])),
                  f"**{'BLOCKED: significant drop in ' + ', '.join(failed) if failed else 'PASSED'}**", ""]
    return ok, "\n".join(lines)
