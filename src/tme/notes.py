"""Written notes for condition C: one line per component, written once by the model.

For each area (the part of a component's name before " / "), the model reads the summaries of the
20 most recent tickets in each of the area's components, all created before the set starts, and
writes one line per component on what its tickets are about. The notes are written once for a set
and never updated, so by the end of the test year they are up to a year old.
"""

from __future__ import annotations

import json
from pathlib import Path

from tme.data import area, mask
from tme.llm import Cassette
from tme.split import PERIODS

NOTE_TICKETS = 20
NOTES = Path("results/notes")


def note_requests(tickets: list[dict], set_name: str, config: dict, prompt: str) -> list[tuple[str, dict, dict]]:
    """(area, request, schema) for each area, from tickets created before the set starts."""
    start = PERIODS[set_name][0]
    before = [t for t in tickets if t["created"] < start]
    areas: dict[str, list[str]] = {}
    for component in sorted({c for t in before for c in t["components"]}):
        areas.setdefault(area(component), []).append(component)
    requests = []
    for name, components in areas.items():
        sections = []
        for c in components:
            recent = [t for t in before if c in t["components"]][-NOTE_TICKETS:]
            sections.append(f"Component: {c}\nRecent tickets:\n"
                            + "\n".join(f"- {mask(t['summary']).strip()}" for t in recent))
        schema = {"type": "object", "properties": {c: {"type": "string"} for c in components},
                  "required": components}
        request = {
            "model": config["model"],
            "messages": [{"role": "system", "content": prompt.strip()},
                         {"role": "user", "content": f"Area: {name}\n\n" + "\n\n".join(sections)}],
            "format": schema,
            # about 25 tokens a note plus its key; the cap only stops a runaway answer
            "options": {**config["options"], "num_predict": 48 * len(components) + 64},
        }
        if config.get("think") is not None:
            request["think"] = config["think"]
        requests.append((name, request, schema))
    return requests


def write_notes(tickets: list[dict], set_name: str, config: dict, prompt: str, mode: str, client=None,
                cassettes: Path = Path("cassettes"), verbose: bool = False) -> dict:
    cassette = Cassette(cassettes / "notes", mode)
    notes, missing = {}, []
    requests = note_requests(tickets, set_name, config, prompt)
    for n, (name, request, schema) in enumerate(requests, start=1):
        response, _, fresh = cassette.call(request, f"{set_name} notes for {name}", client)
        try:
            answer = json.loads(response["content"])
        except json.JSONDecodeError:
            answer = {}
        for c in schema["properties"]:
            value = answer.get(c) if isinstance(answer, dict) else None
            if isinstance(value, str) and value.strip():
                notes[c] = " ".join(value.split())
            else:
                missing.append(c)
        if verbose:
            took = f"{response['latency_ms'] / 1000:.0f}s" if fresh else "from recording"
            print(f"[{n}/{len(requests)}] {name}: {len(schema['properties'])} components, {took}", flush=True)
    return {"set": set_name, "written_from_tickets_before": PERIODS[set_name][0],
            "notes": notes, "missing": missing}


def load_notes(set_name: str, folder: Path = NOTES) -> dict[str, str]:
    path = folder / f"{set_name}.json"
    if not path.exists():
        raise SystemExit(f"error: no notes for {set_name}. Write them first with `tme notes {set_name}`.")
    return json.loads(path.read_text(encoding="utf-8"))["notes"]
