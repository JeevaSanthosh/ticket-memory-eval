"""Dev (2020) and test (2021 to the end of the data) sets.

Each set holds 100 duplicates and 100 non-duplicates, drawn with a fixed seed from tickets
that are not sub-tasks and have exactly one component. Every design choice is made on dev;
test is only scored.

- Duplicate: resolved as Duplicate, with a Duplicate link to a Flink ticket created before
  it. Those earlier tickets are the right answers (almost always one).
- Non-duplicate: not resolved as Duplicate and with no Duplicate link to an earlier ticket.
- Left out as unclear: resolved as Duplicate with no earlier Flink original (the link points
  to a later ticket, another project, or nowhere), and tickets resolved some other way that
  still carry a Duplicate link to an earlier ticket.
"""

from __future__ import annotations

import random

PERIODS = {"dev": ("2020-01-01", "2021-01-01"), "test": ("2021-01-01", "9999")}
SEED = 20261013
PER_KIND = 100


def earlier_originals(ticket: dict, by_key: dict[str, dict]) -> list[str]:
    """Flink tickets created before this one that it has a Duplicate link to, in either direction."""
    return sorted({
        link["key"] for link in ticket["links"]
        if link["type"] == "Duplicate" and link["key"] in by_key
        and by_key[link["key"]]["created"] < ticket["created"]
    })


def build_split(tickets: list[dict], seed: int = SEED, per_kind: int = PER_KIND) -> dict[str, list[dict]]:
    by_key = {t["key"]: t for t in tickets}
    position = {t["key"]: i for i, t in enumerate(tickets)}
    split = {}
    for name, (start, end) in PERIODS.items():
        rng = random.Random(seed)  # one generator per set, so changing one set never moves the other
        duplicates, others = [], []
        for t in tickets:
            if not (start <= t["created"] < end) or t["issuetype"] == "Sub-task" or len(t["components"]) != 1:
                continue
            originals = earlier_originals(t, by_key)
            if t["resolution"] == "Duplicate" and originals:
                duplicates.append((t, originals))
            elif t["resolution"] != "Duplicate" and not originals:
                others.append((t, []))
        chosen = rng.sample(duplicates, per_kind) + rng.sample(others, per_kind)
        chosen.sort(key=lambda pair: position[pair[0]["key"]])
        split[name] = [
            {"key": t["key"], "created": t["created"], "component": t["components"][0],
             "issue_type": t["issuetype"], "duplicate_of": originals}
            for t, originals in chosen
        ]
    return split
