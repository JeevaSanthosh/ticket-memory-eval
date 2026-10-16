"""Load the Flink tickets and prepare their text.

Every ticket key mentioned in ticket text is masked. Without that, a duplicate that says
"same as FLINK-20123" hands over the answer, and a model could match keys instead of
reading the problem. Memory tickets are still shown with their own key (a separate
field), so the model can name the one it thinks is the original.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

DATA = Path("data/flink_issues.jsonl.gz")
MASK = "[ticket]"

# A Flink key in any case ("FLINK-20123", "Flink-8566", "flink-19123-fix", "%2FFLINK-11116"),
# but not a version or file name that only looks like one ("flink-1.12.0", "flink-1-11-2",
# "orgapacheflink-1213").
KEY_MENTION = re.compile(r"(?i)(?:(?<![a-z0-9])|(?<=%2f))flink-\d+(?!\d|[.-]\d)")

# Ticket types a model can be asked to predict. Sub-tasks take their type from the parent
# ticket rather than from their own text, so they are never test tickets.
ISSUE_TYPES = ("Bug", "Improvement", "New Feature", "Technical Debt")


def load_tickets(path: Path = DATA) -> list[dict]:
    """All tickets, oldest first. `created` is UTC ISO text, so string order is time order."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        tickets = [json.loads(line) for line in f]
    tickets.sort(key=lambda t: (t["created"], int(t["key"].split("-")[1])))
    return tickets


def mask(text: str | None) -> str:
    return KEY_MENTION.sub(MASK, text or "")


def summary_text(ticket: dict) -> str:
    return mask(ticket["summary"])


def full_text(ticket: dict) -> str:
    return mask(f"{ticket['summary'] or ''}\n{ticket['description'] or ''}")


def area(component: str | None) -> str | None:
    """The part of a component name before " / ": "Table SQL / Planner" -> "Table SQL"."""
    return component.split(" / ")[0] if component else None
