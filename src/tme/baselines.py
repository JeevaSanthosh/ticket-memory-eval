"""Answers without a model, for the model runs to beat.

- most_common: the component and issue type seen most often in the ticket's memory
  (everything created before it); never calls anything a duplicate.
- vote_of_5: the 5 tickets the memory search returns vote on component and issue type
  (ties go to the better-ranked ticket). The top ticket is called the original when its
  search score clears a threshold chosen on dev. Condition B shows a model these same 5
  tickets, so B has to beat this to show the model adds anything beyond search.

Also reported, from the search alone: how often the right component is among the 5
tickets, how often the original is among them (what B could find at most), and how many
tickets have a component that no earlier ticket had.
"""

from __future__ import annotations

from collections import Counter

from tme.data import ISSUE_TYPES
from tme.score import accuracy
from tme.search import MemorySearch

TOP_K = 5


def _most_common(values, fallback=None):
    """Most frequent value; ties go to the one seen first."""
    counts = Counter(values)
    return counts.most_common(1)[0][0] if counts else fallback


def choose_threshold(top_hits: list[tuple[str | None, float, list[str]]]) -> float:
    """The search score above which calling the top ticket the original is most often right.

    `top_hits` holds (top ticket's key, its score, the right originals) per dev ticket.
    Ties between thresholds go to the lowest.
    """
    def right(threshold: float) -> int:
        return sum((key in originals) if score >= threshold else not originals
                   for key, score, originals in top_hits)

    candidates = sorted({score for _, score, _ in top_hits}) + [float("inf")]
    return max(candidates, key=lambda t: (right(t), -t))


def run(tickets: list[dict], split: dict[str, list[dict]]) -> dict:
    search = MemorySearch(tickets)
    by_key = {t["key"]: t for t in tickets}
    memory = {g["key"]: search.search(by_key[g["key"]], TOP_K) for gold in split.values() for g in gold}

    def top_hit(g):
        hits = memory[g["key"]]
        return (hits[0][0]["key"], hits[0][1]) if hits else (None, 0.0)

    dev_hits = [(*top_hit(g), g["duplicate_of"]) for g in split["dev"]]
    threshold = choose_threshold(dev_hits)

    results = {"duplicate_threshold": round(threshold, 4), "chosen_on": "dev"}
    for name, gold in split.items():
        most_common, vote = {}, {}
        found = {"component_in_top_5": [], "original_in_top_5": [], "original_first": []}
        unseen = 0
        for g in gold:
            earlier = tickets[:search.memory_size(by_key[g["key"]])]
            common_component = _most_common(c for t in earlier for c in t["components"])
            common_type = _most_common(t["issuetype"] for t in earlier if t["issuetype"] in ISSUE_TYPES)
            most_common[g["key"]] = {"component": common_component, "issue_type": common_type, "duplicate_of": None}

            shown = [t for t, _ in memory[g["key"]]]
            key, score = top_hit(g)
            vote[g["key"]] = {
                "component": _most_common((c for t in shown for c in t["components"]), common_component),
                "issue_type": _most_common((t["issuetype"] for t in shown if t["issuetype"] in ISSUE_TYPES), common_type),
                "duplicate_of": key if score >= threshold else None,
            }

            found["component_in_top_5"].append(any(g["component"] in t["components"] for t in shown))
            if g["duplicate_of"]:
                found["original_in_top_5"].append(any(t["key"] in g["duplicate_of"] for t in shown))
                found["original_first"].append(bool(shown) and shown[0]["key"] in g["duplicate_of"])
            unseen += not any(g["component"] in t["components"] for t in earlier)

        results[name] = {
            "tickets": len(gold),
            "most_common": accuracy(most_common, gold),
            "vote_of_5": accuracy(vote, gold),
            "search": {k: round(sum(v) / len(v), 3) for k, v in found.items()}
            | {"component_never_seen_before": unseen},
        }
    return results
