"""Per-field scoring of predictions against the split's right answers.

A prediction is {"component", "issue_type", "duplicate_of"}. Fields scored:
- component: exact match.
- area: the part of the component before " / " matches ("Table SQL / Planner" vs
  "Table SQL / API" is a wrong component but the right area).
- issue_type: exact match.
- duplicate: for a duplicate, the predicted key is one of its earlier originals; for a
  non-duplicate, the prediction is null. Also reported separately for the two halves,
  because always answering null already scores 0.5.
"""

from __future__ import annotations

from tme.data import area


def correct(prediction: dict, gold: dict) -> dict[str, bool]:
    if gold["duplicate_of"]:
        duplicate = prediction["duplicate_of"] in gold["duplicate_of"]
    else:
        duplicate = prediction["duplicate_of"] is None
    return {
        "component": prediction["component"] == gold["component"],
        "area": area(prediction["component"]) == area(gold["component"]),
        "issue_type": prediction["issue_type"] == gold["issue_type"],
        "duplicate": duplicate,
    }


def accuracy(predictions: dict[str, dict], gold: list[dict]) -> dict[str, float]:
    """`predictions` maps each ticket key in `gold` to its prediction."""
    rows = [(correct(predictions[g["key"]], g), bool(g["duplicate_of"])) for g in gold]

    def mean(values: list[bool]) -> float:
        return round(sum(values) / len(values), 3)

    scores = {field: mean([c[field] for c, _ in rows]) for field in ("component", "area", "issue_type", "duplicate")}
    scores["duplicate_on_duplicates"] = mean([c["duplicate"] for c, is_dup in rows if is_dup])
    scores["duplicate_on_non_duplicates"] = mean([c["duplicate"] for c, is_dup in rows if not is_dup])
    return scores
