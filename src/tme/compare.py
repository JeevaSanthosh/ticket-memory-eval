"""Compare two sets of answers to the same tickets, field by field.

The comparison is paired: both sets answered the same tickets, so each bootstrap resample
draws the same tickets for both and measures the change on identical inputs. That removes the
noise of which tickets happened to be drawn, and gives an interval for the change.
(The method and verdicts are statement-eval-gate's, applied to accuracy instead of F1.)

Verdict per field:
  fail    accuracy fell by more than max_drop and the interval lies wholly below zero
  warn    it fell by more than max_drop but the interval still includes zero, so this
          sample cannot tell a real drop from noise
  better  it rose by more than max_drop and the interval lies wholly above zero
  pass    anything else
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tme.score import correct

FIELDS = ("component", "area", "issue_type", "duplicate", "duplicate_on_duplicates", "duplicate_on_non_duplicates")


@dataclass
class FieldResult:
    field: str
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: str


def load_predictions(path: Path) -> dict[str, dict]:
    """Predictions by ticket key, from a predictions.jsonl file or a folder holding one."""
    path = path / "predictions.jsonl" if path.is_dir() else path
    with path.open(encoding="utf-8") as f:
        return {row["key"]: row["prediction"] for row in map(json.loads, f)}


def _correctness(predictions: dict[str, dict], gold: list[dict]) -> dict[str, np.ndarray]:
    rows = [correct(predictions[g["key"]], g) for g in gold]
    return {field: np.array([r[field] for r in rows], dtype=float) for field in ("component", "area", "issue_type", "duplicate")}


def compare(gold: list[dict], baseline: dict[str, dict], candidate: dict[str, dict], max_drop: float = 0.02,
            confidence: float = 0.95, resamples: int = 10_000, seed: int = 0) -> list[FieldResult]:
    keys = {g["key"] for g in gold}
    if set(baseline) != keys or set(candidate) != keys:
        raise ValueError("Both sets must answer exactly the tickets being compared, so the comparison can be paired.")
    b, c = _correctness(baseline, gold), _correctness(candidate, gold)
    is_dup = np.array([bool(g["duplicate_of"]) for g in gold], dtype=float)
    subsets = {"duplicate_on_duplicates": is_dup, "duplicate_on_non_duplicates": 1 - is_dup}

    idx = np.random.default_rng(seed).integers(0, len(gold), size=(resamples, len(gold)))
    tail = (1 - confidence) / 2 * 100
    results = []
    for field in FIELDS:
        source = "duplicate" if field in subsets else field
        weight = subsets.get(field, np.ones(len(gold)))
        if not weight.any():
            continue

        def accuracy(values, rows=slice(None)):
            w = weight[rows]
            return (values[rows] * w).sum(axis=-1) / np.maximum(w.sum(axis=-1), 1)

        base, cand = float(accuracy(b[source])), float(accuracy(c[source]))
        deltas = accuracy(c[source], idx) - accuracy(b[source], idx)
        low, high = (float(x) for x in np.percentile(deltas, [tail, 100 - tail]))
        delta = cand - base
        if delta < -max_drop and high < 0:
            verdict = "fail"
        elif delta < -max_drop:
            verdict = "warn"
        elif delta > max_drop and low > 0:
            verdict = "better"
        else:
            verdict = "pass"
        results.append(FieldResult(field, base, cand, delta, low, high, verdict))
    return results


ICON = {"fail": "❌ fail", "warn": "⚠️ warn", "better": "✅ better", "pass": "pass"}


def report(results: list[FieldResult], baseline_name: str, candidate_name: str, tickets: int,
           confidence: float = 0.95) -> str:
    lines = [
        f"{candidate_name} against {baseline_name}, {tickets} tickets, paired bootstrap",
        "",
        f"| Field | {baseline_name} | {candidate_name} | Change | {int(confidence * 100)}% interval | Verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.field} | {r.baseline:.3f} | {r.candidate:.3f} | {r.delta:+.3f} | "
                     f"{r.ci_low:+.3f} to {r.ci_high:+.3f} | {ICON[r.verdict]} |")
    return "\n".join(lines) + "\n"
