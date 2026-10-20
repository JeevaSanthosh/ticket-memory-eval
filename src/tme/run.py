"""Run one condition over the dev or test tickets, through the cassette, and score it."""

from __future__ import annotations

import hashlib
import json
import statistics
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from tme.conditions import build_request, components_before
from tme.llm import OPTIONS, Cassette
from tme.score import accuracy
from tme.search import MemorySearch

TOP_K = 5


def load_config(path: Path = Path("eval.toml")) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    return {**config, "options": OPTIONS}


def load_prompts(config: dict) -> dict[str, str]:
    # read_text turns Windows line endings into "\n", so a prompt checked out on Windows
    # gives the same request hash as on Linux and recordings replay in CI.
    return {"route": Path(config["route_prompt"]).read_text(encoding="utf-8"),
            "duplicates": Path(config["duplicate_prompt"]).read_text(encoding="utf-8")}


def parse(content: str, schema: dict) -> tuple[dict, str | None]:
    """Read the answer. A field missing or outside its allowed values counts as no answer."""
    prediction = {"component": None, "issue_type": None, "duplicate_of": None}
    try:
        answer = json.loads(content)
    except json.JSONDecodeError:
        return prediction, "not JSON"
    if not isinstance(answer, dict):
        return prediction, "not a JSON object"
    invalid = []
    for field, spec in schema["properties"].items():
        value = answer.get(field)
        if "properties" in spec:  # one label per shown ticket (judged duplicates)
            labels = spec["properties"]
            if isinstance(value, dict) and all(value.get(k) in s["enum"] for k, s in labels.items()):
                prediction[field] = {k: value[k] for k in labels}
                prediction["duplicate_of"] = next((k for k in labels if value[k] == "same"), None)
            else:
                invalid.append(field)
        elif field in answer and value in spec["enum"]:
            prediction[field] = value
        else:
            invalid.append(field)
    return prediction, (f"invalid {', '.join(invalid)}" if invalid else None)


def _operations(rows: list[dict]) -> dict:
    def values(field):
        return [r[field] for r in rows if r.get(field) is not None]

    def mean(field):
        v = values(field)
        return round(statistics.mean(v)) if v else None

    latency = sorted(values("latency_ms"))
    return {
        "tickets": len(rows),
        "invalid_responses": sum(r["error"] is not None for r in rows),
        "latency_ms_p50": round(statistics.median(latency)) if latency else None,
        "latency_ms_p95": latency[min(len(latency) - 1, round(0.95 * (len(latency) - 1)))] if latency else None,
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_completion_tokens": mean("completion_tokens"),
        "mean_prompt_ms": mean("prompt_ms"),
        "mean_generate_ms": mean("generate_ms"),
    }


def run(condition: str, set_name: str, tickets: list[dict], split: dict, config: dict, prompts: dict,
        mode: str, client=None, cassettes: Path = Path("cassettes"), out: Path | None = None,
        limit: int | None = None, verbose: bool = False) -> dict:
    gold = split[set_name][:limit]
    search = MemorySearch(tickets)
    by_key = {t["key"]: t for t in tickets}
    cassette = Cassette(cassettes / condition, mode)

    rows = []
    for n, g in enumerate(gold, start=1):
        ticket = by_key[g["key"]]
        memory = [t for t, _ in search.search(ticket, TOP_K)] if condition == "B" else []
        components = components_before(tickets, search.memory_size(ticket))
        request, schema = build_request(condition, ticket, memory, components, config, prompts)
        response, key, fresh = cassette.call(request, g["key"], client)
        prediction, error = parse(response["content"], schema)
        rows.append({"key": g["key"], "prediction": prediction, "error": error,
                     "shown": [t["key"] for t in memory], "cassette_key": key,
                     **{k: response.get(k) for k in ("latency_ms", "load_ms", "prompt_ms", "generate_ms",
                                                      "prompt_tokens", "completion_tokens")}})
        if verbose:
            r = rows[-1]
            timing = (f"{r['latency_ms'] / 1000:.1f}s (prompt {r['prompt_tokens']} tokens in "
                      f"{(r['prompt_ms'] or 0) / 1000:.1f}s, answer {r['completion_tokens']} tokens in "
                      f"{(r['generate_ms'] or 0) / 1000:.1f}s)") if fresh else "from recording"
            print(f"[{n}/{len(gold)}] {g['key']} {timing} {error or 'ok'}", flush=True)

    results = {
        "condition": condition,
        "set": set_name,
        "scores": accuracy({r["key"]: r["prediction"] for r in rows}, gold),
        "operations": _operations(rows),
    }
    out = out or Path("runs") / set_name / condition
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    run_info = {
        "condition": condition, "set": set_name, "tickets": len(gold), "model": config["model"],
        "think": config.get("think"), "options": config["options"],
        "prompts": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in prompts.items()},
        "mode": mode, "recorded_now": cassette.misses,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=1) + "\n", encoding="utf-8")
    (out / "scores.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return results
