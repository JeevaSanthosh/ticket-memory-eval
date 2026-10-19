"""The Ollama client and the cassette that records and replays its responses.

Recording happens on a machine running Ollama. Every request is hashed (model, messages,
schema, options) and the response saved under that hash. Replay looks the hash up and never
touches a model, which is how CI scores a change for free.

Change the prompt, the model, the options or the ticket text and the hash changes, so a
stale recording can never pass for a fresh result: replay stops with a CassetteMiss.
(Taken from statement-eval-gate, with Ollama's timing breakdown added.)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

# num_predict caps the answer: a valid one is about 30 tokens, so a runaway answer stops
# early and is counted as invalid instead of running for minutes on a CPU.
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 128}


class CassetteMiss(RuntimeError):
    pass


def request_key(request: dict) -> str:
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: float = 900):
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout

    def complete(self, request: dict) -> dict:
        started = time.perf_counter()
        try:
            r = httpx.post(f"{self.host}/api/chat", json={**request, "stream": False}, timeout=self.timeout)
        except httpx.ConnectError:
            raise SystemExit(f"error: cannot reach Ollama at {self.host}. Start it with `ollama serve`.") from None
        if r.status_code == 404:
            raise SystemExit(f"error: Ollama has no model '{request['model']}'. Run `ollama pull {request['model']}`.")
        if r.status_code >= 400:
            raise SystemExit(f"error: Ollama answered {r.status_code}: {r.text[:500]}")
        body = r.json()
        ms = lambda ns: round(ns / 1e6) if ns is not None else None  # noqa: E731
        return {
            "content": body["message"]["content"],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "load_ms": ms(body.get("load_duration")),
            "prompt_ms": ms(body.get("prompt_eval_duration")),
            "generate_ms": ms(body.get("eval_duration")),
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
        }


class Cassette:
    def __init__(self, root: Path, mode: str):
        if mode not in {"record", "replay"}:
            raise ValueError("mode must be 'record' or 'replay'")
        self.root, self.mode = root, mode
        self.hits = self.misses = 0

    def call(self, request: dict, ticket_key: str, client) -> tuple[dict, str, bool]:
        """Return (response, request hash, whether the model was called just now)."""
        key = request_key(request)
        path = self.root / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"], key, False
        if self.mode == "replay":
            raise CassetteMiss(
                f"No recording for {ticket_key} (request {key[:12]}). The prompt, model, options or "
                f"ticket text changed since the last recording. Record it against Ollama and commit "
                f"the new files in {self.root}."
            )
        if client is None:
            raise CassetteMiss(f"No recording for {ticket_key} and no model to ask.")
        self.misses += 1
        response = client.complete(request)
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"ticket": ticket_key, "model": request["model"], "response": response}
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return response, key, True
