import json

import pytest

from tme.llm import CassetteMiss
from tme.run import load_config, parse, run

from conftest import ROOT


class Stub:
    """Stands in for Ollama: answers with the first allowed value of every field."""

    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        answer = {field: spec["enum"][0] for field, spec in request["format"]["properties"].items()}
        return {"content": json.dumps(answer), "latency_ms": 1000, "load_ms": 0, "prompt_ms": 800,
                "generate_ms": 200, "prompt_tokens": 900, "completion_tokens": 25}


@pytest.fixture
def inputs():
    config = load_config(ROOT / "eval.toml")
    prompts = {name: (ROOT / config[path]).read_text(encoding="utf-8")
               for name, path in (("route", "route_prompt"), ("duplicates", "duplicate_prompt"))}
    return config, prompts


@pytest.mark.parametrize("condition", ["A", "B"])
def test_record_then_replay_gives_the_same_answers_without_a_model(tmp_path, tickets, split, inputs, condition):
    config, prompts = inputs
    stub = Stub()
    common = dict(cassettes=tmp_path / "cassettes", limit=4)
    recorded = run(condition, "dev", tickets, split, config, prompts, "record", client=stub,
                   out=tmp_path / "rec", **common)
    assert stub.calls == 4
    replayed = run(condition, "dev", tickets, split, config, prompts, "replay", out=tmp_path / "rep", **common)
    assert replayed["scores"] == recorded["scores"]
    assert (tmp_path / "rec" / "predictions.jsonl").read_text(encoding="utf-8") == \
        (tmp_path / "rep" / "predictions.jsonl").read_text(encoding="utf-8")
    assert recorded["operations"]["invalid_responses"] == 0
    assert recorded["operations"]["mean_prompt_tokens"] == 900


def test_replay_refuses_to_invent_an_answer(tmp_path, tickets, split, inputs):
    config, prompts = inputs
    with pytest.raises(CassetteMiss):
        run("A", "dev", tickets, split, config, prompts, "replay", cassettes=tmp_path, out=tmp_path / "o", limit=1)


def test_a_changed_prompt_is_a_new_request(tmp_path, tickets, split, inputs):
    config, prompts = inputs
    stub = Stub()
    run("A", "dev", tickets, split, config, prompts, "record", client=stub, cassettes=tmp_path, out=tmp_path / "o", limit=1)
    changed = {**prompts, "route": prompts["route"] + "\nBe brief."}
    with pytest.raises(CassetteMiss):
        run("A", "dev", tickets, split, config, changed, "replay", cassettes=tmp_path, out=tmp_path / "o", limit=1)


SCHEMA = {"properties": {"component": {"enum": ["Tests", "Documentation"]},
                         "issue_type": {"enum": ["Bug", "Improvement"]},
                         "duplicate_of": {"enum": ["FLINK-1", None]}}}


def test_parse():
    good = parse('{"component": "Tests", "issue_type": "Bug", "duplicate_of": null}', SCHEMA)
    assert good == ({"component": "Tests", "issue_type": "Bug", "duplicate_of": None}, None)
    prediction, error = parse('{"component": "Runtime", "issue_type": "Bug", "duplicate_of": "FLINK-1"}', SCHEMA)
    assert prediction == {"component": None, "issue_type": "Bug", "duplicate_of": "FLINK-1"}
    assert error == "invalid component"
    assert parse('{"component": "Tests", "issue_type": "Bu', SCHEMA)[1] == "not JSON"
    assert parse('["Tests"]', SCHEMA)[1] == "not a JSON object"
