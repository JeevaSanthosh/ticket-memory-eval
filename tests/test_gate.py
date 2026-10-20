import json

import pytest

from tme.gate import gate
from tme.run import load_config, run

from conftest import ROOT
from test_run import Stub


@pytest.fixture
def setup(tmp_path, tickets, split):
    config = load_config(ROOT / "eval.toml")
    prompts = {name: (ROOT / config[path]).read_text(encoding="utf-8")
               for name, path in (("route", "route_prompt"), ("duplicates", "duplicate_prompt"))}
    small = {"dev": split["dev"][:6]}
    cassettes, results = tmp_path / "cassettes", tmp_path / "results"
    run("A", "dev", tickets, small, config, prompts, "record", client=Stub(), cassettes=cassettes,
        out=results / "dev" / "A")
    return tickets, small, config, prompts, cassettes, results


def check(setup, baseline=None, prompts=None):
    tickets, small, config, default_prompts, cassettes, results = setup
    return gate(results, baseline, tickets, small, config, prompts or default_prompts, cassettes)


def test_nothing_committed_passes(tmp_path, tickets, split):
    config = load_config(ROOT / "eval.toml")
    ok, text = gate(tmp_path, None, tickets, split, config, {}, tmp_path)
    assert ok and "nothing to check" in text


def test_committed_results_that_replay_pass(setup):
    ok, text = check(setup)
    assert ok and "Recordings replay to the committed answers" in text


def test_edited_results_fail(setup):
    path = setup[5] / "dev" / "A" / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["prediction"]["component"] = "Tests"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    ok, text = check(setup)
    assert not ok and "no longer gives" in text


def test_a_prompt_changed_without_re_recording_fails(setup):
    prompts = {**setup[3], "route": setup[3]["route"] + "\nBe brief."}
    ok, text = check(setup, prompts=prompts)
    assert not ok and "No recording" in text


def test_a_significant_drop_against_the_target_branch_blocks(setup, tmp_path):
    _, small, *_ = setup
    target = tmp_path / "target" / "dev" / "A"
    target.mkdir(parents=True)
    perfect = [{"key": g["key"], "prediction": {"component": g["component"], "issue_type": g["issue_type"],
                                                  "duplicate_of": None}} for g in small["dev"]]
    (target / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in perfect), encoding="utf-8")
    ok, text = check(setup, baseline=tmp_path / "target")
    assert not ok and "BLOCKED: significant drop in component" in text
