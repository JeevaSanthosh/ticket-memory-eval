import re

import pytest

from tme.conditions import build_request, components_before
from tme.run import TOP_K, load_config, load_prompts
from tme.search import MemorySearch

from conftest import ROOT


@pytest.fixture(scope="module")
def setup(tickets):
    config = load_config(ROOT / "eval.toml")
    return MemorySearch(tickets), config, load_prompts(config, ROOT)


def requests_for(tickets, split, setup, condition):
    search, config, prompts = setup
    by_key = {t["key"]: t for t in tickets}
    for gold in split.values():
        for g in gold:
            t = by_key[g["key"]]
            memory = [m for m, _ in search.search(t, TOP_K)] if condition == "B" else []
            components = components_before(tickets, search.memory_size(t))
            request, schema = build_request(condition, t, memory, components, config, prompts)
            yield g, memory, request, schema


@pytest.mark.parametrize("condition", ["A", "B"])
def test_requests_never_show_the_new_tickets_key_or_anything_later(tickets, split, setup, condition):
    created = {t["key"]: t["created"] for t in tickets}
    for g, memory, request, _ in requests_for(tickets, split, setup, condition):
        text = "\n".join(m["content"] for m in request["messages"])
        number = g["key"].split("-")[1]
        assert not re.search(rf"(?i)flink\W{{0,3}}{number}(?!\d)", text), g["key"]
        for key in re.findall(r"FLINK-\d+", text):  # only the shown memory tickets' own keys remain
            assert created[key] < g["created"], (g["key"], key)


def test_the_right_component_is_always_on_offer(tickets, split, setup):
    for g, _, _, schema in requests_for(tickets, split, setup, "A"):
        assert g["component"] in schema["properties"]["component"]["enum"]
        assert g["issue_type"] in schema["properties"]["issue_type"]["enum"]


def test_a_and_b_differ_only_by_the_memory(tickets, split, setup):
    a = next(requests_for(tickets, split, setup, "A"))
    b = next(requests_for(tickets, split, setup, "B"))
    _, memory, b_request, b_schema = b
    _, _, a_request, a_schema = a
    assert "duplicate_of" not in a_schema["properties"]
    assert b_schema["properties"]["duplicate_of"]["enum"] == [m["key"] for m in memory] + [None]
    assert b_request["messages"][1]["content"].endswith(a_request["messages"][1]["content"])
    assert {k: v for k, v in a_schema["properties"].items()} == \
        {k: v for k, v in b_schema["properties"].items() if k != "duplicate_of"}


def test_components_on_offer_are_those_used_before(tickets):
    assert components_before(tickets, 0) == []
    first_three = {c for t in tickets[:3] for c in t["components"]}
    assert components_before(tickets, 3) == sorted(first_three)


def test_prompt_line_endings_do_not_change_the_request(tmp_path, tickets, setup):
    # A prompt checked out on Windows has CRLF line endings; its recordings must replay in CI.
    from tme.llm import request_key
    _, config, _ = setup
    for name in ("route_prompt", "duplicate_prompt"):
        text = (ROOT / config[name]).read_bytes()
        (tmp_path / name).write_bytes(text.replace(b"\n", b"\r\n"))
    crlf = load_prompts({**config, "route_prompt": tmp_path / "route_prompt",
                         "duplicate_prompt": tmp_path / "duplicate_prompt"})
    lf = load_prompts({**config, "route_prompt": ROOT / config["route_prompt"],
                       "duplicate_prompt": ROOT / config["duplicate_prompt"]})
    t = tickets[-1]
    keys = {request_key(build_request("A", t, [], ["X"], config, p)[0]) for p in (crlf, lf)}
    assert len(keys) == 1
