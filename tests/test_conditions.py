import re

import pytest

from tme.conditions import build_request, components_before
from tme.run import TOP_K, load_config, load_prompts
from tme.search import MemorySearch

from conftest import ROOT


@pytest.fixture(scope="module")
def setup(tickets):
    config = load_config(ROOT / "eval.toml")
    prompts = {name: (ROOT / path).read_text(encoding="utf-8")
               for name, path in (("route", config["route_prompt"]), ("duplicates", config["duplicate_prompt"]))}
    return MemorySearch(tickets), config, prompts


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


@pytest.mark.parametrize("answer", ["pick", "judge"])
def test_a_and_b_differ_only_by_the_memory(tickets, split, setup, answer):
    search, config, prompts = setup
    config = {**config, "duplicate_answer": answer}
    a = next(requests_for(tickets, split, (search, config, prompts), "A"))
    b = next(requests_for(tickets, split, (search, config, prompts), "B"))
    _, memory, b_request, b_schema = b
    _, _, a_request, a_schema = a
    extra = "duplicate_of" if answer == "pick" else "earlier_tickets"
    assert set(a_schema["properties"]) == {"component", "issue_type"}
    assert set(b_schema["properties"]) == {"component", "issue_type", extra}
    if answer == "pick":
        assert b_schema["properties"]["duplicate_of"]["enum"] == [m["key"] for m in memory] + [None]
    else:
        assert list(b_schema["properties"]["earlier_tickets"]["properties"]) == [m["key"] for m in memory]
        assert b_schema["properties"]["earlier_tickets"]["required"] == [m["key"] for m in memory]
    assert b_request["messages"][1]["content"].endswith(a_request["messages"][1]["content"])
    assert {k: v for k, v in a_schema["properties"].items()} == \
        {k: v for k, v in b_schema["properties"].items() if k != extra}


def test_the_answer_style_leaves_earlier_requests_unchanged(tickets, split, setup):
    # A's recordings, and B's under v1 and v2, must stay valid when B's answer style is switched.
    from tme.llm import request_key
    search, config, prompts = setup
    plain = {k: v for k, v in config.items() if k != "duplicate_answer"}
    for condition, styles in (("A", ("pick", "judge")), ("B", ("pick",))):
        keys = {request_key(r) for style in styles
                for _, _, r, _ in [next(requests_for(tickets, split, (search, {**plain, "duplicate_answer": style}, prompts), condition))]}
        keys.add(request_key(next(requests_for(tickets, split, (search, plain, prompts), condition))[2]))
        assert len(keys) == 1, condition


def test_only_judged_requests_get_more_room_to_answer(tickets, split, setup):
    search, config, prompts = setup
    judged = next(requests_for(tickets, split, (search, {**config, "duplicate_answer": "judge"}, prompts), "B"))[2]
    picked = next(requests_for(tickets, split, (search, {**config, "duplicate_answer": "pick"}, prompts), "B"))[2]
    assert judged["options"]["num_predict"] == 256
    assert picked["options"] == config["options"]


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
