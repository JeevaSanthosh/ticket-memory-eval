import json

import pytest

from tme.conditions import build_request, components_before, mislabel_first
from tme.gate import gate
from tme.notes import NOTE_TICKETS, note_requests, write_notes
from tme.run import TOP_K, load_config, load_prompts, run
from tme.search import MemorySearch
from tme.split import PERIODS

from conftest import ROOT
from test_run import Stub


@pytest.fixture(scope="module")
def setup(tickets):
    config = load_config(ROOT / "eval.toml")
    return MemorySearch(tickets), config, load_prompts(config, ROOT)


@pytest.mark.parametrize("set_name", ["dev", "test"])
def test_notes_are_written_only_from_tickets_before_the_set(tickets, setup, set_name):
    _, config, prompts = setup
    start = PERIODS[set_name][0]
    earlier = [t for t in tickets if t["created"] < start]
    assert note_requests(tickets, set_name, config, prompts["write_notes"]) == \
        note_requests(earlier, set_name, config, prompts["write_notes"])


def test_every_component_used_before_the_set_gets_a_note_request(tickets, setup):
    _, config, prompts = setup
    requests = note_requests(tickets, "test", config, prompts["write_notes"])
    asked = [c for _, _, schema in requests for c in schema["properties"]]
    assert sorted(asked) == components_before(tickets, sum(t["created"] < "2021-01-01" for t in tickets))
    for _, request, schema in requests:
        text = request["messages"][1]["content"]
        assert text.count("Component: ") == len(schema["properties"])
        assert text.count("\n- ") <= NOTE_TICKETS * len(schema["properties"])


def first_request(tickets, split, setup, condition, notes=None):
    search, config, prompts = setup
    by_key = {t["key"]: t for t in tickets}
    g = split["test"][0]
    t = by_key[g["key"]]
    memory = [m for m, _ in search.search(t, TOP_K)]
    components = components_before(tickets, search.memory_size(t))
    if condition == "D1":
        memory, info = mislabel_first(memory, components, g["component"], g["key"])
    elif condition not in ("B",):
        memory, info = [], None
    else:
        info = None
    return g, memory, components, info, *build_request(condition, t, memory, components, config, prompts, notes)


def test_c_differs_from_a_only_by_the_notes(tickets, split, setup):
    _, config, prompts = setup
    *_, a, a_schema = first_request(tickets, split, setup, "A")
    *_, empty, empty_schema = first_request(tickets, split, setup, "C", notes={})
    assert a_schema == empty_schema and a["messages"][1] == empty["messages"][1]
    assert empty["messages"][0]["content"].replace(prompts["notes"].strip() + "\n\n", "") == a["messages"][0]["content"]
    _, _, components, _, c, _ = first_request(tickets, split, setup, "C", notes={"Tests": "about tests"})
    assert ("Tests: about tests" in c["messages"][0]["content"]) == ("Tests" in components)


def test_d1_differs_from_b_by_one_wrong_label(tickets, split, setup):
    g, _, components, _, b, b_schema = first_request(tickets, split, setup, "B")
    _, memory, _, info, d1, d1_schema = first_request(tickets, split, setup, "D1")
    assert b_schema == d1_schema and b["messages"][0] == d1["messages"][0]
    b_lines, d1_lines = b["messages"][1]["content"].splitlines(), d1["messages"][1]["content"].splitlines()
    changed = [(x, y) for x, y in zip(b_lines, d1_lines) if x != y]
    assert len(b_lines) == len(d1_lines) and len(changed) == 1
    assert changed[0][0].startswith(info["key"]) and f"component: {info['shown_as']};" in changed[0][1]
    assert info["shown_as"] in components
    assert info["shown_as"] not in info["really"] and info["shown_as"] != g["component"]
    assert mislabel_first(memory, components, g["component"], g["key"])[1]["shown_as"] != g["component"]


def test_the_wrong_label_is_the_same_on_every_run(tickets, split, setup):
    one = first_request(tickets, split, setup, "D1")[3]
    two = first_request(tickets, split, setup, "D1")[3]
    assert one == two


def test_edited_notes_fail_the_gate(tmp_path, tickets, split, setup):
    _, config, prompts = setup
    cassettes, results = tmp_path / "cassettes", tmp_path / "results"
    notes = write_notes(tickets, "dev", config, prompts["write_notes"], "record", client=Stub(), cassettes=cassettes)
    (results / "notes").mkdir(parents=True)
    path = results / "notes" / "dev.json"
    path.write_text(json.dumps(notes), encoding="utf-8")
    small = {"dev": split["dev"][:3]}
    run("C", "dev", tickets, small, config, prompts, "record", client=Stub(), cassettes=cassettes,
        out=results / "dev" / "C", notes_folder=results / "notes")
    ok, text = gate(results, None, tickets, small, config, prompts, cassettes)
    assert ok and "Notes for dev replay to the committed notes" in text
    notes["notes"]["Tests"] = "edited by hand"
    path.write_text(json.dumps(notes), encoding="utf-8")
    ok, text = gate(results, None, tickets, small, config, prompts, cassettes)
    assert not ok and "no longer gives" in text


def test_d2_memory_is_at_least_two_years_older_than_each_ticket(tickets, split, setup):
    from datetime import datetime
    from tme.conditions import TIME_FORMAT, stale_cutoff
    search, _, _ = setup
    by_key = {t["key"]: t for t in tickets}
    for g in split["test"]:
        t = by_key[g["key"]]
        cutoff = stale_cutoff(t["created"])
        hits = search.search(t, TOP_K, before=cutoff)
        assert len(hits) == 5
        for m, _ in hits:
            age = datetime.strptime(t["created"], TIME_FORMAT) - datetime.strptime(m["created"], TIME_FORMAT)
            assert m["created"] < cutoff and age.days >= 730, (g["key"], m["key"])


def test_d2_differs_from_b_only_by_which_tickets_are_shown(tickets, split, setup):
    from tme.conditions import stale_cutoff
    search, config, prompts = setup
    by_key = {t["key"]: t for t in tickets}
    t = by_key[split["test"][0]["key"]]
    components = components_before(tickets, search.memory_size(t))
    fresh = [m for m, _ in search.search(t, TOP_K)]
    stale = [m for m, _ in search.search(t, TOP_K, before=stale_cutoff(t["created"]))]
    b, b_schema = build_request("B", t, fresh, components, config, prompts)
    d2, d2_schema = build_request("D2", t, stale, components, config, prompts)
    assert b["messages"][0] == d2["messages"][0] and b["options"] == d2["options"]
    assert d2_schema["properties"]["duplicate_of"]["enum"] == [m["key"] for m in stale] + [None]
    new_ticket = b["messages"][1]["content"].split("\n\nNew ticket\n")[1]
    assert d2["messages"][1]["content"].endswith("\n\nNew ticket\n" + new_ticket)
    assert [m["key"] for m in stale] != [m["key"] for m in fresh]  # B may also show old tickets


def test_a_cutoff_cannot_reach_past_the_ticket_itself(tickets, setup):
    search, _, _ = setup
    t = tickets[1000]
    assert search.search(t, TOP_K, before="9999") == search.search(t, TOP_K)
