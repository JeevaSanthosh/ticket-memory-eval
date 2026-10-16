import re

import pytest

from tme.data import MASK, area, full_text, mask


@pytest.mark.parametrize("text, expected", [
    ("Same as FLINK-20123.", f"Same as {MASK}."),
    ("See [Flink-8566|https://issues.apache.org/jira/browse/FLINK-8566]",
     f"See [{MASK}|https://issues.apache.org/jira/browse/{MASK}]"),
    ("introduced by flink-16537 which", f"introduced by {MASK} which"),
    ("branch flink-19123-fix-tests", f"branch {MASK}-fix-tests"),
    ("_FLINK-24227 intro", f"_{MASK} intro"),
    ("selectedIssue=FLINK%2Fbrowse%2FFLINK-11116&cc", f"selectedIssue=FLINK%2Fbrowse%2F{MASK}&cc"),
    ("(FLINK-1)", f"({MASK})"),
])
def test_mask_hides_ticket_keys(text, expected):
    assert mask(text) == expected


@pytest.mark.parametrize("text", [
    "/opt/flink-1.12.2/bin/flink run",
    "flink-dist/target/flink-1.11-SNAPSHOT-bin/flink-1.11-SNAPSHOT",
    "http://apache-flink.nabble.com/flink-1-11-2-checkpoint-td123.html",
    "orgapacheflink-1213/org/apache/flink",
    "host flink06.internal, flink0308",
])
def test_mask_leaves_versions_and_names_alone(text):
    assert mask(text) == text


def test_no_ticket_key_survives_masking(tickets):
    # Upper-case keys are how Jira writes them; none may survive in any ticket.
    # The single "FLINK-1.10"-style version string in the data is not a key.
    leftover = [t["key"] for t in tickets if re.search(r"FLINK-\d+(?![.\d])", full_text(t))]
    assert leftover == []


def test_split_tickets_never_mention_their_own_or_original_key(tickets, split):
    by_key = {t["key"]: t for t in tickets}
    for gold in split.values():
        for g in gold:
            text = full_text(by_key[g["key"]])
            for key in [g["key"], *g["duplicate_of"]]:
                number = key.split("-")[1]
                assert not re.search(rf"(?i)flink\W{{0,3}}{number}(?!\d)", text), (g["key"], key)


def test_tickets_are_oldest_first_and_in_one_time_zone(tickets):
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+0000", t["created"]) for t in tickets)
    assert [t["created"] for t in tickets] == sorted(t["created"] for t in tickets)
    assert len({t["key"] for t in tickets}) == len(tickets) == 25492


def test_area():
    assert area("Table SQL / Planner") == "Table SQL"
    assert area("Documentation") == "Documentation"
    assert area(None) is None
