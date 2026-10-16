from tme.split import PER_KIND, PERIODS, build_split


def test_committed_split_is_what_the_code_and_seed_produce(tickets, split):
    assert build_split(tickets) == split


def test_split_shape_and_rules(tickets, split):
    by_key = {t["key"]: t for t in tickets}
    assert set(split) == {"dev", "test"}
    assert not {g["key"] for g in split["dev"]} & {g["key"] for g in split["test"]}
    for name, gold in split.items():
        start, end = PERIODS[name]
        assert sum(bool(g["duplicate_of"]) for g in gold) == PER_KIND
        assert sum(not g["duplicate_of"] for g in gold) == PER_KIND
        for g in gold:
            t = by_key[g["key"]]
            assert start <= t["created"] < end
            assert t["issuetype"] != "Sub-task" and t["components"] == [g["component"]]
            assert g["issue_type"] == t["issuetype"]
            assert (t["resolution"] == "Duplicate") == bool(g["duplicate_of"])
            for original in g["duplicate_of"]:
                assert by_key[original]["created"] < t["created"]
