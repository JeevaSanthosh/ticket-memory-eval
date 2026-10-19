import pytest

from tme.compare import compare


def gold(n):
    """n tickets: the first half duplicates of FLINK-1, the rest not duplicates."""
    return [{"key": f"FLINK-{100 + i}", "component": "Tests", "issue_type": "Bug",
             "duplicate_of": ["FLINK-1"] if i < n // 2 else []} for i in range(n)]


def answers(g, right_components, flag_all=False):
    """Right component on the first `right_components` tickets; type always right."""
    return {t["key"]: {"component": "Tests" if i < right_components else "Documentation",
                       "issue_type": "Bug", "duplicate_of": "FLINK-1" if flag_all else None}
            for i, t in enumerate(g)}


def by_field(results):
    return {r.field: r for r in results}


def test_identical_answers_pass_with_no_change():
    g = gold(200)
    results = by_field(compare(g, answers(g, 120), answers(g, 120)))
    assert {r.verdict for r in results.values()} == {"pass"}
    assert results["component"].delta == 0 and results["component"].ci_low == results["component"].ci_high == 0


def test_a_clear_drop_fails_and_a_clear_rise_is_better():
    g = gold(200)
    results = by_field(compare(g, answers(g, 160), answers(g, 100)))
    assert results["component"].verdict == "fail"
    assert results["component"].delta == pytest.approx(-0.3)
    assert results["issue_type"].verdict == "pass"
    assert by_field(compare(g, answers(g, 100), answers(g, 160)))["component"].verdict == "better"


def test_a_small_drop_on_few_tickets_only_warns():
    g = gold(20)
    assert by_field(compare(g, answers(g, 12), answers(g, 11)))["component"].verdict == "warn"


def test_duplicate_halves_are_scored_on_their_own_tickets():
    g = gold(100)
    flagging = by_field(compare(g, answers(g, 0), answers(g, 0, flag_all=True)))
    assert flagging["duplicate"].delta == 0                       # 50 found, 50 false flags
    assert flagging["duplicate_on_duplicates"].delta == pytest.approx(1.0)
    assert flagging["duplicate_on_non_duplicates"].delta == pytest.approx(-1.0)
    assert flagging["duplicate_on_non_duplicates"].verdict == "fail"


def test_both_sets_must_answer_the_same_tickets():
    g = gold(10)
    partial = dict(list(answers(g, 5).items())[:9])
    with pytest.raises(ValueError):
        compare(g, answers(g, 5), partial)
