from tme.baselines import _most_common, choose_threshold
from tme.score import accuracy, correct

DUP = {"key": "FLINK-2", "component": "Table SQL / Planner", "issue_type": "Bug", "duplicate_of": ["FLINK-1"]}
NOT_DUP = {"key": "FLINK-3", "component": "Runtime / Network", "issue_type": "Improvement", "duplicate_of": []}


def test_right_area_wrong_component():
    c = correct({"component": "Table SQL / API", "issue_type": "Bug", "duplicate_of": "FLINK-1"}, DUP)
    assert c == {"component": False, "area": True, "issue_type": True, "duplicate": True}


def test_duplicate_scoring():
    assert not correct({"component": None, "issue_type": None, "duplicate_of": None}, DUP)["duplicate"]
    assert not correct({"component": None, "issue_type": None, "duplicate_of": "FLINK-9"}, DUP)["duplicate"]
    assert correct({"component": None, "issue_type": None, "duplicate_of": None}, NOT_DUP)["duplicate"]
    assert not correct({"component": None, "issue_type": None, "duplicate_of": "FLINK-1"}, NOT_DUP)["duplicate"]


def test_always_null_scores_half_on_duplicates():
    null = {"component": None, "issue_type": None, "duplicate_of": None}
    scores = accuracy({"FLINK-2": null, "FLINK-3": null}, [DUP, NOT_DUP])
    assert scores["duplicate"] == 0.5
    assert scores["duplicate_on_duplicates"] == 0.0
    assert scores["duplicate_on_non_duplicates"] == 1.0


def test_vote_ties_go_to_the_first_seen():
    assert _most_common(["b", "a", "a", "b", "c"]) == "b"
    assert _most_common([], "fallback") == "fallback"


def test_threshold_maximises_right_answers_on_dev():
    hits = [("FLINK-1", 2.0, ["FLINK-1"]),   # duplicate, top hit is its original
            ("FLINK-6", 1.9, []),            # not a duplicate
            ("FLINK-8", 1.5, ["FLINK-8"]),   # duplicate, top hit is its original
            ("FLINK-7", 1.2, [])]            # not a duplicate
    # Right answers by threshold: 1.2 -> 2, 1.5 -> 3, 1.9 -> 2, 2.0 -> 3, never -> 2.
    assert choose_threshold(hits) == 1.5     # ties go to the lowest threshold
    assert choose_threshold([("FLINK-6", 1.5, [])]) == float("inf")  # never flag
