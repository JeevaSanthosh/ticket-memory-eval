import json
from pathlib import Path

import pytest

from tme.data import load_tickets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tickets():
    return load_tickets(ROOT / "data" / "flink_issues.jsonl.gz")


@pytest.fixture(scope="session")
def split():
    return json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
