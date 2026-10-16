import math
import random
from collections import Counter

import numpy as np
import pytest

from tme.search import B, K1, MemorySearch, RollingBM25, tokens


def reference_bm25(corpus, query):
    """Textbook BM25 over `corpus` alone (Lucene's idf), written for clarity, not speed."""
    n = len(corpus)
    average_length = sum(map(len, corpus)) / n
    scores = []
    for doc in corpus:
        tf = Counter(doc)
        s = 0.0
        for term in query:  # a term repeated in the query counts each time
            df = sum(term in d for d in corpus)
            if df and tf[term]:
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * tf[term] * (K1 + 1) / (tf[term] + K1 * (1 - B + B * len(doc) / average_length))
        scores.append(s)
    return np.array(scores)


@pytest.fixture
def corpus():
    rng = random.Random(1)
    words = [f"w{i}" for i in range(30)]
    return [[rng.choice(words) for _ in range(rng.randint(1, 12))] for _ in range(60)]


def test_rolling_bm25_equals_bm25_built_on_earlier_documents_only(corpus):
    index = RollingBM25(corpus)
    for p in (1, 7, 30, 60):
        for query in (["w1"], ["w2", "w3", "w2"], corpus[p - 1], ["unknown", "w4"]):
            np.testing.assert_allclose(index.scores(query, p), reference_bm25(corpus[:p], query))


def test_later_documents_do_not_change_earlier_scores(corpus):
    # The leakage property: what is added after position p, however similar to the
    # query, changes nothing about the ranking of documents before p.
    query = corpus[10]
    before = RollingBM25(corpus[:20]).scores(query, 20)
    after = RollingBM25(corpus[:20] + [query] * 50 + corpus[20:]).scores(query, 20)
    np.testing.assert_array_equal(before, after)


def test_empty_memory():
    assert RollingBM25([["a"]]).scores(["a"], 0).shape == (0,)


def test_tokens_drop_the_mask_and_punctuation():
    assert tokens("Same as [ticket]: NPE in Kafka-Source!") == ["same", "as", "npe", "in", "kafka", "source"]


def test_search_only_returns_tickets_created_before_the_query(tickets, split):
    search = MemorySearch(tickets)
    by_key = {t["key"]: t for t in tickets}
    for gold in split.values():
        for g in gold:
            hits = search.search(by_key[g["key"]], 5)
            assert len(hits) == 5
            assert all(t["created"] < g["created"] for t, _ in hits), g["key"]
            scores = [s for _, s in hits]
            assert scores == sorted(scores, reverse=True)


def test_tickets_created_in_the_same_second_are_not_memory(tickets):
    search = MemorySearch(tickets)
    same = next(i for i in range(1, len(tickets)) if tickets[i]["created"] == tickets[i - 1]["created"])
    assert search.memory_size(tickets[same]) <= same - 1
    assert all(t["created"] < tickets[same]["created"] for t in tickets[:search.memory_size(tickets[same])])
