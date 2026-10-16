"""Memory search: the earlier tickets most similar to a new one.

A ticket's memory is every Flink ticket created before it. The BM25 statistics (number of
documents, document frequencies, average length) are computed from those earlier tickets
only, so nothing about later tickets, including the query ticket itself, leaks into the
ranking. One index serves every query: tickets are stored oldest first, so "created before
the query" is a prefix of the index, and each term's postings are cut at that prefix.

The memory search sums two BM25 rankings, each divided by its best score: summary against
summaries, and summary plus description against the same. On the 100 dev duplicates this
puts the original in the top 5 for 51, against 46 for full text alone.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import Counter

import numpy as np

from tme.data import MASK, full_text, summary_text

K1 = 1.5
B = 0.75


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.replace(MASK, " ").lower())


class RollingBM25:
    """BM25 where a query at position p sees only documents 0..p-1, as if they were the whole corpus."""

    def __init__(self, docs: list[list[str]]):
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                rows, tfs = postings.setdefault(term, ([], []))
                rows.append(i)
                tfs.append(tf)
        # Documents are added in order, so each term's rows are already sorted.
        self.postings = {t: (np.array(r), np.array(f, dtype=float)) for t, (r, f) in postings.items()}
        self.length = np.array([len(d) for d in docs], dtype=float)
        self.cumulative_length = np.concatenate([[0.0], np.cumsum(self.length)])

    def scores(self, query: list[str], p: int) -> np.ndarray:
        """Scores for documents 0..p-1. Each query term counts as often as it occurs in the query."""
        scores = np.zeros(p)
        if p == 0:
            return scores
        average_length = self.cumulative_length[p] / p
        for term, query_tf in Counter(query).items():
            if term not in self.postings:
                continue
            rows, tfs = self.postings[term]
            df = int(np.searchsorted(rows, p))  # postings in documents before p
            if df == 0:
                continue
            rows, tfs = rows[:df], tfs[:df]
            idf = np.log(1 + (p - df + 0.5) / (df + 0.5))
            norm = K1 * (1 - B + B * self.length[rows] / average_length)
            scores[rows] += query_tf * idf * tfs * (K1 + 1) / (tfs + norm)
        return scores


class MemorySearch:
    def __init__(self, tickets: list[dict]):
        """`tickets` must be oldest first, as `load_tickets` returns them."""
        self.tickets = tickets
        self.created = [t["created"] for t in tickets]
        assert self.created == sorted(self.created), "tickets must be sorted by creation time"
        self.summary = RollingBM25([tokens(summary_text(t)) for t in tickets])
        self.full = RollingBM25([tokens(full_text(t)) for t in tickets])

    def memory_size(self, ticket: dict) -> int:
        """How many tickets were created strictly before this one."""
        return bisect_left(self.created, ticket["created"])

    def search(self, ticket: dict, k: int = 5) -> list[tuple[dict, float]]:
        """The k earlier tickets that score highest, best first, with their combined scores."""
        p = self.memory_size(ticket)
        a = self.summary.scores(tokens(summary_text(ticket)), p)
        b = self.full.scores(tokens(full_text(ticket)), p)
        combined = a / (a.max(initial=0) or 1) + b / (b.max(initial=0) or 1)
        best = np.argsort(-combined, kind="stable")[:k]  # ties: the older ticket first
        return [(self.tickets[i], float(combined[i])) for i in best if combined[i] > 0]
