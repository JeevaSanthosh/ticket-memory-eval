# ticket-memory-eval

Does memory of past tickets make a small local model better at handling new ones, which kind
of memory is worth its prompt tokens, and when does memory make answers worse?

A ticket assistant reads a new Jira ticket from Apache Flink and answers two questions:

1. **Routing:** which component does it belong to, and what type is it (Bug, Improvement,
   New Feature, Technical Debt)?
2. **Duplicates:** does it repeat an earlier ticket, and if so, which one?

The model runs locally through [Ollama](https://ollama.com). CI never calls a model.

**Status:** data, dev and test sets, memory search and the baselines that need no model. The
model runs come next.

## Data

All 25,492 Apache Flink tickets from [The Public Jira Dataset](https://doi.org/10.5281/zenodo.5882881)
(CC BY 4.0), June 2014 to January 2022. Flink has 69 components, grouped into 27 areas by the
part before " / " (`Table SQL / Planner` is in area `Table SQL`), and more duplicates of earlier
tickets than any other Apache project in the dataset. [data/README.md](data/README.md) covers the
source, the fields and how the subset was extracted.

## Dev and test sets

| Set | Tickets created | Duplicates | Non-duplicates |
|---|---|---:|---:|
| dev | 2020 | 100 | 100 |
| test | 2021 to 5 Jan 2022 | 100 | 100 |

Every design choice (search settings, prompts, the duplicate threshold) is made on dev. Test
is only scored. Both sets are drawn with a fixed seed from tickets that are not sub-tasks (a
sub-task's type comes from its parent, not from its text) and have exactly one component.

- **Duplicate:** resolved as Duplicate, with a Duplicate link to an earlier Flink ticket. That
  earlier ticket is the right answer.
- **Non-duplicate:** not resolved as Duplicate and not linked as a duplicate of anything earlier.
- Tickets that are neither (a Duplicate resolution pointing to a later ticket, another project
  or nowhere; a Duplicate link on a ticket resolved some other way) are left out.

**Memory is rolling:** a ticket's memory is every Flink ticket created before it, not a fixed
cutoff. Duplicates are usually filed within weeks of the original (median 21 days in 2021), so a
fixed cutoff would leave too few whose original is in memory. Ticket keys mentioned in ticket
text (`FLINK-20123`, `Flink-8566`, ...) are masked as `[ticket]`, so a duplicate saying "same as
FLINK-20123" cannot give its answer away. Memory tickets are still shown with their own key, so
the model can name one.

## Memory search

[src/tme/search.py](src/tme/search.py) finds the 5 earlier tickets most similar to a new one by
adding two BM25 rankings, each divided by its best score: summary against summaries, and summary
plus description against the same. On the 100 dev duplicates this puts the original among the 5
for 51, against 46 for full text alone and 34 for summaries alone.

BM25's statistics (document count, document frequencies, average length) come only from the
tickets created before the query, so later tickets, including the query itself, cannot change a
ranking. One index serves every query: tickets are stored oldest first, so each query's memory is
a prefix of the index. Tests check the rankings against textbook BM25 built from the earlier
tickets alone, and that no search ever returns a ticket created at or after the query.

## Baselines without a model

`tme baselines` scores two ways of answering without a model. CI rebuilds them from the data and
fails if they differ from [results/baselines.json](results/baselines.json).

- **Most common:** the component and type seen most often in the ticket's memory
  (`Runtime / Coordination`, Bug); never a duplicate.
- **Vote of 5:** the 5 tickets the memory search returns vote on component and type. The top
  ticket is called the original when its search score clears a threshold chosen on dev (1.97 of
  a possible 2: in practice, both rankings put the same ticket first).

| | dev | test |
|---|---:|---:|
| **Component** — most common | 0.030 | 0.065 |
| Component — vote of 5 | 0.520 | 0.455 |
| Component among the 5 shown (the most a model can pick from them) | 0.745 | 0.730 |
| **Area** — most common | 0.220 | 0.280 |
| Area — vote of 5 | 0.635 | 0.595 |
| **Issue type** — most common | 0.580 | 0.665 |
| Issue type — vote of 5 | 0.710 | 0.710 |
| **Duplicate** (right original, or none) — most common | 0.500 | 0.500 |
| Duplicate — vote of 5 | 0.570 | 0.520 |
| … on duplicates: original found | 0.200 | 0.170 |
| … on non-duplicates: correctly none | 0.940 | 0.870 |
| Original among the 5 shown (the most a model can find from them) | 0.510 | 0.560 |

A model shown the same 5 tickets has to beat the vote of 5 to show it adds anything beyond
search. On duplicates it can find at most the 51–56% whose original is among them.

## Planned conditions

Each runs over all 200 test tickets with the same model (`qwen3.5:4b`), and answers as JSON with
the component and type restricted to the valid values.

| | What the model sees |
|---|---|
| A. No memory | The ticket and the list of components |
| B. Similar past tickets | A + the 5 tickets the memory search returns (key, summary, start of description, component, type) |
| C. Written notes | A + one line per component, written once by the model from tickets before 2021 |
| D1. Mislabelled memory | B, with one of the 5 tickets given a wrong component |
| D2. Stale memory | B, with memory only from tickets 2+ years older (if time allows) |

## Running it

```
pip install -e ".[dev]"
tme split        # rebuild data/split.json
tme baselines    # rebuild results/baselines.json
pytest
```

## Limitations known so far

- The snapshot shows each ticket as it was in January 2022. A memory ticket's component, type and
  description may have been set or edited after a later ticket was filed, so memory can be
  slightly better informed than it would have been at the time.
- Flink's public Jira may be in the model's training data. Keys are masked and every condition
  uses the same model, so comparisons between conditions still hold.
- Component labels are set by people and are not always consistent.

## Licence

Code: MIT. Data: CC BY 4.0, from The Public Jira Dataset (see [data/README.md](data/README.md)).
