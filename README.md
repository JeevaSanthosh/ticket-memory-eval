# ticket-memory-eval

Does memory of past tickets make a small local model better at handling new ones, which kind
of memory is worth its prompt tokens, and when does memory make answers worse?

A ticket assistant reads a new Jira ticket from Apache Flink and answers two questions:

1. **Routing:** which component does it belong to, and what type is it (Bug, Improvement,
   New Feature, Technical Debt)?
2. **Duplicates:** does it repeat an earlier ticket, and if so, which one?

The model runs locally through [Ollama](https://ollama.com). CI never calls a model.

**Status:** conditions A (no memory), B (similar past tickets), C (written notes) and D1
(mislabelled memory) recorded on test; A and B also on dev.

## Results on test

`qwen3.5:4b` on a CPU, 200 test tickets from 2021 that played no part in any design choice,
0 invalid answers. B and D1 use the duplicate prompt chosen on dev (v2). Changes are paired
bootstrap 95% intervals from `tme compare`; answers are in [results/test/](results/test/).

| | A, no memory | B, 5 similar earlier tickets | C, written notes | D1, B with one wrong label | Vote of B's 5 (no model) |
|---|---:|---:|---:|---:|---:|
| Component | 0.515 | **0.595** | 0.485 | 0.545 | 0.455 |
| Area | 0.655 | **0.720** | 0.595 | 0.675 | 0.595 |
| Issue type | 0.745 | **0.770** | 0.765 | 0.760 | 0.710 |
| Duplicate (right original, or none) | 0.500 | **0.550** | 0.500 | 0.520 | 0.520 |
| … original found, on duplicates | 0.000 | 0.110 | 0.000 | 0.060 | **0.170** |
| … correctly none, on non-duplicates | **1.000** | 0.990 | **1.000** | 0.980 | 0.870 |
| Prompt tokens, mean | 856 | 1,635 | 1,884 | 1,635 | |
| Seconds per ticket, median | 15 | 33 | 17 | 34 | |

| Change in component accuracy | | 95% interval |
|---|---:|---|
| B against A (memory) | **+0.080** | +0.020 to +0.140 |
| B against the vote of its 5 tickets (the model over search) | **+0.140** | +0.070 to +0.210 |
| C against A (notes) | −0.030 | −0.085 to +0.025 |
| C against B (notes instead of similar tickets) | **−0.110** | −0.175 to −0.045 |
| D1 against B (one wrong label) | **−0.050** | −0.095 to −0.010 |

- **Similar past tickets help routing, and the model adds something beyond search.** With the 5
  most similar earlier tickets in the prompt, the model picks the right component on 16 more
  tickets than without them, and on 28 more than a vote of those same 5 tickets. It takes its
  answer from one of the shown tickets 171 times out of 200, but doesn't simply follow the
  majority: it is right on 40 tickets where the vote is wrong, and wrong on 12 where the vote is
  right. The gain over A is smaller than on dev (+0.170): A scored 0.465 on dev and 0.515 on
  test, B 0.635 and 0.595.
- **Written notes don't help; they cost area accuracy** (−0.060, −0.110 to −0.010, against A).
  The 73 notes are short (9 words at the median), but they lose the project's habits. Flink
  files a failing test ("JMXJobManagerMetricTest crashes on azure") under the component whose
  code is tested. The note for `Build System / Azure Pipelines`, written from its own recent
  tickets, reads "Azure CI stability, test execution, and Python wheel builds". C answers
  that component 18 times, against 8 for A and 4 for B, and it is right for only 4 test
  tickets. Similar tickets carry the habit; a summary of each component doesn't.
- **One wrong label in memory costs 10 right components, but is rarely copied.** D1 shows the
  most similar ticket with a wrong component. The model answers with that wrong label only 4
  times out of 200. Instead it loses its footing: of the 111 tickets where B's answer matched
  that ticket's real component, 25 switch to some other component in D1. The wrong label also
  hides duplicates. B found 8 originals in first place; D1, with that original's component
  shown wrong, finds 3.
- **Memory doesn't help the issue type.** A and B give the same type on 170 of 200 tickets.
- **On duplicates the model is no better than search.** B names an earlier ticket 17 times and
  is right 11 times; the vote names 39 and is right 17 times. Overall the two are level. On dev,
  three prompt designs only moved how readily the model called a duplicate, never how well it
  told one from a related ticket (see [Dev results](#dev-results)).
- **Cost:** B nearly doubles the prompt (+780 tokens) and C adds more (+1,030), but the time
  doesn't follow the tokens. Ollama sometimes reuses the processed start of a prompt from the
  previous call, and C's notes sit in the part that never changes. So C ran nearly as fast as A,
  while B's 5 tickets change with every ticket and took twice as long. Prompt tokens are the
  fairer measure of cost; the time depends on the runtime.

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

## Conditions

Each runs over all 200 test tickets with the same model and settings (`eval.toml`). The answer
is JSON, held by Ollama to a schema: the component must be one on offer, the type one of the
four, and the duplicate one of the tickets shown or null.

| | What the model sees | |
|---|---|---|
| A. No memory | Instructions ([prompts/route_v1.md](prompts/route_v1.md)), the components used by any earlier ticket, and the new ticket | Ready |
| B. Similar past tickets | A + [prompts/duplicates_v2.md](prompts/duplicates_v2.md) and the same 5 tickets the vote of 5 uses: key, summary, first 300 characters of the description, component, type | Ready |
| C. Written notes | A, with each component in the list followed by a note of at most 15 words on what its tickets are about ([prompts/notes_v1.md](prompts/notes_v1.md)). The model writes the notes once per set, one area at a time, from the summaries of each component's 20 most recent tickets before the set starts ([prompts/write_notes_v1.md](prompts/write_notes_v1.md), [src/tme/notes.py](src/tme/notes.py)), so by the end of 2021 they are up to a year old | Test |
| D1. Mislabelled memory | B, except that the most similar earlier ticket is shown with a wrong component, drawn at random (seeded by the new ticket's key) from the components on offer other than its own and the new ticket's | Test |
| D2. Stale memory | B, except that memory holds only tickets created at least two years (730 days) before the new one, as if the store had not been updated since; the model is not told | Ready |

A and B share every word except the memory part, so a difference between them comes from the
memory. [src/tme/conditions.py](src/tme/conditions.py) builds the requests; tests check that no
request shows the new ticket's own key or any ticket created after it.

### Dev results

`qwen3.5:4b` on a CPU, all 200 dev tickets, 0 invalid answers. A uses `route_v1`; B adds
`duplicates_v1` or `duplicates_v2`, which differ only in the duplicate instructions. The answers
for A and B v2 are in [results/dev/](results/dev/). Changes and intervals come from `tme compare`
(paired bootstrap, 95%).

| | A, no memory | B, v1 | B, v2 | Vote of the same 5 |
|---|---:|---:|---:|---:|
| Component | 0.465 | **0.640** | 0.635 | 0.520 |
| Area | 0.645 | **0.775** | **0.775** | 0.635 |
| Issue type | **0.760** | **0.760** | 0.755 | 0.710 |
| Duplicate | 0.500 | 0.415 | 0.535 | **0.570** |
| … original found, on duplicates | 0.000 | **0.390** | 0.080 | 0.200 |
| … correctly none, on non-duplicates | **1.000** | 0.440 | 0.990 | 0.940 |
| Seconds per ticket, median | 23 | 35 | 36 | |

- **Memory helps routing.** B v2 picks the right component on 34 more tickets than A (+0.170,
  +0.095 to +0.245) and on 23 more than the vote of the 5 tickets it was shown (+0.115, +0.050
  to +0.180), so the model adds something beyond search. Issue type doesn't move.
- **On duplicates, the prompt sets how readily the model calls a duplicate, but not how well it
  tells one apart.** v1 names a ticket on 142 of 200: it finds 39 of the 51 originals that are
  among the 5, but wrongly flags 56 of the 100 non-duplicates. v2 says the 5 were found by
  keyword search and that usually none is a duplicate. It names a ticket on only 14, finding 8 and
  wrongly flagging 1. Overall v2 is better (+0.120, +0.030 to +0.210), and routing is unchanged.
  But neither beats the vote of 5 (v2 −0.035, −0.090 to +0.015). Keeping only those v1 picks
  whose search score clears a threshold set on dev reaches 0.575, no better than the vote.
- **A third design, v3, asked for a label on each earlier ticket (same, related or different)**
  ([#6](https://github.com/JeevaSanthosh/ticket-memory-eval/pull/6), kept as a record, not
  merged). The real original got "same" 37 times out of 51, but so did 90 of the other 949 shown
  tickets. v3 landed between v1 and v2 (0.500) and was the slowest (48 s a ticket), so v2 went to
  test: it had the highest dev duplicate accuracy, routing was the same for all three, and it was
  the cheapest.

### Recording and replaying

Model answers are recorded once on a machine running [Ollama](https://ollama.com), and saved in
`cassettes/` under a hash of the whole request (model, prompt, ticket text, schema, options).
Replaying reads them back without a model. Any change to the request changes the hash, so an old
recording can never pass for a new prompt: replay stops and asks for a new recording.

## Running it

```
pip install -e ".[dev]"
tme split        # rebuild data/split.json
tme baselines    # rebuild results/baselines.json
pytest

# with Ollama running and `ollama pull qwen3.5:4b` done:
tme run B dev --limit 5        # record the first 5 dev tickets, printing the time each takes
tme notes test                 # C only: the model writes the component notes, once per set
tme run B dev --mode replay    # score from recordings only, as CI does

# compare two sets of answers to the same tickets, e.g. B against the vote of the same 5 tickets
tme compare results/baselines/dev/vote_of_5.jsonl runs/dev/B --set dev
```

`tme run` writes its predictions, scores and timings to `runs/<set>/<condition>/` (or `--out`).
`tme baselines` also writes each baseline's answers to `results/baselines/<set>/`. Committed model
results live in `results/<set>/<condition>/`, made with
`tme run <condition> <set> --mode replay --out results/<set>/<condition>`.

On every pull request, CI's `gate` job replays the recordings and checks they give the committed
results exactly. Then it compares each condition with the target branch's results and blocks
the merge on a significant drop (as `tme compare` defines it) in component, area, issue type or
duplicate accuracy.

`tme compare` is paired: each bootstrap resample (10,000) draws the same tickets for both sets of
answers, so the 95% interval is for the change on identical tickets. A field is **better** when
it rises by more than 0.02 and the interval lies above zero, **fail** when it falls by more than
0.02 and the interval lies below zero, **warn** when it falls by more than 0.02 but the interval
includes zero, and **pass** otherwise.

## Limitations known so far

- Dev was used for three duplicate prompts and the rules tried on them. The test numbers above
  are the honest ones; the dev numbers are where the choices were made.
- One ticket was filed twice on the same day (FLINK-21615 and FLINK-21618, both duplicates of
  FLINK-21614), so both are in the test set. Without memory their requests are identical, and A
  answered them from one recording.
- The snapshot shows each ticket as it was in January 2022. A memory ticket's component, type and
  description may have been set or edited after a later ticket was filed, so memory can be
  slightly better informed than it would have been at the time.
- Flink's public Jira may be in the model's training data. Keys are masked and every condition
  uses the same model, so comparisons between conditions still hold.
- Component labels are set by people and are not always consistent.

## Licence

Code: MIT. Data: CC BY 4.0, from The Public Jira Dataset (see [data/README.md](data/README.md)).
