# Data

`flink_issues.jsonl.gz` holds all 25,492 Apache Flink tickets (June 2014 to 5 January 2022),
one JSON object per line. `split.json` lists the dev and test tickets and their right answers;
`tme split` rebuilds it from the tickets and a fixed seed.

## Source and licence

Derived from **The Public Jira Dataset**, version 7 (file `2025-06-23 ThePublicJiraDataset.zip`),
https://doi.org/10.5281/zenodo.5882881, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It accompanies:

> Lloyd Montgomery, Clara Lüders and Walid Maalej. "An Alternative Issue Tracking Dataset of
> Public Jira Repositories." *Proceedings of the 19th International Conference on Mining
> Software Repositories (MSR '22)*, 2022.

Changes made here: only the Apache Flink project; only the fields below; descriptions cut to
their first 2,000 characters. The dataset is already anonymised, and no people fields (reporter,
assignee, watchers, comments) are kept.

## Fields

| Field | Meaning |
|---|---|
| `key` | e.g. `FLINK-20123` |
| `created`, `resolved` | UTC, e.g. `2021-03-04T09:12:44.000+0000` |
| `status`, `resolution`, `issuetype`, `priority` | Names as shown in Jira |
| `components`, `labels` | Lists of names |
| `summary`, `description` | Title and the first 2,000 characters of the description |
| `description_chars` | Length of the full description |
| `links` | Links to other tickets: `type` (e.g. `Duplicate`), `direction` (`inward`/`outward`), `label` (e.g. `is duplicated by`) and the other ticket's `key` |

## How it was made

1. `scripts/extract_apache.py` reads the dataset's MongoDB dump straight out of the Zenodo zip
   (no unzipping, no MongoDB) and writes the 1,014,926 Apache tickets to `apache_issues.jsonl.gz`
   (250 MB, about 5 minutes). Every repository's document count matched the dataset paper.
2. `scripts/flink_subset.py` copies the Flink lines unchanged into `flink_issues.jsonl.gz`.
   The SHA-256 of the uncompressed lines is
   `9fa0929cc68830aa900798320472f15d1d9d4428edad851ce57362e03939927a`.

Run both from the folder holding the zip. `extract_apache.py` needs `pymongo` for its BSON reader
(not the separate `bson` package, which clashes with it); `flink_subset.py` needs only Python.

The snapshot shows each ticket as it was in January 2022. Field history was not extracted, so a
ticket's component, type and description may have been set or edited after the date a later
ticket was filed.
