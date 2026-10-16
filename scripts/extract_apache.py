"""Stream the Public Jira Dataset's mongodump archive and keep Apache issues.

Reads the archive straight out of the Zenodo zip (no unzip, no MongoDB), and
writes one compact JSON record per Apache issue to apache_issues.jsonl.gz:
key, project, dates, status, resolution, type, priority, components, labels,
summary, description (first 2,000 characters) and issue links. No people
fields are kept. Archive format: magic number, then blocks of
[header BSON, body BSONs..., terminator 0xFFFFFFFF] (mongo-tools archive/parser.go).

    python extract_apache.py
"""
import gzip
import json
import struct
import sys
import time
import zipfile
from collections import Counter

from bson.raw_bson import RawBSONDocument

ZIP = "2025-06-23 ThePublicJiraDataset.zip"
MEMBER = "ThePublicJiraDataset/3. DataDump/mongodump-JiraReposAnon.archive"
REPO = "Apache"
MAGIC = 0x8199E26D
DESCRIPTION_CHARS = 2000


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError
    return data


def read_bson_or_terminator(f):
    head = f.read(4)
    if len(head) < 4:
        raise EOFError
    if head == b"\xff\xff\xff\xff":
        return None
    (length,) = struct.unpack("<i", head)
    return head + read_exact(f, length - 4)


def name_of(value):
    return value.get("name") if value else None


def compact(doc):
    fields = doc["fields"]
    links = []
    for link in fields.get("issuelinks") or []:
        kind = link.get("type") or {}
        for direction in ("inwardIssue", "outwardIssue"):
            other = link.get(direction)
            if other:
                links.append({"type": kind.get("name"), "direction": direction[:-5],
                              "label": kind.get(direction[:-5]), "key": other.get("key")})
    description = fields.get("description") or ""
    return {
        "key": doc.get("key"),
        "project": (fields.get("project") or {}).get("key"),
        "created": fields.get("created"),
        "resolved": fields.get("resolutiondate"),
        "status": name_of(fields.get("status")),
        "resolution": name_of(fields.get("resolution")),
        "issuetype": name_of(fields.get("issuetype")),
        "priority": name_of(fields.get("priority")),
        "components": [c.get("name") for c in fields.get("components") or []],
        "labels": list(fields.get("labels") or []),
        "summary": fields.get("summary"),
        "description": description[:DESCRIPTION_CHARS],
        "description_chars": len(description),
        "links": links,
    }


def main():
    started = time.time()
    counts, errors = Counter(), Counter()
    sample_written = False
    log = open("extract.log", "w", encoding="utf-8")

    def report(message):
        print(message, flush=True)
        print(message, file=log, flush=True)

    report("Reading the archive; Apache progress is shown every 50,000 issues (about 20 per run).")
    with zipfile.ZipFile(ZIP) as z, z.open(MEMBER) as raw, gzip.GzipFile(fileobj=raw) as f, \
            gzip.open("apache_issues.jsonl.gz.part", "wt", encoding="utf-8") as out:
        (magic,) = struct.unpack("<I", read_exact(f, 4))
        assert magic == MAGIC, hex(magic)
        while True:
            try:
                header = read_bson_or_terminator(f)
            except EOFError:
                break
            if header is None:
                continue
            ns = RawBSONDocument(header)
            collection = ns.get("collection") if "collection" in ns else "<prelude>"
            while True:
                body = read_bson_or_terminator(f)
                if body is None:
                    break
                counts[collection] += 1
                if collection != REPO:
                    continue
                doc = RawBSONDocument(body)
                if not sample_written:
                    with open("sample_structure.json", "w", encoding="utf-8") as s:
                        json.dump({"top_level": list(doc.keys()),
                                   "fields": sorted(doc["fields"].keys())}, s, indent=1)
                    sample_written = True
                try:
                    out.write(json.dumps(compact(doc), ensure_ascii=False) + "\n")
                except Exception as e:  # keep going; report how many failed and why
                    errors[type(e).__name__] += 1
                if counts[REPO] % 50000 == 0:
                    report(f"{time.time() - started:5.0f}s  Apache issues: {counts[REPO]:,}  errors: {sum(errors.values())}")
    json.dump({"documents_per_collection": counts, "apache_errors": errors,
               "seconds": round(time.time() - started)}, open("extract_summary.json", "w", encoding="utf-8"), indent=1)
    import os
    os.replace("apache_issues.jsonl.gz.part", "apache_issues.jsonl.gz")
    report(f"Done in {time.time() - started:.0f}s. Apache issues: {counts[REPO]:,}, errors: {sum(errors.values())}. Wrote apache_issues.jsonl.gz")


if __name__ == "__main__":
    sys.exit(main())
