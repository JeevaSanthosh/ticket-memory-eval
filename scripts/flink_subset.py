"""Keep the Apache Flink tickets from extract_apache.py's output.

Copies each FLINK line unchanged from apache_issues.jsonl.gz to flink_issues.jsonl.gz and
prints the SHA-256 of the uncompressed result, to compare with data/README.md.

    python flink_subset.py [apache_issues.jsonl.gz] [flink_issues.jsonl.gz]
"""
import gzip
import hashlib
import json
import sys

source = sys.argv[1] if len(sys.argv) > 1 else "apache_issues.jsonl.gz"
target = sys.argv[2] if len(sys.argv) > 2 else "flink_issues.jsonl.gz"

count, digest = 0, hashlib.sha256()
with gzip.open(source, "rb") as f, gzip.open(target, "wb") as out:
    for line in f:
        if b'"FLINK"' in line and json.loads(line)["project"] == "FLINK":
            out.write(line)
            digest.update(line)
            count += 1
print(f"{count} Flink tickets written to {target}; sha256 of the uncompressed lines: {digest.hexdigest()}")
