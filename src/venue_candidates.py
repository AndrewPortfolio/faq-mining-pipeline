# Stage 1: creates a most likely client venue names in data/pii/venue_candidates.csv 
# LOCATION is kept by default
# A place only seen a few time --> most likely client venue 
# No redactions: real venues are copied into data/pii/venue_denylist.txt by hand
# Counts threads not spans --> 20 thread convo is the same client 

# Usage:
#     venv/bin/python src/venue_candidates.py                  # -> data/pii/venue_candidates.csv
#     venv/bin/python src/venue_candidates.py --max-threads 1  # tighter list

from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import sys

from analyze_pii import DEFAULT_OUTDIR, review_spans, write_atomic
from shared.decisions import COMMON_LOCATION_PATH, Rules, normalize_term
from shared.pii_recognizers import VENUELIST_PATH


# Configuration

DEFAULT_OUT = "data/pii/venue_candidates.csv"  # beside the venue list it feeds, out of data/review
DEFAULT_MAX_THREADS = 2  # starting point; the printed distribution tunes this number 
COLUMNS = ["term", "threads", "spans", "context"]


# Counting

def count_locations(spans: list[dict], rules: Rules) -> dict[str, dict]:
    # common words and terms already on the venue list are settled, so they're left out of the counts
    terms: dict[str, dict] = {}
    for span in spans:
        if span["entity_type"] != "LOCATION":
            continue
        if rules.decide("LOCATION", span["text"])[1] != "location_baseline":
            continue
        term = normalize_term(span["text"])
        if not term:
            continue
        entry = terms.setdefault(term, {"threads": set(), "spans": 0, "context": span["context"]})
        entry["threads"].add(span["thrid"] or span["email_id"])  # no thread id: the email is its own thread
        entry["spans"] += 1
    return terms


def candidates(terms: dict[str, dict], max_threads: int) -> list[dict]:
    rows = [{"term": term, "threads": len(e["threads"]), "spans": e["spans"], "context": e["context"]}
            for term, e in terms.items() if len(e["threads"]) <= max_threads]
    return sorted(rows, key=lambda r: (-r["spans"], r["term"]))


def distribution(terms: dict[str, dict]) -> str:
    counts = collections.Counter(min(len(e["threads"]), 6) for e in terms.values())
    return "\n".join(f"  {'6+' if n == 6 else n} thread(s): {counts[n]:>6} terms" for n in sorted(counts))


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="List rare LOCATION terms as venue-list candidates")
    ap.add_argument("--reviewdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-threads", type=int, default=DEFAULT_MAX_THREADS)
    ap.add_argument("--common-location", default=COMMON_LOCATION_PATH)
    ap.add_argument("--venues", default=VENUELIST_PATH)
    return ap.parse_args(argv)


def run(args) -> int:
    spans = review_spans(args.reviewdir)
    if not spans:
        print(f"error: no review files in {args.reviewdir}", file=sys.stderr)
        return 1
    rules = Rules.load(location_path=args.common_location, venuelist_path=args.venues)
    terms = count_locations(spans, rules)
    rows = candidates(terms, args.max_threads)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    out = args.out
    write_atomic(out, buf.getvalue())

    print(f"{len(terms)} distinct LOCATION terms (after common words and the venue list):")
    print(distribution(terms))
    print(f"\n{len(rows)} candidates at <= {args.max_threads} thread(s) -> {out}")
    print("copy real venues into the venue list; everything else stays kept")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
