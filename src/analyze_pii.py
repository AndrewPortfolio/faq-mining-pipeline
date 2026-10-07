# Stage 1: flag PII for review. Detection only -- nothing is redacted here
# Per input shard it writes three files:
#   emails-NNNNN.spans.csv       the review file edit (flip decision to keep)
#   emails-NNNNN.spans.orig.csv  untouched copy, so apply_redactions can spot deleted rows
#   emails-NNNNN.view.txt        every email with its spans bracketed, for reading straight through
# Usage:
#     venv/bin/python src/analyze_pii.py              # resumes; ~1.1 h for all 15 shards
#     venv/bin/python src/analyze_pii.py --limit 50   # smoke test into data/review/smoke
#     venv/bin/python src/analyze_pii.py --force      # re-analyze, never touches edited files

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import sys

from shared.pii_recognizers import (ALLOWLIST_PATH, DENYLIST_PATH, ENTITIES, PLACELIST_PATH,
                                    VENUELIST_PATH, build_analyzer, load_allowlist)
from shared.decisions import COMMON_DATETIME_PATH, COMMON_LOCATION_PATH, Rules
from shared.pipeline import (Checkpoint, Progress, Stats, count_rows, read_shard, shard_paths,
                             write_atomic)
from shared.signatures import SIGNATURE_ENTITY, SIGNATURE_RULE, signature_zones


# Configuration

DEFAULT_INDIR = "data/extracted"
DEFAULT_OUTDIR = "data/review"

# Free text only: the headers get hashed in apply_redactions, so running NER on them is wasted work
FIELDS = ("subject", "body")

CONTEXT_CHARS = 40    # enough either side to judge a hit without opening view.txt
PROGRESS_EVERY = 200  # ~30 s between lines at trf's ~7 emails/s

COLUMNS = ["email_id", "thrid", "field", "entity_type", "start", "end", "text", "context",
           "score", "recognizer", "decision", "rule", "note"]


# Reading

def review_spans(reviewdir: str) -> list[dict]:
    # every shard's current review file; the .orig copies don't match the pattern
    spans = []
    for path in sorted(glob.glob(os.path.join(reviewdir, "emails-*.spans.csv"))):
        with open(path, encoding="utf-8", newline="") as fh:
            spans.extend(csv.DictReader(fh))
    return spans


# Detection

def context_snippet(text: str, start: int, end: int) -> str:
    # [[ ]] marks the span itself; newlines collapse so the cell stays one spreadsheet line
    snippet = f"{text[max(0, start - CONTEXT_CHARS):start]}[[{text[start:end]}]]{text[end:end + CONTEXT_CHARS]}"
    return " ".join(snippet.split())


def review_row(row: dict, field: str, text: str, start: int, end: int, *, entity_type: str,
               score: float, recognizer: str, decision: str, rule: str) -> dict:
    # every review-file row is built here, new columns apply to all fields
    return {
        "email_id": row.get("id") or "",
        "thrid": row.get("thrid") or "",
        "field": field,
        "entity_type": entity_type,
        "start": start,
        "end": end,
        "text": text[start:end],
        "context": context_snippet(text, start, end),
        "score": round(score, 2),
        "recognizer": recognizer,
        "decision": decision,
        "rule": rule,
        "note": "",
    }


def row_spans(analyzer, row: dict, allow_list: list[str], rules: Rules) -> list[dict]:
    # decision is pre-filled by shared.decisions; rule names which one fired
    spans = []
    for field in FIELDS:
        text = row.get(field) or ""
        if not text:
            continue
        results = analyzer.analyze(text=text, language="en", entities=ENTITIES, allow_list=allow_list)
        for result in sorted(results, key=lambda r: (r.start, -r.score)):
            recognizer = (result.recognition_metadata or {}).get("recognizer_name", "")
            decision, rule = rules.decide(result.entity_type, text[result.start:result.end],
                                          result.score, recognizer)
            spans.append(review_row(row, field, text, result.start, result.end,
                                    entity_type=result.entity_type, score=result.score,
                                    recognizer=recognizer, decision=decision, rule=rule))
    # The signature block is another row
    body = row.get("body") or ""
    redact = [(s["start"], s["end"], s["entity_type"]) for s in spans
              if s["field"] == "body" and s["decision"] == "redact"]
    for start, end in signature_zones(body, redact):
        spans.append(review_row(row, "body", body, start, end,
                                entity_type=SIGNATURE_ENTITY, score=1.0,
                                recognizer=SIGNATURE_RULE, decision="redact", rule=SIGNATURE_RULE))
    return spans


# Writing

def render_csv(spans: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(spans)
    return buf.getvalue()


def render_view(row: dict, spans: list[dict]) -> str:
    # markers are inserted right to left so the earlier offsets don't shift
    lines = [f"=== {row.get('id')} | {row.get('date')} | {row.get('direction')} | "
             f"thrid {row.get('thrid')} | {len(spans)} spans"]
    for field in FIELDS:
        text = row.get(field) or ""
        mine = [s for s in spans if s["field"] == field]
        # a signature shows as one block otherwise the spans inside it would garble its offsets
        blocks = [(s["start"], s["end"]) for s in mine if s["entity_type"] == SIGNATURE_ENTITY]
        shown = [s for s in mine if s["entity_type"] == SIGNATURE_ENTITY
                 or not any(a <= s["start"] and s["end"] <= b for a, b in blocks)]
        for span in sorted(shown, key=lambda s: -s["start"]):
            marked = f"[[{span['entity_type']}:{text[span['start']:span['end']]}]]"
            text = f"{text[:span['start']]}{marked}{text[span['end']:]}"
        lines.append(f"{field}: {text}")
    return "\n".join(lines) + "\n"


def edited(base: str) -> bool:
    # review edits are never overwriten 
    original = f"{base}.spans.orig.csv"
    if not os.path.exists(original):
        return True  # no baseline to compare against, so assume the file is edited 
    with open(f"{base}.spans.csv", encoding="utf-8") as review, open(original, encoding="utf-8") as orig:
        return review.read() != orig.read()


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1: flag PII spans for review")
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--denylist", default=DENYLIST_PATH)
    ap.add_argument("--venues", default=VENUELIST_PATH)
    ap.add_argument("--places", default=PLACELIST_PATH)
    ap.add_argument("--allowlist", default=ALLOWLIST_PATH)
    ap.add_argument("--common-datetime", default=COMMON_DATETIME_PATH)
    ap.add_argument("--common-location", default=COMMON_LOCATION_PATH)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: analyze N emails into <outdir>/smoke")
    ap.add_argument("--force", action="store_true",
                    help="re-analyze shards already done (edited review files are still left alone)")
    return ap.parse_args(argv)


def run(args) -> int:
    paths = shard_paths(args.indir)
    if not paths:
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1

    # A shard cut short by --limit would look finished to the next run, smoke tests go elsewhere
    outdir = os.path.join(args.outdir, "smoke") if args.limit else args.outdir
    if args.limit:
        print(f"smoke test: {args.limit} emails -> {outdir} (kept apart from the real review set)")
    os.makedirs(outdir, exist_ok=True)

    checkpoint = Checkpoint(os.path.join(outdir, "checkpoint.json"))
    if args.force:
        checkpoint.clear()
    stats = Stats()
    stats.update((checkpoint.load() or {}).get("stats", {}))

    allow_list = load_allowlist(args.allowlist)
    rules = Rules.load(args.common_datetime, args.common_location)
    analyzer = build_analyzer(args.denylist, args.venues, args.places)  # loads trf

    progress = Progress(total=count_rows(args.indir), every=PROGRESS_EVERY,
                        start=stats["emails"], unit="emails", scale=1.0)
    done = 0
    try:
        for path in paths:
            base = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0])
            if os.path.exists(f"{base}.spans.csv"):
                if edited(base):
                    print(f"skipping {base}.spans.csv: it holds your edits")
                    stats["shards_edited"] += 1
                    continue
                if not args.force:
                    stats["shards_done"] += 1
                    continue

            rows = read_shard(path)
            if args.limit:
                rows = rows[:max(0, args.limit - done)]
                if not rows:
                    break

            spans, views = [], []
            for row in rows:
                found = row_spans(analyzer, row, allow_list, rules)
                spans.extend(found)
                views.append(render_view(row, found))
                stats["emails"] += 1
                stats["spans"] += len(found)
                if not found:
                    stats["emails_no_spans"] += 1
                if not row.get("id"):
                    stats["missing_id"] += 1
                for span in found:
                    stats[span["entity_type"]] += 1
                    stats[f"decision_{span['decision']}"] += 1
                progress.tick(stats["emails"], note=os.path.basename(base))

            table = render_csv(spans)
            write_atomic(f"{base}.spans.csv", table)
            write_atomic(f"{base}.spans.orig.csv", table)  # byte-identical on purpose: the pair finds deleted rows
            write_atomic(f"{base}.view.txt", "\n".join(views))
            stats["shards"] += 1
            done += len(rows)
            # the review files are the resume signal; this carries stats between runs
            if not args.limit:
                checkpoint.save(last_shard=os.path.basename(base), stats=dict(stats))
    except KeyboardInterrupt:
        print("\ninterrupted; finished shards are saved", file=sys.stderr)

    print(f"\ndone in {progress.elapsed_min():.1f} min -> {outdir}")
    print(stats.render())
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
