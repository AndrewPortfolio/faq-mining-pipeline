# Stage 1: apply the redactions approved --> the script that writes redacted text
# Reads the extracted shards plus edited review files, writes to data/redacted/
# A deleted review row, an unknown decision, or a span whose text is gone
# is a hard error, because each one would quietly leave PII in the output
# Usage:
#     venv/bin/python src/apply_redactions.py           # every shard that has a review file
#     venv/bin/python src/apply_redactions.py --force   # rewrite shards already redacted

from __future__ import annotations

import argparse
import csv
import collections
import hashlib
import hmac
import json
import os
import secrets
import sys
from email.utils import getaddresses

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

# the review-file contract lives with the script that writes it
from analyze_pii import FIELDS, read_shard, write_atomic
from shared.pipeline import Checkpoint, Stats, shard_paths


# Configuration

DEFAULT_INDIR = "data/extracted"
DEFAULT_REVIEWDIR = "data/review"
DEFAULT_OUTDIR = "data/redacted"
DEFAULT_KEY_PATH = "data/pii/hash_key"

DECISIONS = {"redact", "keep"}
HASH_CHARS = 16       # 64 bits, more than enough to keep 29k senders apart
MAX_ERRORS = 20       # printed per shard; the rest are counted


# Hashing

def load_key(path: str) -> bytes:
    # Keyed, because a plain digest of an email address can be cracked by hashing guesses
    # Same key in, same hashes out, so "these two emails are the same client" still holds later
    if os.path.exists(path):
        key = open(path, encoding="utf-8").read().strip()
        if key:
            return key.encode()
    key = secrets.token_hex(32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(key + "\n")
    os.chmod(path, 0o600)
    print(f"wrote a new hash key to {path}: keep it, or later runs won't match these hashes")
    return key.encode()


def keyed_hash(value: str, key: bytes) -> str:
    # normalized first so "Thao <t@x.com>" and "t@x.com" land on the same hash
    value = (value or "").strip().lower()
    return hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:HASH_CHARS] if value else ""


def plain_hash(value: str) -> str:
    # Message-IDs carry enough randomness that a plain digest can't be guessed back
    value = (value or "").strip()
    return hashlib.sha256(value.encode()).hexdigest()[:HASH_CHARS] if value else ""


# Review files

def read_spans(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        spans = list(csv.DictReader(fh))
    for line, span in enumerate(spans, start=2):  # +2: header row, then 1-based
        span["_line"] = line
    return spans


def span_key(span: dict) -> tuple:
    # No entity_type: relabelling a row is allowed, deleting one is not
    return (span["email_id"], span["field"], span["start"], span["end"])


def locate(span: dict, text: str) -> list[tuple[int, int]]:
    # Offsets count only while they still slice out the same text. Once they don't, the row was
    # hand-edited (or added with no offsets at all), so every occurrence of the text goes
    wanted = span["text"]
    if not wanted:
        return []
    start, end = span["start"], span["end"]
    if start.isdigit() and end.isdigit():
        s, e = int(start), int(end)
        if 0 <= s < e <= len(text) and text[s:e] == wanted:
            return [(s, e)]
    hits, at = [], text.find(wanted)
    while at >= 0:
        hits.append((at, at + len(wanted)))
        at = text.find(wanted, at + len(wanted))
    return hits


def analyzed_ids(path: str) -> set[str]:
    # view.txt is the record of which emails were analyzed. A shard row missing from it was never
    # reviewed, so it would otherwise pass straight through with nothing redacted
    ids = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("=== "):
                ids.add(line[4:].split(" | ", 1)[0].strip())
    return ids


def check_decisions(spans: list[dict], review: str) -> list[str]:
    errors = []
    for span in spans:
        decision = (span.get("decision") or "").strip().lower()
        if decision not in DECISIONS:
            shown = decision or "(empty)"
            errors.append(f"{review}:{span['_line']}: decision {shown!r} is not redact or keep")
        span["decision"] = decision
    return errors


def check_nothing_deleted(spans: list[dict], original: list[dict], review: str) -> list[str]:
    missing = collections.Counter(map(span_key, original)) - collections.Counter(map(span_key, spans))
    if not missing:
        return []
    # A deleted row has no decision at all, so it would slip through unredacted
    return [f"{review}: {sum(missing.values())} row(s) from the original are gone; "
            f"set decision to keep instead of deleting. First: {list(missing)[:3]}"]


# Redacting

def redact_field(text: str, spans: list[dict], anonymizer) -> tuple[str, int, list[dict]]:
    results, unfound = [], []
    for span in spans:
        hits = locate(span, text)
        if not hits:
            unfound.append(span)
            continue
        results.extend(RecognizerResult(span["entity_type"] or "PII", s, e, 1.0) for s, e in hits)
    if not results:
        return text, 0, unfound
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, len(anonymized.items), unfound


def redact_row(row: dict, spans: list[dict], key: bytes, anonymizer) -> tuple[dict, list[dict]]:
    fields, applied, unfound = {}, 0, []
    for field in FIELDS:
        wanted = [s for s in spans if s["field"] == field and s["decision"] == "redact"]
        text, count, missing = redact_field(row.get(field) or "", wanted, anonymizer)
        fields[field] = text
        applied += count
        unfound.extend(missing)
    body = fields["body"]
    redacted = {
        "id": plain_hash(row.get("id")),
        "thrid": row.get("thrid"),
        "date": row.get("date"),
        "direction": row.get("direction"),
        "labels": row.get("labels") or [],
        # "from" is dropped rather than hashed: its only extra content was the display name
        "sender": keyed_hash(row.get("sender") or "", key),
        "to": [keyed_hash(addr, key) for _name, addr in getaddresses([row.get("to") or ""]) if addr],
        "is_automated": row.get("is_automated", False),
        "subject": fields["subject"],
        "body": body,
        "n_words": len(body.split()),
        "body_source": row.get("body_source"),
        "n_attachments": row.get("n_attachments", 0),
        "attachments": [{"filename": keyed_hash(att.get("filename", ""), key),
                         "content_type": att.get("content_type", ""),
                         "approx_bytes": att.get("approx_bytes", 0)}
                        for att in row.get("attachments") or []],
        "offset": row.get("offset"),
        "n_redacted": applied,
    }
    return redacted, unfound


def redact_shard(rows: list[dict], spans: list[dict], review: str, key: bytes,
                 anonymizer, stats: Stats) -> tuple[list[dict], list[str]]:
    by_id: dict[str, list[dict]] = collections.defaultdict(list)
    for span in spans:
        by_id[span["email_id"]].append(span)

    ids = {row.get("id") or "" for row in rows}
    errors = [f"{review}:{span['_line']}: email_id {span['email_id']} is not in this shard"
              for span in spans if span["email_id"] not in ids]

    out = []
    for row in rows:
        redacted, unfound = redact_row(row, by_id.get(row.get("id") or "", []), key, anonymizer)
        errors.extend(f"{review}:{span['_line']}: text is no longer in the {span['field']} "
                      f"of {span['email_id']}" for span in unfound)
        stats["spans_redacted"] += redacted["n_redacted"]
        out.append(redacted)
    return out, errors


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1b: apply reviewed redactions")
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--reviewdir", default=DEFAULT_REVIEWDIR)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--hash-key", default=DEFAULT_KEY_PATH)
    ap.add_argument("--force", action="store_true", help="rewrite shards already redacted")
    return ap.parse_args(argv)


def run(args) -> int:
    paths = shard_paths(args.indir)
    if not paths:
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    key = load_key(args.hash_key)
    anonymizer = AnonymizerEngine()
    checkpoint = Checkpoint(os.path.join(args.outdir, "checkpoint.json"))
    if args.force:
        checkpoint.clear()
    stats = Stats()
    stats.update((checkpoint.load() or {}).get("stats", {}))
    failed = False

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        review = os.path.join(args.reviewdir, f"{name}.spans.csv")
        original = os.path.join(args.reviewdir, f"{name}.spans.orig.csv")
        view = os.path.join(args.reviewdir, f"{name}.view.txt")
        out_path = os.path.join(args.outdir, f"{name}.jsonl")

        if not os.path.exists(review):
            print(f"skipping {name}: no review file yet")
            stats["shards_unreviewed"] += 1
            continue
        if os.path.exists(out_path) and not args.force:
            stats["shards_done"] += 1
            continue

        spans = read_spans(review)
        rows = read_shard(path)
        errors = check_decisions(spans, review)
        if os.path.exists(view):
            analyzed = analyzed_ids(view)
            unanalyzed = [r for r in rows if str(r.get("id") or "") not in analyzed]
            if unanalyzed:
                errors.append(f"{view}: {len(unanalyzed)} of {len(rows)} emails in this shard were "
                              f"never analyzed (a --limit run, or a stale review file)")
        else:
            errors.append(f"{view}: missing, so unanalyzed emails can't be detected")
        if os.path.exists(original):
            errors += check_nothing_deleted(spans, read_spans(original), review)
        else:
            errors.append(f"{original}: missing, so deleted rows can't be detected")

        redacted, span_errors = redact_shard(rows, spans, review, key, anonymizer, stats)
        errors += span_errors

        if errors:
            failed = True
            stats["shards_failed"] += 1
            print(f"\n{name}: {len(errors)} problem(s), nothing written", file=sys.stderr)
            for message in errors[:MAX_ERRORS]:
                print(f"  {message}", file=sys.stderr)
            if len(errors) > MAX_ERRORS:
                print(f"  ... and {len(errors) - MAX_ERRORS} more", file=sys.stderr)
            continue

        write_atomic(out_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in redacted))
        stats["shards"] += 1
        stats["emails"] += len(redacted)
        stats["spans_kept"] += sum(1 for s in spans if s["decision"] == "keep")
        checkpoint.save(last_shard=name, stats=dict(stats))

    print(f"\n-> {args.outdir}")
    print(stats.render())
    return 1 if failed else 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
