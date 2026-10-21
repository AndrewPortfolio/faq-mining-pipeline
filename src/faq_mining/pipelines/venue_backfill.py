# Stage 1 backfill: run the venue recognizers alone over every shard and append what they find to the review files
# analyze_pii --force never touches an edited review file (14 of 15 are), so a deny-list or label-pattern change
# reaches them through here instead of the save/reset/re-add process. No trf, so a full pass takes seconds
#   candidates  writes data/pii/venue_candidates.txt: each Location:/Venue: value no list has yet, with how many
#               redacted emails still hold it. Copy the real venues into venue_denylist.txt by hand
#   backfill    appends a review row for every label value and deny-list venue no redact row covers yet, decided by
#               shared.decisions like any other. Never rewrites an existing row or spans.orig.csv, so
#               apply_redactions still sees nothing deleted, and a second run appends nothing
# Both print counts only: the values themselves are PII
# Usage:
#     PYTHONPATH=src .venv/bin/python -m faq_mining.pipelines.venue_backfill candidates
#     PYTHONPATH=src .venv/bin/python -m faq_mining.pipelines.venue_backfill backfill --dry-run
#     PYTHONPATH=src .venv/bin/python -m faq_mining.pipelines.venue_backfill backfill
#     .venv/bin/python src/apply_redactions.py --force                # then re-embed, recluster, retriage

import argparse
import collections
import csv
import os
import re
import sys
from collections.abc import Iterable

from presidio_analyzer import PatternRecognizer

from analyze_pii import FIELDS, read_rows, render_csv, review_row
from apply_redactions import locate
from faq_mining import config
from faq_mining.storage.shards import iter_shard_rows, shard_paths, write_atomic
from faq_mining.utils.reporting import Stats
from shared.decisions import Rules, normalize_term
from shared.pii_recognizers import load_entries, load_places, load_venues, pattern_recognizers, venue_recognizers

# Configuration

LABEL_RECOGNIZER = "label_location"
CANDIDATES_FILE = f"{config.PII_DIR}/venue_candidates.txt"
NOTE = "backfill"

_TAG_RE = re.compile(r"<[A-Z_]+>")
_WORD_RE = re.compile(r"\w+")


# Recognizers


def label_recognizer() -> PatternRecognizer:
    return next(r for r in pattern_recognizers() if r.name == LABEL_RECOGNIZER)


def found(recognizer: PatternRecognizer, text: str) -> list[tuple[int, int, float]]:
    # (start, end, score) per hit; nlp_artifacts=None, since none of these read trf
    return [(r.start, r.end, r.score) for r in recognizer.analyze(text, recognizer.supported_entities, None)]


# Candidates


def candidate(value: str) -> str:
    # the venue part of a label value: "Sunny Acres Pavilion, Irvine CA" -> "Sunny Acres Pavilion"
    return value.split(",", 1)[0].strip(" \t.;:-")


def harvest(texts: Iterable[str], listed: set[str]) -> dict[str, tuple[str, int]]:
    # normalized value -> (its most common spelling, label lines). Left out: no capital (a description, not a name),
    # a leading digit (a street address, STREET_ADDRESS's job), and anything a list already has
    recognizer = label_recognizer()
    spellings = collections.defaultdict(collections.Counter)
    for text in texts:
        for start, end, _ in found(recognizer, text):
            value = candidate(text[start:end])
            term = normalize_term(value)
            if term and not value[0].isdigit() and any(c.isupper() for c in value) and term not in listed:
                spellings[term][value] += 1
    return {term: (c.most_common(1)[0][0], sum(c.values())) for term, c in spellings.items()}


def leaking(terms: list[str], texts: Iterable[str]) -> collections.Counter:
    # term -> redacted emails still holding it outside a tag. The word-set check first keeps this to seconds
    patterns = {t: re.compile(rf"(?<!\w){re.escape(t)}(?!\w)") for t in terms}
    needs = {t: set(_WORD_RE.findall(t)) for t in terms}
    counts = collections.Counter()
    for text in texts:
        text = _TAG_RE.sub(" ", text).lower()
        words = set(_WORD_RE.findall(text))
        counts.update(t for t in terms if needs[t] <= words and patterns[t].search(text))
    return counts


def render_candidates(found_terms: dict[str, tuple[str, int]], leaks: collections.Counter) -> str:
    lines = [
        "# Location:/Venue: values no list has yet, written by `venue_backfill candidates`",
        "# Copy the real venues into venue_denylist.txt. Leave out generic ones (Backyard, Home): an entry redacts",
        "# that word everywhere, FAQ text included",
        "# leaking = redacted emails still holding it, lines = label lines it came from",
        f"# {'leaking':>7}  {'lines':>5}  value",
    ]
    order = sorted(found_terms, key=lambda t: (-leaks[t], -found_terms[t][1], t))
    lines += [f"  {leaks[t]:>7}  {found_terms[t][1]:>5}  {found_terms[t][0]}" for t in order]
    return "\n".join(lines) + "\n"


# Backfill


def covered(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    # every character of [start, end) is inside some redact span already
    at = start
    for s, e in sorted(intervals):
        if s > at:
            break
        at = max(at, e)
        if at >= end:
            return True
    return False


def new_spans(row: dict, existing: list[dict], recognizers: list[PatternRecognizer], rules: Rules) -> list[dict]:
    # review rows for this email that the file doesn't hold yet. A row at the same offsets counts whatever its
    # decision, so a span flipped to keep by hand stays kept
    email_id = row.get("id") or ""
    mine = [s for s in existing if s["email_id"] == email_id]
    keys = {(s["field"], s["start"], s["end"]) for s in mine}
    out = []
    for field in FIELDS:
        text = row.get(field) or ""
        if not text:
            continue
        redacted = [hit for s in mine if s["field"] == field and s["decision"] == "redact" for hit in locate(s, text)]
        for recognizer in recognizers:
            for start, end, score in found(recognizer, text):
                if (field, str(start), str(end)) in keys or covered(start, end, redacted):
                    continue
                decision, rule = rules.decide("LOCATION", text[start:end], score, recognizer.name)
                span = review_row(
                    row,
                    field,
                    text,
                    start,
                    end,
                    entity_type="LOCATION",
                    score=score,
                    recognizer=recognizer.name,
                    decision=decision,
                    rule=rule,
                )
                out.append({**span, "note": NOTE})
                keys.add((field, str(start), str(end)))
                if decision == "redact":
                    redacted.append((start, end))
    return out


def read_review(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def appended(path: str, spans: list[dict]) -> str:
    # the file's own bytes, then the new rows without a second header
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    if text and not text.endswith("\n"):
        text += "\n"
    return text + render_csv(spans).split("\n", 1)[1]


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--indir", default=config.EXTRACTED_DIR, help="Stage 0's output, what review offsets count in")
    ap = argparse.ArgumentParser(description="Stage 1 backfill: venue candidates, or append venue rows to review")
    commands = ap.add_subparsers(dest="command", required=True)
    cands = commands.add_parser("candidates", parents=[common], help=f"write {CANDIDATES_FILE}")
    cands.add_argument("--redacted", default=config.REDACTED_DIR, help="where leaking mentions are counted")
    cands.add_argument("--venues", default=config.VENUE_DENYLIST)
    cands.add_argument("--places", default=config.PLACE_DENYLIST)
    cands.add_argument("--common-location", default=config.COMMON_LOCATION)
    cands.add_argument("--out", default=CANDIDATES_FILE)
    fill = commands.add_parser("backfill", parents=[common], help="append uncovered venue spans to the review files")
    fill.add_argument("--reviewdir", default=config.REVIEW_DIR)
    fill.add_argument("--venues", default=config.VENUE_DENYLIST)
    fill.add_argument("--common-datetime", default=config.COMMON_DATETIME)
    fill.add_argument("--common-location", default=config.COMMON_LOCATION)
    fill.add_argument("--dry-run", action="store_true", help="count the rows, write nothing")
    return ap.parse_args(argv)


def run_candidates(args: argparse.Namespace) -> int:
    paths = shard_paths(args.indir)
    if not paths:
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1
    places, capitalized = load_places(args.places)
    listed = {normalize_term(v) for v in load_venues(args.venues) + places + capitalized}
    listed |= {normalize_term(v) for v in load_entries(args.common_location)}
    bodies = (row.get("body") or "" for path in paths for row in read_rows(path))
    found_terms = harvest(bodies, listed)
    redacted = (f"{row.get('subject') or ''}\n{row.get('body') or ''}" for row in iter_shard_rows(args.redacted))
    leaks = leaking(list(found_terms), redacted)
    write_atomic(args.out, render_candidates(found_terms, leaks))
    stats = Stats(
        candidates=len(found_terms),
        label_lines=sum(n for _, n in found_terms.values()),
        leaking_candidates=sum(1 for t in found_terms if leaks[t]),
        leaking_mentions=sum(leaks.values()),
    )
    print(f"-> {args.out}")
    print(stats.render())
    return 0


def run_backfill(args: argparse.Namespace) -> int:
    paths = shard_paths(args.indir)
    if not paths:
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1
    recognizers = [label_recognizer(), *venue_recognizers(load_venues(args.venues))]
    rules = Rules.load(args.common_datetime, args.common_location)
    stats = Stats()
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        review = os.path.join(args.reviewdir, f"{name}.spans.csv")
        if not os.path.exists(review):
            print(f"skipping {name}: no review file yet")
            stats["shards_unreviewed"] += 1
            continue
        existing = read_review(review)
        by_id = collections.defaultdict(list)
        for span in existing:
            by_id[span["email_id"]].append(span)
        spans = [s for row in read_rows(path) for s in new_spans(row, by_id[row.get("id") or ""], recognizers, rules)]
        shard = Stats(f"{s['recognizer']}_{s['decision']}" for s in spans)
        detail = ", ".join(f"{k} {v:,}" for k, v in shard.items())
        print(f"{name}: +{len(spans):,} rows" + (f" ({detail})" if detail else ""))
        stats.update(shard)
        stats["rows"] += len(spans)
        if spans and not args.dry_run:
            write_atomic(review, appended(review, spans))
            stats["shards_written"] += 1
    print(("\ndry run, nothing written" if args.dry_run else f"\n-> {args.reviewdir}") + "\n" + stats.render())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_candidates(args) if args.command == "candidates" else run_backfill(args)


if __name__ == "__main__":
    sys.exit(main())
