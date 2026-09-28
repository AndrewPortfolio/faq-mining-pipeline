# Stage 1 prep: seed the place deny-list that fills in trf's misses on place names
# trf tags a city in one sentence and skips it in the next (27% of mentions of common SoCal cities),
# so a term it called a place in most of its mentions gets flagged in all of them
# Reads trf's calls from the review files and every mention from the extracted text
# Writes data/pii/place_denylist.txt --> holds real place and venue names
# Usage:
#     python src/seed_place_denylist.py           # refuses to overwrite an edited list
#     python src/seed_place_denylist.py --force   # regenerate from scratch

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable, Iterator

from analyze_pii import DEFAULT_INDIR, DEFAULT_OUTDIR, FIELDS, review_spans, write_atomic
from seed_name_denylist import DEFAULT_DICT, english_in_use, load_english
from shared.decisions import COMMON_LOCATION_PATH, normalize_term
from shared.pii_recognizers import PLACELIST_PATH, load_entries
from shared.pipeline import shard_paths


# Configuration

# Only trf's own spans vote: a re-seed after a run would otherwise count this list's own hits
# and push every listed term toward 100%
VOTER = "SpacyRecognizer"

MIN_MENTIONS = 2        # a single mention is already whatever trf made of it; nothing to carry over
MIN_TAGGED_SHARE = 0.5  # measured: drops lions (1 of 18,537) and paris; 0.7 would also drop anaheim (56%)
MIN_CHARS = 4           # "ca", "la", "us" are the region keep-list's job, and short terms match everywhere

HEADER = """\
# Place deny-list for the PII recognizer: every mention of these is flagged LOCATION
# Holds real place and venue names: keep it under data/, which is gitignored
# Seeded from terms trf tagged as a place in at least half their mentions; the comment is tagged/mentions
# [places]       matched in any case
# [capitalized]  also an everyday English word in this inbox ("orange" is a lion color): needs the capital
# Delete any line that isn't a place. The seed script won't overwrite this file without --force
"""


# Counting

def analyzed_rows(indir: str, reviewdir: str) -> Iterator[dict]:
    # only shards with a review file: an unanalyzed email would count every mention as untagged
    for path in shard_paths(indir):
        base = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(reviewdir, f"{base}.spans.csv")):
            with open(path, encoding="utf-8") as fh:
                yield from (json.loads(line) for line in fh if line.strip())


def is_place_term(term: str) -> bool:
    # letters with inner spaces, dots, hyphens, apostrophes ("mt. sac"); digits are the zip rule's job
    # ASCII only: all 80 non-ASCII terms trf tagged here were newsletter phrases or foreign places,
    # and with 2-4 mentions each the share rule can't sort them out
    return (len(term) >= MIN_CHARS and term.isascii() and term[0].isalpha()
            and all(c.isalpha() or c in " .'-" for c in term))


def tally(spans: list[dict], rows: Iterable[dict]) -> tuple[Counter, Counter]:
    # mentions: every appearance in the text; tagged: the ones inside one of trf's LOCATION spans
    tagged_at = defaultdict(list)
    terms = set()
    for span in spans:
        if span["entity_type"] == "LOCATION" and span["recognizer"] == VOTER:
            tagged_at[(span["email_id"], span["field"])].append((int(span["start"]), int(span["end"])))
            term = normalize_term(span["text"])
            if is_place_term(term):
                terms.add(term)
    mentions, tagged = Counter(), Counter()
    if not terms:
        return mentions, tagged
    # longest first, so "huntington beach" wins over a bare "huntington"; any whitespace between words
    alternation = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in sorted(terms, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w@./])(?:{alternation})(?![\w@])", re.IGNORECASE)
    for row in rows:
        for field in FIELDS:
            for m in pattern.finditer(row.get(field) or ""):
                term = normalize_term(m.group(0))
                mentions[term] += 1
                if any(a <= m.start() and m.end() <= b for a, b in tagged_at[(row.get("id"), field)]):
                    tagged[term] += 1
    return mentions, tagged


def select(mentions: Counter, tagged: Counter, common: set[str]) -> list[str]:
    return sorted((t for t, n in mentions.items()
                   if n >= MIN_MENTIONS and tagged[t] / n >= MIN_TAGGED_SHARE and t not in common),
                  key=lambda t: (-mentions[t], t))


def split_entries(terms: list[str], english: set[str]) -> tuple[list[str], list[str]]:
    # a one-word place that's also everyday English here keeps only its capitalized spelling
    places, capitalized = [], []
    for term in terms:
        if " " not in term and term in english:
            capitalized.append(term[:1].upper() + term[1:])
        else:
            places.append(term)
    return places, capitalized


def render(places: list[str], capitalized: list[str], mentions: Counter, tagged: Counter) -> str:
    def lines(entries):
        return "".join(f"{e:<32}# {tagged[e.lower()]}/{mentions[e.lower()]}\n" for e in entries)
    return f"{HEADER}\n[places]\n{lines(places)}\n[capitalized]\n{lines(capitalized)}"


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1 prep: seed the place deny-list")
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--reviewdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--out", default=PLACELIST_PATH)
    ap.add_argument("--dict", default=DEFAULT_DICT, help="word list used to find everyday-English places")
    ap.add_argument("--common-location", default=COMMON_LOCATION_PATH)
    ap.add_argument("--force", action="store_true", help="overwrite an existing (possibly edited) list")
    return ap.parse_args(argv)


def run(args) -> int:
    if os.path.exists(args.out) and not args.force:
        print(f"error: {args.out} exists and may hold your edits; pass --force to regenerate",
              file=sys.stderr)
        return 1
    spans = review_spans(args.reviewdir)
    if not spans:
        print(f"error: no review files in {args.reviewdir}; run analyze_pii.py first", file=sys.stderr)
        return 1

    mentions, tagged = tally(spans, analyzed_rows(args.indir, args.reviewdir))
    common = {normalize_term(w) for w in load_entries(args.common_location)}
    chosen = select(mentions, tagged, common)
    english = english_in_use(load_english(args.dict), analyzed_rows(args.indir, args.reviewdir))
    places, capitalized = split_entries(chosen, english)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_atomic(args.out, render(places, capitalized, mentions, tagged))

    # counts only: the list is real place and venue names, keep them out of terminal scrollback
    recovered = sum(mentions[t] - tagged[t] for t in chosen)
    print(f"{len(places):,} places, {len(capitalized):,} capitalized "
          f"({len(mentions):,} terms trf tagged somewhere; {recovered:,} mentions it skipped now flagged) "
          f"-> {args.out}")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
