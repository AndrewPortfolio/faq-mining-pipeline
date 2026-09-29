# Stage 1: sample for a confidence read before anything is redacted for real
# Uses the whole email, not just the span: PII the recognizers missed never becomes a span

# One email per top PERSON term (the heaviest single decisions), the rest drawn uniformly
# Read for two things: PII left unmarked or marked keep, and FAQ content marked REDACT

# Usage:
#     venv/bin/python src/pii_sample.py            # -> data/review/sample.txt
#     venv/bin/python src/pii_sample.py --seed 7   # a different draw

from __future__ import annotations

import argparse
import collections
import os
import random
import sys

from analyze_pii import DEFAULT_INDIR, DEFAULT_OUTDIR, FIELDS, read_shard, review_spans, write_atomic
from shared.decisions import normalize_term
from shared.pipeline import shard_paths


# Configuration

DEFAULT_SIZE = 300
DEFAULT_TOP_PERSON = 20
DEFAULT_SEED = 0  # fixed, so a re-generated sample is the same one already read


# Drawing

def analyzed_emails(indir: str, reviewdir: str) -> list[dict]:
    # only shards with a review file: an unanalyzed email would read as "nothing was caught"
    emails = []
    for path in shard_paths(indir):
        base = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(reviewdir, f"{base}.spans.csv")):
            emails.extend(read_shard(path))
    return emails


def redacted_person(spans: list[dict]) -> list[dict]:
    # kept ambiguous words ("The", "To", An) are the most frequent PERSON text, but not names worth a slot
    return [s for s in spans if s["entity_type"] == "PERSON" and s["decision"] == "redact"]


def top_person_terms(spans: list[dict], n: int) -> list[str]:
    counts = collections.Counter(normalize_term(s["text"]) for s in redacted_person(spans))
    counts.pop("", None)
    return [term for term, _ in counts.most_common(n)]


def draw(emails: list[dict], spans: list[dict], size: int, top_person: int,
         seed: int) -> list[tuple[dict, str]]:
    rng = random.Random(seed)
    by_term = collections.defaultdict(list)
    for span in redacted_person(spans):
        by_term[normalize_term(span["text"])].append(span["email_id"])

    ids = {e.get("id"): e for e in emails}
    chosen: dict[str, str] = {}
    for rank, term in enumerate(top_person_terms(spans, top_person), 1):
        options = sorted({i for i in by_term[term] if i in ids and i not in chosen})
        if options and len(chosen) < size:
            chosen[rng.choice(options)] = f"top PERSON #{rank}: {term}"
    rest = sorted(i for i in ids if i not in chosen)
    for email_id in rng.sample(rest, min(len(rest), size - len(chosen))):
        chosen[email_id] = "random"
    return [(ids[i], reason) for i, reason in chosen.items()]


# Rendering

def clusters(spans: list[dict]) -> list[dict]:
    # Overlapping spans (ORG "Desert Hills Outlets" over LOCATION "Desert Hills") become one marker,
    # redacted if any member is, because that's what apply_redactions will do to the text
    out: list[dict] = []
    for span in sorted(spans, key=lambda s: (int(s["start"]), -int(s["end"]))):
        start, end = int(span["start"]), int(span["end"])
        if out and start < out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], end)
            out[-1]["spans"].append(span)
        else:
            out.append({"start": start, "end": end, "spans": [span]})
    return out


def marker(text: str, cluster: dict) -> str:
    members = cluster["spans"]
    shown = text[cluster["start"]:cluster["end"]]
    if any(s["decision"] == "redact" for s in members):
        return f"[[REDACT {'/'.join(dict.fromkeys(s['entity_type'] for s in members))}: {shown}]]"
    why = "/".join(dict.fromkeys(f"{s['entity_type']} {s.get('rule') or '?'}" for s in members))
    return f"[[keep {why}: {shown}]]"


def render(email: dict, spans: list[dict], n: int, total: int, reason: str) -> str:
    redact = sum(1 for s in spans if s["decision"] == "redact")
    lines = [f"=== [{n}/{total}] {reason} | {email.get('id')} | thrid {email.get('thrid')} | "
             f"{len(spans)} spans ({redact} redact, {len(spans) - redact} keep)"]
    for field in FIELDS:
        text = email.get(field) or ""
        for cluster in reversed(clusters([s for s in spans if s["field"] == field])):
            text = f"{text[:cluster['start']]}{marker(text, cluster)}{text[cluster['end']:]}"
        lines.append(f"{field}: {text}")
    return "\n".join(lines) + "\n"


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stratified sample of analyzed emails for a confidence read")
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--reviewdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--out", default=None, help="default: <reviewdir>/sample.txt")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--top-person", type=int, default=DEFAULT_TOP_PERSON)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args(argv)


def run(args) -> int:
    spans = review_spans(args.reviewdir)
    emails = analyzed_emails(args.indir, args.reviewdir)
    if not emails:
        print(f"error: no analyzed shards in {args.indir} / {args.reviewdir}", file=sys.stderr)
        return 1
    by_email = collections.defaultdict(list)
    for span in spans:
        by_email[span["email_id"]].append(span)

    picked = draw(emails, spans, args.size, args.top_person, args.seed)
    out = args.out or os.path.join(args.reviewdir, "sample.txt")
    write_atomic(out, "\n".join(render(e, by_email[e.get("id")], n, len(picked), reason)
                                for n, (e, reason) in enumerate(picked, 1)))
    top = sum(1 for _, reason in picked if reason != "random")
    print(f"{len(picked)} emails ({top} top-PERSON, {len(picked) - top} random, seed {args.seed}) -> {out}")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
