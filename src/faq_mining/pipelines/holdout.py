# Test holdout: list the held-out threads' client questions to label by hand, and check none leaked downstream
# Which threads are held out is faq_mining/utils/holdout_split.py's hash of the thread id. Stage 2 never embeds them, so
# they're only in data/redacted/, which both commands read:
#   queries  writes data/holdout/:
#              labels.csv  each held-out thread's first client email in a random but fixed order, lang and kind
#                          filled in, faq left blank to label by hand. The top ~300 are the test, the rest the reserve
#              run.json    the seed and share, what each filter dropped, and the near-duplicates it found
#   check    counts held-out email and thread ids in the later stages' outputs, never their text; exit 1 on any
# labels.csv is written once: a rewrite would throw the labels away
# Usage:
#     PYTHONPATH=src .venv/bin/python -m faq_mining.pipelines.holdout check            # after every re-embed
#     PYTHONPATH=src .venv/bin/python -m faq_mining.pipelines.holdout queries          # once the FAQ exists
#     PYTHONPATH=src .venv/bin/python -m faq_mining.pipelines.holdout queries --force  # rewrite, before any labels

import argparse
import collections
import csv
import io
import json
import os
import re
import sys

from cluster import _when
from embed import cut_quoted, has_content, pii_hits
from faq_mining import config
from faq_mining.storage.shards import iter_shard_rows, shard_paths, write_atomic
from faq_mining.utils.holdout_split import SEED, SHARE, is_held_out, rank
from faq_mining.utils.reporting import Stats
from triage import is_vietnamese

# Configuration

LABELS_FILE = "labels.csv"
LABEL_COLUMNS = ["id", "thrid", "lang", "kind", "faq", "text"]

# Where held-out ids must never turn up
CHECK_DIRS = [config.EMBEDDINGS_DIR, config.CLUSTERS_DIR, config.FAQS_DIR]

SHINGLE_WORDS = 5  # runs of 5 words rarely repeat by chance
BOILERPLATE_EMAILS = 20  # a run in more emails than this is a form label or a signature, not anyone's own words
NEAR_DUP_SHARE = 0.5  # of a query's own runs that one email in another thread holds

# A reply's subject, also under a forward ("Fwd: Re:"). A bare "Fwd:" opens a thread: a client forwarding a venue's
# email to ask about it
_REPLY_RE = re.compile(r"^\s*(?:fwd?\s*:\s*)*re\s*:", re.IGNORECASE)
# Out-of-office notes and bounces. Most are replies anyway; these catch the ones that open a thread
_AUTO_SUBJECT_RE = re.compile(
    r"^\s*(?:automatic reply|auto[- ]?(?:reply|response)|autoreply|out of (?:the )?office|undeliverable|"
    r"delivery status|returned mail|mail delivery)",
    re.IGNORECASE,
)
_AUTO_BODY_RE = re.compile(
    r"\bout of (?:the )?office\b|\bautomatic reply\b|\bauto[- ]?(?:reply|response)\b|"
    r"\blimited access to (?:my )?e-?mail\b|\bthis is an automated\b",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[A-Z_]+>")
_WORD_RE = re.compile(r"[^\W\d_]+")  # letters only, accented ones included
# Email ids are 16 hex digits and thread ids 19 decimal ones
_ID_RE = re.compile(r"\b[0-9a-f]{16,20}\b")


# Rules


def is_reply(subject: str) -> bool:
    return bool(_REPLY_RE.match(subject))


def is_auto_reply(subject: str, text: str) -> bool:
    return bool(_AUTO_SUBJECT_RE.match(subject) or _AUTO_BODY_RE.search(text))


def pick_query(emails: list[dict], own: set[str]) -> tuple[dict | None, str]:
    # -> (a held-out thread's test query, "") or (None, why it has none). The query is the thread's first client
    # email: inbound and not ATLD's own draft or note. The filter is coarse on purpose: every query is labeled by hand
    if not all(_when(email) for email in emails):
        return None, "undated"  # nobody can tell who wrote first
    emails = sorted(emails, key=lambda email: (_when(email), email["id"]))
    first = next((e for e in emails if e.get("direction") == "inbound" and e.get("sender") not in own), None)
    if first is None:
        return None, "no_client_email"
    subject, text = first.get("subject") or "", cut_quoted(first.get("body") or "")
    if is_reply(subject):
        return None, "reply"
    if is_auto_reply(subject, text):
        return None, "auto_reply"
    if pii_hits(first):
        return None, "tripwire"  # the candidate models embed every query
    if not has_content(text):
        return None, "no_content"
    # a follow-up came after mail ATLD sent, so it may lean on context it doesn't hold, and is scored apart.
    # A draft before it never reached the client, so it doesn't count
    atld_wrote_first = any(email.get("direction") == "outbound" for email in emails[: emails.index(first)])
    return {
        "id": first["id"],
        "thrid": first["thrid"],
        "lang": "vi" if is_vietnamese(text) else "en",
        "kind": "follow_up" if atld_wrote_first else "opener",
        "text": text,
    }, ""


# Loading


def load_threads(indir: str) -> tuple[dict[str, list[dict]], set[str]]:
    # -> (every thread's emails, ATLD's sender hashes). ATLD is whoever sent the Sent mail, as in embed.py
    rows = list(iter_shard_rows(indir))
    own = {row.get("sender") for row in rows if row.get("direction") == "outbound"} - {None, ""}
    threads = collections.defaultdict(list)
    for row in rows:
        threads[row.get("thrid") or ""].append(row)
    return dict(threads), own


def held_out_ids(threads: dict[str, list[dict]]) -> set[str]:
    # every held-out thread id and the id of every email in one
    ids = set()
    for thrid, emails in threads.items():
        if is_held_out(thrid):
            ids.add(thrid)
            ids.update(email["id"] for email in emails)
    return ids


# Near-duplicates


def shingles(text: str) -> set[str]:
    # every run of SHINGLE_WORDS words, lowercased, with Stage 1's tags taken out
    words = _WORD_RE.findall(_TAG_RE.sub(" ", text).lower())
    return {" ".join(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


def near_duplicates(queries: list[dict], threads: dict[str, list[dict]]) -> dict[str, tuple[str, float]]:
    # query id -> (the email in another thread holding most of its own words, the share it holds), for each query
    # at NEAR_DUP_SHARE or more. A client re-asking in a new thread leaves a copy of the test where a model can see
    # it, and exact-text dedupe misses it. Runs in more than BOILERPLATE_EMAILS emails don't count, or every form
    # message would be a copy of every other
    runs = {e["id"]: shingles(cut_quoted(e.get("body") or "")) for emails in threads.values() for e in emails}
    seen_in = collections.Counter(run for found in runs.values() for run in found)
    holders = collections.defaultdict(list)
    for email_id, found in runs.items():
        for run in found:
            if seen_in[run] <= BOILERPLATE_EMAILS:
                holders[run].append(email_id)
    thread_of = {email["id"]: thrid for thrid, emails in threads.items() for email in emails}
    found = {}
    for query in queries:
        rare = [run for run in runs[query["id"]] if seen_in[run] <= BOILERPLATE_EMAILS]
        shared = collections.Counter(
            email_id for run in rare for email_id in holders[run] if thread_of[email_id] != query["thrid"]
        )
        if not shared:
            continue
        twin, n = min(shared.items(), key=lambda kv: (-kv[1], kv[0]))  # the id breaks a tie, so reruns agree
        if n / len(rare) >= NEAR_DUP_SHARE:
            found[query["id"]] = (twin, round(n / len(rare), 2))
    return found


# Writing


def render_labels(queries: list[dict]) -> str:
    # faq is filled in by hand: an FAQ entry, none (no FAQ answers it) or skip (not a real question)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=LABEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({**query, "faq": "", "text": " ".join(query["text"].split())} for query in queries)
    return buf.getvalue()


def file_leaks(path: str, ids: set[str]) -> int:
    # held-out ids in one file. Anything shaped like an id is looked up, so no file's layout has to be known and
    # outputs that don't exist yet are covered too
    with open(path, encoding="utf-8", errors="replace") as fh:
        return sum(token in ids for token in _ID_RE.findall(fh.read()))


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--indir", default=config.REDACTED_DIR, help="Stage 1's output, where held-out threads are")
    ap = argparse.ArgumentParser(description="Test holdout: list held-out questions to label, or check for leaks")
    commands = ap.add_subparsers(dest="command", required=True)
    queries = commands.add_parser("queries", parents=[common], help="write labels.csv, once the FAQ exists")
    queries.add_argument("--outdir", default=config.HOLDOUT_DIR)
    queries.add_argument("--force", action="store_true", help="rewrite labels.csv; only safe before any labeling")
    check = commands.add_parser("check", parents=[common], help="look for held-out ids in the pipeline's outputs")
    check.add_argument("dirs", nargs="*", default=CHECK_DIRS, help=f"default: {' '.join(CHECK_DIRS)}")
    return ap.parse_args(argv)


def run_queries(args: argparse.Namespace) -> int:
    target = os.path.join(args.outdir, LABELS_FILE)
    if os.path.exists(target) and not args.force:
        print(
            f"error: {target} is already written and may hold labels; --force rewrites it, which is only safe "
            f"before labeling starts",
            file=sys.stderr,
        )
        return 1
    if not shard_paths(args.indir):
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1
    threads, own = load_threads(args.indir)

    stats, queries = Stats(), []
    for thrid, emails in threads.items():
        if not is_held_out(thrid):
            continue
        stats["held_out_threads"] += 1
        stats["held_out_emails"] += len(emails)
        query, reason = pick_query(emails, own)
        if query:
            queries.append(query)
        else:
            stats[reason] += 1
    dups = near_duplicates(queries, threads)
    stats["near_duplicate"] = len(dups)
    queries = sorted((query for query in queries if query["id"] not in dups), key=lambda query: rank(query["thrid"]))
    stats["rows"] = len(queries)
    stats.update(query["kind"] for query in queries)
    stats.update(query["lang"] for query in queries)
    summary = {
        "seed": SEED,
        "share": SHARE,
        "indir": args.indir,
        **stats,
        "near_duplicates": [{"id": qid, "twin": twin, "share": share} for qid, (twin, share) in sorted(dups.items())],
    }

    os.makedirs(args.outdir, exist_ok=True)
    write_atomic(os.path.join(args.outdir, "run.json"), json.dumps(summary, indent=2) + "\n")
    write_atomic(target, render_labels(queries))  # last: this file existing is what marks it written
    print(f"-> {args.outdir}")
    print(stats.render())
    return 0


def run_check(args: argparse.Namespace) -> int:
    if not shard_paths(args.indir):
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1
    threads, _ = load_threads(args.indir)
    ids = held_out_ids(threads)
    held = sum(is_held_out(thrid) for thrid in threads)
    print(f"{held:,} held-out threads ({len(ids) - held:,} emails), looked for in {', '.join(args.dirs)}")
    for top in args.dirs:
        if not os.path.isdir(top):
            print(f"  {top}: missing, skipped")
    paths = sorted(os.path.join(root, name) for top in args.dirs for root, _, names in os.walk(top) for name in names)
    counts = {path: file_leaks(path, ids) for path in paths}
    leaks = {path: n for path, n in counts.items() if n}
    if not leaks:
        print(f"ok: none in {len(paths):,} files")
        return 0
    width = max(len(path) for path in leaks)
    for path, n in leaks.items():
        print(f"  {path:<{width}}  {n:,}")
    print(
        f"error: held-out ids in {len(leaks):,} of {len(paths):,} files; rebuild them from a Stage 2 run that skips "
        f"the holdout (embed.py --force)",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_queries(args) if args.command == "queries" else run_check(args)


if __name__ == "__main__":
    sys.exit(main())
