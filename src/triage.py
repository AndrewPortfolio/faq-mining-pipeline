# Stage 4: sort Stage 3's clusters into FAQ candidates, clusters to review, and noise to skip
# Reads data/clusters/ (assignments.jsonl, clusters.csv, run.json) and the chunk texts in data/embeddings/clustering/
# Every rule reads a cluster's text, never its number: each Stage 3 run renumbers the clusters
# Nothing is deleted; a skipped cluster is still listed with the reason it was skipped
# Writes to data/faqs/:
#   triage.csv        one row per cluster: its decision, the reason, and the shares behind it
#   candidates.txt    FAQ candidates, then the clusters to review, in report.txt's format; skipped ones one line each
#   read_by_hand.txt  every Vietnamese chunk, wherever it landed, by thread, with the reply it got
#   run.json          the Stage 3 run this triaged and the thresholds used
# Rerun it after every cluster.py run
# Usage:
#     .venv/bin/python src/triage.py

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import re
import sys

from cluster import first_reply, load_chunks, render_report, replies_by_thread, replies_for
from faq_mining import config
from faq_mining.utils.reporting import Stats
from shared.pipeline import write_atomic
from shared.signatures import FORM_MARKERS

# Configuration

# Each skip threshold sits in a wide gap measured on the first full run (lowest flagged vs highest not flagged)
VIETNAMESE_SHARE = 0.5  # 0.94 vs 0.21: nomic-embed-text is mostly English, so it groups these by language
VIETNAMESE_LETTERS = 5  # letters only Vietnamese spelling uses, per chunk
TEMPLATE_SHARE = 0.5  # 0.63 vs 0.33: the website form, and clients filling in the confirmation template
FOOTER_SHARE = 0.3  # 0.57 vs 0.04
# Review, never skip: 107 is HTML around a real topic, and the question share has no gap at all
# (14 clusters at 0%, 64 at <=20%) with some inquiries written as statements
MARKUP_TERMS = 3  # top terms made only of HTML/CSS words
QUESTION_SHARE = 0.10

TRIAGE_COLUMNS = [
    "cluster",
    "decision",
    "reason",
    "n_chunks",
    "n_emails",
    "question_share",
    "vietnamese_share",
    "template_share",
    "footer_share",
    "markup_terms",
    "top_terms",
]

# (decision, reason, heading) in the order candidates.txt lists them
SECTIONS = [
    ("faq", "", "FAQ candidates"),
    ("review", "few_questions", "Review: few questions, mostly thanks and confirmations; keep any whose examples ask"),
    ("review", "markup", "Review: HTML left in the text, which can hide a real topic"),
]

_VIETNAMESE_RE = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.IGNORECASE)
# Notice phrases, not signatures.DISCLAIMER: its bare "confidential" also turns up in real questions
_FOOTER_RE = re.compile(
    r"confidential(?:ity)?(?: and privilege)? (?:notice|warning)|statement of confidentiality|"
    r"intended recipient|the preceding (?:e-?mail )?message|privileged and confidential",
    re.IGNORECASE,
)
# Left by Stage 0's HTML->text fallback: "div style", "font face aptos", "line height"
_MARKUP_WORDS = {
    "div",
    "span",
    "br",
    "font",
    "face",
    "aptos",
    "calibri",
    "style",
    "margin",
    "nbsp",
    "line",
    "height",
    "size",
}


# Rules


def is_vietnamese(text: str) -> bool:
    return len(_VIETNAMESE_RE.findall(text)) >= VIETNAMESE_LETTERS


def signals(texts: list[str], terms: list[str]) -> dict:
    # shares over every chunk in the cluster, copies included, as its size counts them
    n = len(texts) or 1
    return {
        "question_share": sum("?" in t for t in texts) / n,
        "vietnamese_share": sum(is_vietnamese(t) for t in texts) / n,
        "template_share": sum(any(m in t for m in FORM_MARKERS) for t in texts) / n,
        "footer_share": sum(bool(_FOOTER_RE.search(t)) for t in texts) / n,
        "markup_terms": sum(all(w in _MARKUP_WORDS for w in term.split()) for term in terms),
    }


def decide(s: dict) -> tuple[str, str]:
    # -> (decision, reason). A skip needs a strong signal; a weak one only sends the cluster to review
    if s["vietnamese_share"] >= VIETNAMESE_SHARE:
        return "skip", "non_english"
    if s["template_share"] >= TEMPLATE_SHARE:
        return "skip", "template"
    if s["footer_share"] >= FOOTER_SHARE:
        return "skip", "footer"
    if s["markup_terms"] >= MARKUP_TERMS:
        return "review", "markup"
    if s["question_share"] <= QUESTION_SHARE:
        return "review", "few_questions"
    return "faq", ""


# Loading


def load(clusters_dir: str, indir: str) -> tuple[list[dict], list[dict], dict, dict, dict]:
    # -> (cluster summaries with decisions, inbound rows, chunk_id -> cluster, reply threads, Stage 3's run.json)
    inbound, _, outbound, model = load_chunks(indir)
    with open(os.path.join(clusters_dir, "assignments.jsonl"), encoding="utf-8") as fh:
        label = {row["chunk_id"]: row["cluster"] for row in map(json.loads, fh)}
    with open(os.path.join(clusters_dir, "clusters.csv"), encoding="utf-8", newline="") as fh:
        clusters = list(csv.DictReader(fh))
    with open(os.path.join(clusters_dir, "run.json"), encoding="utf-8") as fh:
        stage3 = json.load(fh)
    # re-embedded chunks with old clusters would triage one run's texts under another run's numbers
    if set(label) != {row["chunk_id"] for row in inbound} or stage3.get("model") != model:
        raise ValueError(f"{clusters_dir} wasn't built from {indir} as it is now; rerun cluster.py first")

    members = collections.defaultdict(list)
    for row in inbound:
        members[label[row["chunk_id"]]].append(row["text"])
    by_id = {row["chunk_id"]: row for row in inbound}
    threads = replies_by_thread(outbound)
    summaries = []
    for c in clusters:
        k, terms = int(c["cluster"]), [t for t in c["top_terms"].split("; ") if t]
        s = signals(members[k], terms)
        decision, reason = decide(s)
        reps = [by_id[chunk_id] for chunk_id in c["representatives"].split()]
        summaries.append(
            {
                "cluster": k,
                "n_chunks": int(c["n_chunks"]),
                "n_emails": int(c["n_emails"]),
                "n_threads": int(c["n_threads"]),
                "top_terms": terms,
                "representatives": reps,
                "replies": replies_for(reps, threads),
                "decision": decision,
                "reason": reason,
                **s,
            }
        )
    return summaries, inbound, label, threads, stage3


# Writing


def render_triage(summaries: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=TRIAGE_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for s in summaries:
        writer.writerow(
            {
                **s,
                **{k: round(v, 2) for k, v in s.items() if k.endswith("_share")},
                "top_terms": "; ".join(s["top_terms"]),
            }
        )
    return buf.getvalue()


def render_candidates(summaries: list[dict], header: list[str]) -> str:
    parts = ["\n".join(header)]
    for decision, reason, heading in SECTIONS:
        group = [s for s in summaries if (s["decision"], s["reason"]) == (decision, reason)]
        parts.append(render_report(group, [f"##### {heading}: {len(group)} clusters"]))
    skipped = [s for s in summaries if s["decision"] == "skip"]
    parts.append(
        "\n".join(
            [f"##### Skipped: {len(skipped)} clusters"]
            + [
                f"  cluster {s['cluster']} | {s['n_chunks']:,} chunks | {s['reason']} | {', '.join(s['top_terms'])}"
                for s in skipped
            ]
        )
        + "\n"
    )
    return "\n\n".join(part.rstrip("\n") for part in parts) + "\n"


def render_read_by_hand(inbound: list[dict], label: dict, threads: dict) -> str:
    # whole chunks, untrimmed: the point is to read every one
    chunks = collections.defaultdict(list)
    for row in inbound:
        if is_vietnamese(row["text"]):
            chunks[row["thrid"]].append(row)
    total = sum(len(rows) for rows in chunks.values())
    lines = [
        f"# {total:,} Vietnamese chunks in {len(chunks):,} threads, read by hand: the model groups them "
        f"by language, not topic"
    ]
    for thrid, rows in sorted(chunks.items(), key=lambda kv: min(r.get("date") or "" for r in kv[1])):
        rows.sort(key=lambda r: (r.get("date") or "", r["chunk_index"]))
        lines += ["", f"=== thread {thrid}"]
        shown = set()
        for row in rows:
            lines.append(
                f"  [{row.get('date') or '?'} | cluster {label[row['chunk_id']]}] {' '.join(row['text'].split())}"
            )
            reply = first_reply(row, threads)
            if reply and reply["chunk_id"] not in shown:  # an email's later chunks share its reply
                shown.add(reply["chunk_id"])
                lines.append(f"  reply: {' '.join(reply['text'].split())}")
    return "\n".join(lines) + "\n"


# CLI


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: sort clusters into FAQ candidates, review, and noise")
    ap.add_argument("--clusters", default=config.CLUSTERS_DIR, help="Stage 3's output")
    ap.add_argument("--indir", default=config.CLUSTERING_EMBEDDINGS_DIR, help="the clustering vectors Stage 3 read")
    ap.add_argument("--outdir", default=config.FAQS_DIR)
    return ap.parse_args(argv)


def run(args) -> int:
    try:
        summaries, inbound, label, threads, stage3 = load(args.clusters, args.indir)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    stats = Stats(f"{s['decision']} {s['reason']}".strip() for s in summaries)
    counts = collections.Counter(s["decision"] for s in summaries)
    header = [
        f"# {counts['faq']} FAQ candidates, {counts['review']} to review, {counts['skip']} skipped "
        f"of {len(summaries)} clusters in {args.clusters}",
        f"# Stage 3: min_cluster_size {stage3.get('min_cluster_size')}, method {stage3.get('method')}, "
        f"seed {stage3.get('seed')} | {stage3.get('model')}",
    ]
    thresholds = {
        "vietnamese_share": VIETNAMESE_SHARE,
        "vietnamese_letters": VIETNAMESE_LETTERS,
        "template_share": TEMPLATE_SHARE,
        "footer_share": FOOTER_SHARE,
        "markup_terms": MARKUP_TERMS,
        "question_share": QUESTION_SHARE,
    }

    os.makedirs(args.outdir, exist_ok=True)
    write_atomic(os.path.join(args.outdir, "triage.csv"), render_triage(summaries))
    write_atomic(os.path.join(args.outdir, "candidates.txt"), render_candidates(summaries, header))
    write_atomic(os.path.join(args.outdir, "read_by_hand.txt"), render_read_by_hand(inbound, label, threads))
    write_atomic(
        os.path.join(args.outdir, "run.json"),
        json.dumps(
            {"clusters": args.clusters, "stage3": stage3, "thresholds": thresholds, "decisions": dict(stats)}, indent=2
        )
        + "\n",
    )

    print(f"-> {args.outdir}")
    print(stats.render())
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
