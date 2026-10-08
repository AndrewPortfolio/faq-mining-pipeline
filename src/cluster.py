# Stage 3: group the inbound (client) chunks into question clusters with UMAP + HDBSCAN
# Reads Stage 2's clustering vectors (data/embeddings/clustering/) and writes to data/clusters/:
#   assignments.jsonl  each inbound chunk's cluster (-1 = noise, a one-off question) and membership strength
#   clusters.csv       one row per cluster: sizes, top terms, its most typical chunks
#   report.txt         every cluster, biggest first, with its typical messages and how they were answered
#   run.json           the settings and input behind these files
# Only inbound chunks are clustered; your replies are looked up through the thread instead
# Usage:
#     .venv/bin/python src/cluster.py --sweep                  # compare HDBSCAN settings, writes nothing
#     .venv/bin/python src/cluster.py                          # cluster with the defaults below
#     .venv/bin/python src/cluster.py --min-cluster-size 25    # fewer, broader clusters

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from shared.pipeline import Stats, read_shard, shard_paths, write_atomic


# Configuration

DEFAULT_INDIR = "data/embeddings/clustering"
DEFAULT_OUTDIR = "data/clusters"

# HDBSCAN's density estimates break down in 768 dimensions, so UMAP squeezes the vectors into 5 that keep
# each chunk's neighbours. min_dist 0 packs similar chunks tightly, which is what density clustering wants
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST = 0.0

MIN_CLUSTER_SIZE = 15  # smallest group of chunks that counts as an FAQ
MIN_SAMPLES = 5        # lower means fewer chunks written off as noise
METHOD = "eom"         # "leaf" splits into more, finer clusters
SEED = 42              # same seed, same clusters; it costs UMAP its multi-threading

TOP_TERMS = 8
REPRESENTATIVES = 5
REPLIES = 2
TRIM_CHARS = 300

# --sweep tries every combination on one UMAP run
SWEEP_SIZES = (10, 15, 25, 40)
SWEEP_SAMPLES = (None, 5)  # None is HDBSCAN's default: the same as min_cluster_size
SWEEP_METHODS = ("eom", "leaf")

CLUSTER_COLUMNS = ["cluster", "n_chunks", "n_emails", "n_threads", "top_terms", "representatives"]

_TAG_RE = re.compile(r"<[A-Z_]+>")


# Loading

def load_chunks(indir: str) -> tuple[list[dict], np.ndarray, list[dict], str]:
    # -> (inbound rows, their vectors, outbound rows, model). Vectors from another model or task
    # aren't comparable, so a mix is refused rather than clustered
    inbound, vectors, outbound, kinds = [], [], [], set()
    for path in shard_paths(indir):
        for row in read_shard(path):
            kinds.add((row.get("task"), row.get("model")))
            vector = row.pop("embedding")
            if row.get("direction") == "inbound":
                inbound.append(row)
                vectors.append(np.asarray(vector, dtype=np.float32))
            else:
                outbound.append(row)
    if len(kinds) != 1 or next(iter(kinds))[0] != "clustering":
        raise ValueError(f"{indir} holds {sorted(kinds, key=str)}; Stage 3 needs one model's clustering vectors")
    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
    return inbound, matrix, outbound, next(iter(kinds))[1]


def dedupe(rows: list[dict]) -> tuple[list[int], np.ndarray]:
    # -> (the first row holding each distinct text, and for every row the position of its text in that list)
    # Identical chunks sit on top of each other and distort UMAP's neighbour graph
    first, slot, where = [], {}, []
    for i, row in enumerate(rows):
        key = " ".join(row["text"].lower().split())
        if key not in slot:
            slot[key] = len(first)
            first.append(i)
        where.append(slot[key])
    return first, np.asarray(where, dtype=int)


# UMAP + HDBSCAN

def reduce(vectors: np.ndarray, seed: int) -> np.ndarray:
    # imported here: umap compiles its numba code on import (~30 s the first time), and tests swap this out
    import umap
    reducer = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, n_components=UMAP_COMPONENTS, min_dist=UMAP_MIN_DIST,
                        metric="cosine", random_state=seed, n_jobs=1)
    return reducer.fit_transform(vectors)


def find_clusters(points: np.ndarray, min_cluster_size: int, min_samples: int | None,
                  method: str) -> tuple[np.ndarray, np.ndarray]:
    # -> (labels, membership strength). -1 is noise: a one-off question, never forced into a cluster
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    cluster_selection_method=method, copy=True).fit(points)
    return model.labels_, model.probabilities_


def by_size(labels: np.ndarray) -> dict[int, int]:
    # old label -> new label, so cluster 0 is the biggest; noise stays -1
    sizes = collections.Counter(int(label) for label in labels if label >= 0)
    ranked = sorted(sizes, key=lambda label: (-sizes[label], label))
    return {**{old: new for new, old in enumerate(ranked)}, -1: -1}


def sweep(points: np.ndarray, where: np.ndarray) -> list[dict]:
    # every setting on the same UMAP output; counts are per chunk, copies included
    results = []
    for method in SWEEP_METHODS:
        for size in SWEEP_SIZES:
            for samples in SWEEP_SAMPLES:
                labels = find_clusters(points, size, samples, method)[0][where]
                found = labels[labels >= 0]
                results.append({"method": method, "min_cluster_size": size, "min_samples": samples,
                                "clusters": len(set(found.tolist())),
                                "noise_pct": 100 * float((labels < 0).mean()),
                                "largest_pct": 100 * np.bincount(found).max() / len(labels) if len(found) else 0.0})
    return results


# Describing clusters

def top_terms(documents: list[str], n: int = TOP_TERMS) -> list[list[str]]:
    # c-TF-IDF: each cluster's text as one document, so a term ranks high when it's common in its
    # cluster and rare across the rest. Greetings and sign-offs show up everywhere and sink
    if not documents:
        return []
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    try:
        counts = vectorizer.fit_transform(_TAG_RE.sub(" ", doc) for doc in documents).astype(float)
    except ValueError:  # nothing left but stop words and tags
        return [[] for _ in documents]
    words = np.asarray(counts.sum(axis=1)).ravel()
    idf = np.log(1 + words.mean() / np.asarray(counts.sum(axis=0)).ravel())
    scores = normalize(counts, norm="l1").multiply(idf).tocsr()
    terms = vectorizer.get_feature_names_out()
    ranked = []
    for i in range(scores.shape[0]):
        start, end = scores.indptr[i], scores.indptr[i + 1]
        best = np.argsort(-scores.data[start:end], kind="stable")[:n]
        ranked.append([str(terms[j]) for j in scores.indices[start:end][best]])
    return ranked


def representatives(vectors: np.ndarray, n: int = REPRESENTATIVES) -> np.ndarray:
    # positions of the n vectors closest to their mean direction (unit vectors, so a dot product ranks by cosine)
    return np.argsort(-(vectors @ vectors.mean(axis=0)), kind="stable")[:n]


def _when(row: dict) -> datetime | None:
    # Stage 0 wrote ISO dates with their own offsets; one without an offset is taken as UTC so all compare
    try:
        when = datetime.fromisoformat(row.get("date") or "")
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def replies_by_thread(outbound: list[dict]) -> dict[str, list[tuple[datetime, dict]]]:
    # each thread's outbound chunks in the order they were sent
    threads = collections.defaultdict(list)
    for row in outbound:
        when = _when(row)
        if when:
            threads[row.get("thrid")].append((when, row))
    for sent in threads.values():
        sent.sort(key=lambda pair: (pair[0], pair[1]["chunk_index"]))
    return threads


def first_reply(row: dict, threads: dict) -> dict | None:
    # the first outbound chunk in the same thread sent after this message
    asked = _when(row)
    if asked is None:
        return None
    return next((reply for when, reply in threads.get(row.get("thrid"), ()) if when > asked), None)


def describe(inbound: list[dict], vectors: np.ndarray, first: list[int], labels: np.ndarray,
             labels_distinct: np.ndarray, threads: dict) -> list[dict]:
    # one summary per cluster, biggest first
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    members = [np.flatnonzero(labels == k) for k in range(n_clusters)]
    terms = top_terms([" ".join(inbound[i]["text"] for i in rows) for rows in members])
    first = np.asarray(first)
    summaries = []
    for k, rows in enumerate(members):
        distinct = first[labels_distinct == k]  # one row per distinct text, so no copy shows twice
        reps = [inbound[i] for i in distinct[representatives(vectors[distinct])]]
        replies = []
        for rep in reps:
            reply = first_reply(rep, threads)
            if reply and reply["text"] not in {r["text"] for r in replies}:
                replies.append(reply)
        summaries.append({
            "cluster": k,
            "n_chunks": len(rows),
            "n_emails": len({inbound[i]["id"] for i in rows}),
            "n_threads": len({inbound[i]["thrid"] for i in rows}),
            "top_terms": terms[k],
            "representatives": reps,
            "replies": replies[:REPLIES],
        })
    return summaries


# Writing

def _trim(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= TRIM_CHARS else flat[:TRIM_CHARS - 1] + "…"


def render_report(summaries: list[dict], header: list[str]) -> str:
    lines = list(header)
    for s in summaries:
        lines += ["", f"=== cluster {s['cluster']} | {s['n_chunks']:,} chunks | {s['n_emails']:,} emails | "
                      f"{s['n_threads']:,} threads | {', '.join(s['top_terms'])}"]
        lines += [f"  {i}. {_trim(row['text'])}" for i, row in enumerate(s["representatives"], 1)]
        lines += [f"  reply: {_trim(row['text'])}" for row in s["replies"]]
    return "\n".join(lines) + "\n"


def render_clusters(summaries: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CLUSTER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for s in summaries:
        writer.writerow({"cluster": s["cluster"], "n_chunks": s["n_chunks"], "n_emails": s["n_emails"],
                         "n_threads": s["n_threads"], "top_terms": "; ".join(s["top_terms"]),
                         "representatives": " ".join(row["chunk_id"] for row in s["representatives"])})
    return buf.getvalue()


def render_sweep(results: list[dict]) -> str:
    lines = [f"{'method':<8}{'min_size':>9}{'min_samples':>13}{'clusters':>10}{'noise':>8}{'largest':>9}"]
    for r in results:
        samples = "=size" if r["min_samples"] is None else r["min_samples"]
        lines.append(f"{r['method']:<8}{r['min_cluster_size']:>9}{samples:>13}{r['clusters']:>10}"
                     f"{r['noise_pct']:>7.1f}%{r['largest_pct']:>8.1f}%")
    return "\n".join(lines)


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3: cluster inbound chunks into FAQ candidates")
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                    help="0 means the same as --min-cluster-size")
    ap.add_argument("--method", choices=("eom", "leaf"), default=METHOD)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sweep", action="store_true",
                    help="compare HDBSCAN settings on one UMAP run; writes nothing")
    return ap.parse_args(argv)


def run(args) -> int:
    if not shard_paths(args.indir):
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1
    started = time.time()
    try:
        inbound, vectors, outbound, model = load_chunks(args.indir)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if not inbound:
        print(f"error: no inbound chunks in {args.indir}", file=sys.stderr)
        return 1
    first, where = dedupe(inbound)
    print(f"loaded {len(inbound):,} inbound chunks ({len(first):,} distinct texts) and "
          f"{len(outbound):,} outbound in {time.time() - started:.0f}s")

    started = time.time()
    points = reduce(vectors[first], args.seed)
    print(f"UMAP: {vectors.shape[1]} -> {points.shape[1]} dimensions in {time.time() - started:.0f}s")
    if args.sweep:
        print(render_sweep(sweep(points, where)))
        return 0

    min_samples = args.min_samples or None
    labels_distinct, strength_distinct = find_clusters(points, args.min_cluster_size, min_samples, args.method)
    order = by_size(labels_distinct[where])
    labels_distinct = np.array([order[int(label)] for label in labels_distinct])
    labels, strength = labels_distinct[where], strength_distinct[where]
    summaries = describe(inbound, vectors, first, labels, labels_distinct, replies_by_thread(outbound))

    stats = Stats()
    stats["inbound_chunks"] = len(inbound)
    stats["distinct_texts"] = len(first)
    stats["clusters"] = len(summaries)
    stats["noise_chunks"] = int((labels < 0).sum())
    noise_pct = 100 * stats["noise_chunks"] / len(inbound)
    settings = {"min_cluster_size": args.min_cluster_size, "min_samples": min_samples, "method": args.method,
                "seed": args.seed, "umap": {"n_neighbors": UMAP_NEIGHBORS, "n_components": UMAP_COMPONENTS,
                                            "min_dist": UMAP_MIN_DIST, "metric": "cosine"}}
    header = [f"# {len(summaries)} clusters from {len(inbound):,} inbound chunks; {stats['noise_chunks']:,} "
              f"({noise_pct:.1f}%) are noise (one-off questions)",
              f"# min_cluster_size {args.min_cluster_size}, min_samples {min_samples or 'same'}, "
              f"method {args.method}, seed {args.seed} | input {args.indir} ({model})"]

    os.makedirs(args.outdir, exist_ok=True)
    write_atomic(os.path.join(args.outdir, "assignments.jsonl"), "".join(
        json.dumps({"chunk_id": row["chunk_id"], "id": row["id"], "thrid": row["thrid"],
                    "cluster": int(label), "probability": round(float(p), 3)}) + "\n"
        for row, label, p in zip(inbound, labels, strength)))
    write_atomic(os.path.join(args.outdir, "clusters.csv"), render_clusters(summaries))
    write_atomic(os.path.join(args.outdir, "report.txt"), render_report(summaries, header))
    write_atomic(os.path.join(args.outdir, "run.json"), json.dumps(
        {**settings, "input": args.indir, "model": model, **stats, "noise_pct": round(noise_pct, 1)},
        indent=2) + "\n")

    print(f"\n-> {args.outdir}")
    print(stats.render())
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
