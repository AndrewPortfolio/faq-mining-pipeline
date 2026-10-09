#Tests for cluster (Stage 3)

#Fixtures are synthetic chunk vectors and texts written to tmp_path --> no real data, nothing in git
#Most tests swap UMAP for a quick SVD projection so numba never compiles; one test runs the real thing

import csv
import io
import json
import re

import numpy as np

import cluster
from cluster import (by_size, dedupe, first_reply, main, render_page, replies_by_thread,
                     representatives, top_terms)

MODEL = "nomic-embed-text@0a109f422b47"
TOPICS = {
    "deposit": "Is the deposit refundable if we move the date? How much deposit holds it?",
    "parking": "Is parking free at the venue? Where should the drummers find parking?",
    "newyear": "Are you free for lunar new year? We want a lion dance at our lunar new year opening.",
    "performance": "What does the performance include?",
    "cost": "How much does it cost for the instrument team too?",
}
PER_TOPIC = 40
COPIES = 3   # extra copies of the first deposit chunk


def _row(n, text, vector, *, direction="inbound", thrid=None, date="2025-01-01T10:00:00+00:00",
         chunk_index=0, task="clustering"):
    return {"chunk_id": f"{n:016x}-{chunk_index}", "id": f"{n:016x}", "thrid": thrid or f"t{n}",
            "date": date, "direction": direction, "subject": "Question", "chunk_index": chunk_index,
            "n_chunks": 1, "n_words": len(text.split()), "text": text, "task": task, "model": MODEL,
            "embedding": [float(x) for x in vector]}


def _unit(v):
    return v / np.linalg.norm(v)


def _corpus(seed=0, dims=32):
    # 5 tight topics, their copies, scattered one-offs, and a later reply in each topic chunk's thread
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(TOPICS), dims))
    rows, n = [], 0
    for t, text in enumerate(TOPICS.values()):
        for i in range(PER_TOPIC):
            n += 1
            rows.append(_row(n, f"Hi <PERSON>, {text} Thanks, note {i}", _unit(centres[t] + rng.normal(scale=0.4, size=dims))))
            rows.append(_row(n + 5000, f"Happy to help with that, answer {n}.", _unit(rng.normal(size=dims)),
                             direction="outbound", thrid=f"t{n}", date="2025-01-01T12:00:00+00:00"))
    for c in range(COPIES):
        rows.append(_row(9000 + c, rows[0]["text"], np.array(rows[0]["embedding"])))
    for i in range(20):
        rows.append(_row(7000 + i, f"Random one-off message {i}", _unit(rng.normal(size=dims))))
    return rows


def _write(tmp_path, rows):
    indir = tmp_path / "embeddings"
    indir.mkdir(exist_ok=True)
    half = len(rows) // 2
    for k, part in enumerate((rows[:half], rows[half:])):
        (indir / f"emails-{k:05d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")
    return indir


def _svd_reduce(vectors, seed, n_components=5, min_dist=0.0):
    # stand-in for UMAP: the top principal components, deterministic and instant
    centred = vectors - vectors.mean(axis=0)
    return centred @ np.linalg.svd(centred, full_matrices=False)[2][:n_components].T


def _page_data(page):
    # the JSON the page embeds, read back the way the browser reads it
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1))


def _run(monkeypatch, tmp_path, *extra, rows=None, real_umap=False, outdir="clusters"):
    if not real_umap:
        monkeypatch.setattr(cluster, "reduce", _svd_reduce)
    indir = _write(tmp_path, rows if rows is not None else _corpus())
    return main(["--indir", str(indir), "--outdir", str(tmp_path / outdir), *extra]), tmp_path / outdir


def _assignments(outdir):
    return [json.loads(line) for line in (outdir / "assignments.jsonl").read_text(encoding="utf-8").splitlines()]


def _topic_labels(assignments):
    # topic -> labels of its chunks (ids 1..40 are the first topic, 41..80 the second, ...)
    by_id = {a["id"]: a["cluster"] for a in assignments}
    return {topic: [by_id[f"{t * PER_TOPIC + i + 1:016x}"] for i in range(PER_TOPIC)]
            for t, topic in enumerate(TOPICS)}


def _assert_topics_recovered(assignments):
    main_label = {}
    for topic, labels in _topic_labels(assignments).items():
        label = max(set(labels), key=labels.count)
        assert label != -1 and labels.count(label) >= 0.9 * PER_TOPIC, topic
        main_label[topic] = label
    assert len(set(main_label.values())) == len(TOPICS)
    return main_label


# Helpers

def test_dedupe_maps_every_copy_to_one_text():
    first, where = dedupe([{"text": "Same  words"}, {"text": "same words"}, {"text": "other"}])
    assert first == [0, 2] and where.tolist() == [0, 0, 1]


def test_by_size_numbers_the_biggest_cluster_zero():
    assert by_size(np.array([3, 3, 3, -1, 7, 7, 1])) == {3: 0, 7: 1, 1: 2, -1: -1}


def test_top_terms_favour_each_clusters_own_words():
    terms = top_terms([TOPICS["deposit"] + " Hi <PERSON>", TOPICS["parking"] + " Hi <PERSON>"])
    assert terms[0][0] == "deposit" and "parking" in terms[1][:3]
    assert not any("person" in term for cluster_terms in terms for term in cluster_terms)


def test_top_terms_without_words_are_empty():
    assert top_terms(["<PERSON>", "the and of"]) == [[], []]
    assert top_terms([]) == []


def test_representatives_are_nearest_the_centre():
    vectors = np.array([[1, 0], [0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert representatives(vectors, n=2).tolist() == [3, 1]


def test_first_reply_is_the_next_outbound_chunk_in_the_same_thread():
    asked = {"thrid": "t1", "date": "2025-01-01T10:00:00-08:00"}  # 18:00 UTC
    outbound = [
        {"thrid": "t1", "date": "2025-01-01T12:00:00+00:00", "chunk_index": 0, "text": "before it, in UTC"},
        {"thrid": "t1", "date": "2025-01-01T20:00:00+00:00", "chunk_index": 1, "text": "after, second chunk"},
        {"thrid": "t1", "date": "2025-01-01T20:00:00+00:00", "chunk_index": 0, "text": "after, first chunk"},
        {"thrid": "t2", "date": "2025-01-01T19:00:00+00:00", "chunk_index": 0, "text": "another thread"},
    ]
    threads = replies_by_thread(outbound)
    assert first_reply(asked, threads)["text"] == "after, first chunk"
    assert first_reply({"thrid": "t1", "date": None}, threads) is None
    assert first_reply({"thrid": "t9", "date": "2025-01-01T10:00:00+00:00"}, threads) is None


# End to end with the SVD stand-in

def test_recovers_each_topic_and_covers_every_inbound_chunk(tmp_path, monkeypatch):
    code, outdir = _run(monkeypatch, tmp_path)
    assert code == 0
    assignments = _assignments(outdir)
    inbound = [r for r in _corpus() if r["direction"] == "inbound"]
    assert sorted(a["chunk_id"] for a in assignments) == sorted(r["chunk_id"] for r in inbound)
    _assert_topics_recovered(assignments)


def test_copies_share_a_label_and_count_toward_size(tmp_path, monkeypatch):
    _, outdir = _run(monkeypatch, tmp_path)
    assignments = _assignments(outdir)
    by_id = {a["id"]: a["cluster"] for a in assignments}
    copies = {by_id[f"{9000 + c:016x}"] for c in range(COPIES)}
    assert copies == {by_id[f"{1:016x}"]}
    rows = {int(r["cluster"]): r for r in csv.DictReader(io.StringIO((outdir / "clusters.csv").read_text()))}
    label = copies.pop()
    assert int(rows[label]["n_chunks"]) == sum(a["cluster"] == label for a in assignments)
    assert int(rows[label]["n_emails"]) == int(rows[label]["n_chunks"])  # each fixture email is one chunk


def test_clusters_csv_names_each_topic(tmp_path, monkeypatch):
    _, outdir = _run(monkeypatch, tmp_path)
    labels = _assert_topics_recovered(_assignments(outdir))
    rows = {int(r["cluster"]): r for r in csv.DictReader(io.StringIO((outdir / "clusters.csv").read_text()))}
    assert "deposit" in rows[labels["deposit"]]["top_terms"]
    assert "parking" in rows[labels["parking"]]["top_terms"]
    assert "lunar" in rows[labels["newyear"]]["top_terms"]
    assert "performance" in rows[labels["performance"]]["top_terms"]
    assert "cost" in rows[labels["cost"]]["top_terms"]
    assert sorted(rows) == list(range(len(rows)))  # numbered 0.. by size
    sizes = [int(rows[k]["n_chunks"]) for k in sorted(rows)]
    assert sizes == sorted(sizes, reverse=True)


def test_report_shows_typical_messages_and_replies(tmp_path, monkeypatch):
    _, outdir = _run(monkeypatch, tmp_path)
    report = (outdir / "report.txt").read_text(encoding="utf-8")
    assert report.startswith("# ") and "=== cluster 0 |" in report
    assert "  1. Hi <PERSON>," in report and "  reply: Happy to help with that" in report
    run = json.loads((outdir / "run.json").read_text())
    assert (run["min_cluster_size"], run["min_samples"], run["method"], run["model"]) == (15, 5, "eom", MODEL)
    assert run["inbound_chunks"] == len(TOPICS) * PER_TOPIC + COPIES + 20


def test_refuses_vectors_from_another_task(tmp_path, monkeypatch):
    rows = _corpus() + [_row(8000, "From the RAG set", np.ones(32) / np.sqrt(32), task="search_document")]
    code, outdir = _run(monkeypatch, tmp_path, rows=rows)
    assert code == 1 and not outdir.exists()


def test_sweep_prints_every_setting_and_writes_nothing(tmp_path, monkeypatch, capsys):
    code, outdir = _run(monkeypatch, tmp_path, "--sweep")
    table = capsys.readouterr().out.split("UMAP:")[1].splitlines()[1:]
    assert code == 0 and not outdir.exists()
    assert table[0].split()[:4] == ["method", "min_size", "min_samples", "clusters"]
    assert len(table) == 1 + len(cluster.SWEEP_SIZES) * len(cluster.SWEEP_SAMPLES) * len(cluster.SWEEP_METHODS)


# The HTML page

def test_page_holds_every_dot_and_cluster(tmp_path, monkeypatch):
    _, outdir = _run(monkeypatch, tmp_path)
    data = _page_data((outdir / "clusters.html").read_text(encoding="utf-8"))
    distinct = len({r["text"] for r in _corpus() if r["direction"] == "inbound"})
    assert len(data["points"]) == 3 * distinct
    rows = list(csv.DictReader(io.StringIO((outdir / "clusters.csv").read_text())))
    assert [c["id"] for c in data["clusters"]] == [int(r["cluster"]) for r in rows]
    assert {label for label in data["points"][2::3]} <= {c["id"] for c in data["clusters"]} | {-1}
    terms = {term for c in data["clusters"] for term in c["terms"]}
    assert {"deposit", "parking", "lunar"} <= terms


def test_page_never_holds_message_text(tmp_path, monkeypatch):
    _, outdir = _run(monkeypatch, tmp_path)
    page = (outdir / "clusters.html").read_text(encoding="utf-8")
    for text in TOPICS.values():
        assert text[:25] not in page
    assert "Happy to help with that" not in page


def test_page_escapes_text_that_would_close_its_script_tag():
    header = ["# 1 cluster </script><script>alert(1)</script>", "# settings"]
    summaries = [{"cluster": 0, "n_chunks": 2, "n_emails": 2, "n_threads": 1, "top_terms": ["deposit"]}]
    page = render_page(summaries, np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, -1]), header)
    assert "</script><script>alert(1)" not in page
    assert _page_data(page)["header"][0] == "1 cluster </script><script>alert(1)</script>"


# The real UMAP

def test_real_umap_recovers_topics_and_repeats_exactly(tmp_path, monkeypatch):
    code, first = _run(monkeypatch, tmp_path, real_umap=True, outdir="first")
    assert code == 0
    _assert_topics_recovered(_assignments(first))
    _, second = _run(monkeypatch, tmp_path, real_umap=True, outdir="second")
    for name in ("assignments.jsonl", "clusters.html"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
