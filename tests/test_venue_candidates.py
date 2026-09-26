#Tests for venue_candidates (Stage 1)

#Fixtures are synthetic review files written with analyze_pii's own COLUMNS --> no real PII


import csv

from analyze_pii import COLUMNS
from venue_candidates import main


def _span(text, thrid, entity="LOCATION", n=0):
    return {"email_id": f"<m{thrid}-{n}@example.com>", "thrid": thrid, "field": "body",
            "entity_type": entity, "start": "0", "end": str(len(text)), "text": text,
            "context": f"[[{text}]]", "score": "0.85", "recognizer": "SpacyRecognizer",
            "decision": "keep", "rule": "location_baseline", "note": ""}


def _run(tmp_path, spans, *extra):
    reviewdir = tmp_path / "review"
    reviewdir.mkdir()
    with open(reviewdir / "emails-00000.spans.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(spans)
    common = tmp_path / "common.txt"
    common.write_text("home\n", encoding="utf-8")
    venues = tmp_path / "venues.txt"
    venues.write_text("Casa Romantica\n", encoding="utf-8")
    out = tmp_path / "venue_candidates.csv"
    code = main(["--reviewdir", str(reviewdir), "--out", str(out), "--common-location", str(common),
                 "--venues", str(venues), *extra])
    with open(out, encoding="utf-8", newline="") as fh:
        return code, list(csv.DictReader(fh))


class TestCandidates:

    def test_counts_threads_not_spans(self, tmp_path):
        #a venue quoted down one long thread is still one client
        spans = [_span("Rancho Las Lomas", "t1", n=i) for i in range(5)]
        _, rows = _run(tmp_path, spans)
        assert [(r["term"], r["threads"], r["spans"]) for r in rows] == [("rancho las lomas", "1", "5")]

    def test_threshold_drops_repeat_places(self, tmp_path):
        spans = [_span("Irvine", t) for t in ("t1", "t2", "t3")] + [_span("Rancho Las Lomas", "t1")]
        _, rows = _run(tmp_path, spans)
        assert [r["term"] for r in rows] == ["rancho las lomas"]

    def test_max_threads_is_tunable(self, tmp_path):
        spans = [_span("Irvine", t) for t in ("t1", "t2", "t3")]
        _, rows = _run(tmp_path, spans, "--max-threads", "3")
        assert [r["term"] for r in rows] == ["irvine"]

    def test_skips_common_words_venue_list_and_other_entities(self, tmp_path):
        spans = [_span("home", "t1"), _span("Casa Romantica", "t1"), _span("Thao", "t1", entity="PERSON")]
        code, rows = _run(tmp_path, spans)
        assert code == 0 and rows == []

    def test_sorted_by_span_count(self, tmp_path):
        spans = [_span("Oak Hall", "t1")] + [_span("Rose Barn", "t2", n=i) for i in range(3)]
        _, rows = _run(tmp_path, spans)
        assert [r["term"] for r in rows] == ["rose barn", "oak hall"]
