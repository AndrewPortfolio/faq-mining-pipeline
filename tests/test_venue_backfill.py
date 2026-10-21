# Tests for faq_mining.pipelines.venue_backfill (Stage 1 backfill)

# Fixtures are synthetic emails and review files written to tmp_path --> no real PII, nothing in git
# The review CSVs are written with analyze_pii's own COLUMNS, as in test_apply_redactions


import csv
import json

import apply_redactions
from analyze_pii import COLUMNS
from faq_mining.pipelines.venue_backfill import (
    NOTE,
    covered,
    harvest,
    label_recognizer,
    leaking,
    main,
    new_spans,
)
from shared.decisions import Rules
from shared.pii_recognizers import venue_recognizers

FORM = "Name: Ann\nLocation: Sunny Acres Pavilion, Irvine CA\nDate/Time of Event: May 5\nMessage: how much?"
VALUE = "Sunny Acres Pavilion, Irvine CA"


def _email(n=1, body=FORM, subject="New form entry"):
    return {
        "id": f"<m{n}@example.com>",
        "thrid": f"179000{n}",
        "date": "2025-09-01T10:00:00",
        "direction": "inbound",
        "labels": ["Inbox"],
        "from": f"ann{n}@example.com",
        "sender": f"ann{n}@example.com",
        "is_automated": False,
        "to": "shop@example.com",
        "subject": subject,
        "body": body,
        "n_words": len(body.split()),
        "body_source": "text/plain",
        "n_attachments": 0,
        "attachments": [],
        "offset": 100 * n,
    }


def _span(text, body=FORM, entity="LOCATION", decision="redact", start=None):
    start = body.index(text) if start is None else start
    return {
        "email_id": "<m1@example.com>",
        "thrid": "1790001",
        "field": "body",
        "entity_type": entity,
        "start": str(start),
        "end": str(start + len(text)),
        "text": text,
        "context": f"[[{text}]]",
        "score": "0.85",
        "recognizer": "SpacyRecognizer",
        "decision": decision,
        "rule": "",
        "note": "",
    }


def _write_csv(path, spans):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(spans)


def _rules():
    return Rules(common_location=["orange county"])


# candidates


class TestHarvest:
    def test_cuts_at_the_comma_and_counts_lines(self):
        found = harvest([FORM, FORM.replace("Location:", "Venue:")], set())
        assert found == {"sunny acres pavilion": ("Sunny Acres Pavilion", 2)}

    def test_drops_descriptions_addresses_and_listed_values(self):
        body = "Location: my backyard\nLocation: 12 Oak St, Irvine\nVenue: Big Oak Barn\nLocation: White Palace"
        assert harvest([body], {"white palace"}) == {"big oak barn": ("Big Oak Barn", 1)}


class TestLeaking:
    def test_counts_emails_holding_it_outside_a_tag(self):
        texts = ["we loved Big Oak Barn!", "<LOCATION> and big oak barn", "a big oak barnyard", "<LOCATION>"]
        assert leaking(["big oak barn"], texts) == {"big oak barn": 2}


# backfill


class TestCovered:
    def test_union_of_spans_covers(self):
        assert covered(10, 20, [(0, 2), (10, 15), (14, 20)])
        assert not covered(10, 20, [(10, 15), (16, 20)])
        assert not covered(10, 20, [])


class TestNewSpans:
    def test_label_value_is_appended_as_a_redact_row(self):
        spans = new_spans(_email(), [], [label_recognizer()], _rules())
        assert [(s["text"], s["decision"], s["recognizer"], s["note"]) for s in spans] == [
            (VALUE, "redact", "label_location", NOTE)
        ]

    def test_skips_what_a_redact_row_already_covers(self):
        existing = [_span("Sunny Acres Pavilion"), _span(", Irvine CA")]
        assert new_spans(_email(), existing, [label_recognizer()], _rules()) == []

    def test_partial_cover_still_appends(self):
        # trf took "Irvine" but not the venue: the venue is what leaked
        spans = new_spans(_email(), [_span("Irvine")], [label_recognizer()], _rules())
        assert [s["text"] for s in spans] == [VALUE]

    def test_a_row_kept_by_hand_stays_kept(self):
        existing = [_span(VALUE, decision="keep")]
        assert new_spans(_email(), existing, [label_recognizer()], _rules()) == []

    def test_region_word_is_decided_keep(self):
        email = _email(body="Location: Orange County\nMessage: hi")
        spans = new_spans(email, [], [label_recognizer()], _rules())
        assert [(s["text"], s["decision"], s["rule"]) for s in spans] == [("Orange County", "keep", "common_word")]

    def test_deny_list_venues_in_free_text_and_subject(self):
        email = _email(body="see you at the white palace ballroom", subject="White Palace booking")
        recognizers = venue_recognizers(["White Palace", "Palace Ballroom"])
        spans = new_spans(email, [], recognizers, _rules())
        assert sorted((s["field"], s["text"]) for s in spans) == [
            ("body", "palace ballroom"),
            ("body", "white palace"),
            ("subject", "White Palace"),
        ]

    def test_second_pass_finds_nothing(self):
        recognizers = [label_recognizer(), *venue_recognizers(["Sunny Acres Pavilion"])]
        first = new_spans(_email(), [], recognizers, _rules())
        assert first
        read_back = [{k: str(v) for k, v in s.items()} for s in first]  # as csv.DictReader returns them
        assert new_spans(_email(), read_back, recognizers, _rules()) == []


class TestRun:
    def _setup(self, tmp_path, spans):
        indir = tmp_path / "extracted"
        indir.mkdir()
        (indir / "emails-00000.jsonl").write_text(json.dumps(_email()) + "\n", encoding="utf-8")
        reviewdir = tmp_path / "review"
        reviewdir.mkdir()
        _write_csv(reviewdir / "emails-00000.spans.csv", spans)
        _write_csv(reviewdir / "emails-00000.spans.orig.csv", spans)
        (reviewdir / "emails-00000.view.txt").write_text("=== <m1@example.com> | x | inbound | 1 spans\n")
        venues = tmp_path / "venues.txt"
        venues.write_text("# none yet\n", encoding="utf-8")
        common = tmp_path / "common.txt"
        common.write_text("", encoding="utf-8")
        args = ["--indir", str(indir), "--reviewdir", str(reviewdir), "--venues", str(venues)]
        args += ["--common-datetime", str(common), "--common-location", str(common)]
        return indir, reviewdir, args

    def test_appends_once_and_apply_redactions_accepts_it(self, tmp_path):
        indir, reviewdir, args = self._setup(tmp_path, [_span("Irvine")])
        review = reviewdir / "emails-00000.spans.csv"
        before, orig = review.read_text(), (reviewdir / "emails-00000.spans.orig.csv").read_text()

        assert main(["backfill", *args]) == 0
        after = review.read_text()
        assert after.startswith(before) and after.count("\n") == before.count("\n") + 1
        assert (reviewdir / "emails-00000.spans.orig.csv").read_text() == orig

        assert main(["backfill", *args]) == 0
        assert review.read_text() == after

        outdir = tmp_path / "redacted"
        code = apply_redactions.main(
            ["--indir", str(indir), "--reviewdir", str(reviewdir), "--outdir", str(outdir)]
            + ["--hash-key", str(tmp_path / "hash_key")]
        )
        assert code == 0
        body = json.loads((outdir / "emails-00000.jsonl").read_text())["body"]
        assert "Location: <LOCATION>\n" in body and "Sunny" not in body

    def test_dry_run_writes_nothing(self, tmp_path):
        _, reviewdir, args = self._setup(tmp_path, [])
        before = (reviewdir / "emails-00000.spans.csv").read_text()
        assert main(["backfill", *args, "--dry-run"]) == 0
        assert (reviewdir / "emails-00000.spans.csv").read_text() == before
