#Tests for analyze_pii (Stage 1)

#A fake analyzer stands in for the trf-backed engine, so the suite never loads the model
#Fixtures are synthetic --> no real PII


import csv
import json

import pytest
from presidio_analyzer import RecognizerResult

import analyze_pii
from analyze_pii import COLUMNS, main


class _FakeAnalyzer:
    #flags every occurrence of each needle, minus anything on the allow list
    #a needle is "Thao" (PERSON, 0.85) or ("The", "PERSON", 0.2, "denylist_ambiguous")
    def __init__(self, needles=("Thao",)):
        self.needles = needles
        self.texts = []

    def analyze(self, text, language, entities, allow_list=None):
        self.texts.append(text)
        results = []
        for needle in self.needles:
            parts = (needle,) if isinstance(needle, str) else needle
            needle, entity, score, recognizer = parts + ("", "PERSON", 0.85, "")[len(parts):]
            if needle in (allow_list or ()):
                continue
            meta = {"recognizer_name": recognizer} if recognizer else None
            start = text.find(needle)
            while start >= 0:
                results.append(RecognizerResult(entity, start, start + len(needle), score,
                                                recognition_metadata=meta))
                start = text.find(needle, start + 1)
        return results


@pytest.fixture
def fake(monkeypatch):
    analyzer = _FakeAnalyzer()
    monkeypatch.setattr(analyze_pii, "build_analyzer", lambda *a, **kw: analyzer)
    return analyzer


BODY = "Hi Thao, are you free?"


def _row(n, subject="Question", body=BODY):
    return {"id": f"<m{n}@example.com>", "thrid": f"179000{n}", "date": "2025-09-01T10:00:00",
            "direction": "inbound", "subject": subject, "body": body}


def _shards(tmp_path, *shards):
    indir = tmp_path / "extracted"
    indir.mkdir()
    for n, rows in enumerate(shards):
        (indir / f"emails-{n:05d}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return indir


def _run(tmp_path, indir, *extra):
    allow = tmp_path / "allow.txt"
    allow.write_text("WeddingWire\n", encoding="utf-8")
    common_dt = tmp_path / "common_dt.txt"
    common_dt.write_text("morning\n", encoding="utf-8")
    common_loc = tmp_path / "common_loc.txt"
    common_loc.write_text("home\n", encoding="utf-8")
    venues = tmp_path / "venues.txt"
    venues.write_text("Casa Romantica\n", encoding="utf-8")
    places = tmp_path / "places.txt"
    places.write_text("[places]\n", encoding="utf-8")
    outdir = tmp_path / "review"
    code = main(["--indir", str(indir), "--outdir", str(outdir), "--allowlist", str(allow),
                 "--common-datetime", str(common_dt), "--common-location", str(common_loc),
                 "--venues", str(venues), "--places", str(places), *extra])
    return code, outdir


def _spans(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# outputs

class TestOutputs:

    def test_writes_three_files_per_shard(self, tmp_path, fake):
        indir = _shards(tmp_path, [_row(1)], [_row(2)])
        code, outdir = _run(tmp_path, indir)
        assert code == 0
        assert sorted(p.name for p in outdir.iterdir()) == [
            "checkpoint.json",
            "emails-00000.spans.csv", "emails-00000.spans.orig.csv", "emails-00000.view.txt",
            "emails-00001.spans.csv", "emails-00001.spans.orig.csv", "emails-00001.view.txt"]

    def test_columns_offsets_and_prefilled_decision(self, tmp_path, fake):
        indir = _shards(tmp_path, [_row(1)])
        _, outdir = _run(tmp_path, indir)
        spans = _spans(outdir / "emails-00000.spans.csv")
        assert len(spans) == 1
        span = spans[0]
        assert list(span) == COLUMNS
        assert (span["email_id"], span["thrid"], span["field"]) == ("<m1@example.com>", "1790001", "body")
        assert (span["entity_type"], span["score"], span["decision"]) == ("PERSON", "0.85", "redact")
        #offsets must slice the same text back out, or apply_redactions can't trust them
        assert BODY[int(span["start"]):int(span["end"])] == span["text"] == "Thao"
        assert "[[Thao]]" in span["context"]

    def test_orig_copy_is_identical(self, tmp_path, fake):
        indir = _shards(tmp_path, [_row(1)])
        _, outdir = _run(tmp_path, indir)
        review = (outdir / "emails-00000.spans.csv").read_text(encoding="utf-8")
        assert review == (outdir / "emails-00000.spans.orig.csv").read_text(encoding="utf-8")

    def test_view_includes_emails_with_no_spans(self, tmp_path, fake):
        #a span file alone can't show an email the model missed entirely
        indir = _shards(tmp_path, [_row(1), _row(2, body="No names here at all")])
        _, outdir = _run(tmp_path, indir)
        view = (outdir / "emails-00000.view.txt").read_text(encoding="utf-8")
        assert "[[PERSON:Thao]]" in view
        assert "<m2@example.com>" in view and "No names here at all" in view

    def test_decision_and_rule_come_from_the_rules(self, tmp_path, monkeypatch):
        analyzer = _FakeAnalyzer(needles=("Thao", ("Irvine", "LOCATION"), ("Casa Romantica", "LOCATION"),
                                          ("morning", "DATE_TIME"), ("October 22", "DATE_TIME"),
                                          ("The", "PERSON", 0.2, "denylist_ambiguous")))
        monkeypatch.setattr(analyze_pii, "build_analyzer", lambda *a, **kw: analyzer)
        body = "Hi Thao, good morning! The lions: October 22 at Casa Romantica in Irvine?"
        indir = _shards(tmp_path, [_row(1, body=body)])
        _, outdir = _run(tmp_path, indir)
        got = {s["text"]: (s["decision"], s["rule"]) for s in _spans(outdir / "emails-00000.spans.csv")}
        assert got == {"Thao": ("redact", "person_always"),
                       "morning": ("keep", "common_word"),
                       "October 22": ("redact", "calendar_date"),
                       "Casa Romantica": ("redact", "location_always"),
                       "Irvine": ("redact", "location_always"),
                       "The": ("keep", "ambiguous_word")}

    def test_signature_block_becomes_one_reviewable_row(self, tmp_path, monkeypatch):
        analyzer = _FakeAnalyzer(needles=(("Thao Nguyen", "PERSON"), ("thao@example.com", "EMAIL_ADDRESS"),
                                          ("714-555-0100", "PHONE_NUMBER")))
        monkeypatch.setattr(analyze_pii, "build_analyzer", lambda *a, **kw: analyzer)
        body = "Can we book two lions?\nThanks,\nThao Nguyen\nthao@example.com | 714-555-0100\n"
        indir = _shards(tmp_path, [_row(1, body=body)])
        _, outdir = _run(tmp_path, indir)
        rows = [s for s in _spans(outdir / "emails-00000.spans.csv") if s["entity_type"] == "SIGNATURE"]
        assert [(s["text"], s["decision"], s["rule"]) for s in rows] == [
            (body[body.index("Thanks,"):], "redact", "signature_block")]
        #one block marker; the redactions inside it aren't drawn over its offsets
        view = (outdir / "emails-00000.view.txt").read_text(encoding="utf-8")
        assert "[[SIGNATURE:Thanks,\nThao Nguyen\nthao@example.com | 714-555-0100\n]]" in view
        assert "[[PERSON:" not in view

    def test_allow_list_suppresses_hits(self, tmp_path, monkeypatch):
        analyzer = _FakeAnalyzer(needles=("WeddingWire",))
        monkeypatch.setattr(analyze_pii, "build_analyzer", lambda *a, **kw: analyzer)
        indir = _shards(tmp_path, [_row(1, body="Booked via WeddingWire")])
        _, outdir = _run(tmp_path, indir)
        assert _spans(outdir / "emails-00000.spans.csv") == []


# resuming

class TestResume:

    def test_second_run_skips_finished_shards(self, tmp_path, fake):
        indir = _shards(tmp_path, [_row(1)])
        _run(tmp_path, indir)
        fake.texts.clear()
        assert _run(tmp_path, indir)[0] == 0
        assert fake.texts == []

    def test_force_reanalyses_unedited_shards(self, tmp_path, fake):
        indir = _shards(tmp_path, [_row(1)])
        _run(tmp_path, indir)
        fake.texts.clear()
        _run(tmp_path, indir, "--force")
        assert fake.texts

    def test_force_never_overwrites_edited_review_files(self, tmp_path, fake):
        indir = _shards(tmp_path, [_row(1)])
        _, outdir = _run(tmp_path, indir)
        review = outdir / "emails-00000.spans.csv"
        kept = review.read_text(encoding="utf-8").replace(",redact,", ",keep,")
        review.write_text(kept, encoding="utf-8")
        _run(tmp_path, indir, "--force")
        assert review.read_text(encoding="utf-8") == kept


# smoke tests

class TestLimit:

    def test_limit_writes_to_a_separate_dir(self, tmp_path, fake):
        #a half-analyzed shard in data/review would look finished to the next run
        indir = _shards(tmp_path, [_row(1), _row(2)])
        _, outdir = _run(tmp_path, indir, "--limit", "1")
        assert len(_spans(outdir / "smoke" / "emails-00000.spans.csv")) == 1
        assert not (outdir / "emails-00000.spans.csv").exists()
