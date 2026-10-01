#Tests for pii_sample (Stage 1)

#Fixtures are synthetic emails and review files --> no real PII


import csv
import json

from analyze_pii import COLUMNS
from pii_sample import main


def _email(n, body):
    return {"id": f"<m{n}@example.com>", "thrid": f"t{n}", "subject": "Question", "body": body}


def _span(n, text, body, entity="PERSON", decision="redact", rule="person_always"):
    start = body.index(text)
    return {"email_id": f"<m{n}@example.com>", "thrid": f"t{n}", "field": "body",
            "entity_type": entity, "start": str(start), "end": str(start + len(text)), "text": text,
            "context": "", "score": "0.85", "recognizer": "SpacyRecognizer", "decision": decision,
            "rule": rule, "note": ""}


def _setup(tmp_path, emails, spans, unanalyzed=()):
    indir = tmp_path / "extracted"
    indir.mkdir()
    (indir / "emails-00000.jsonl").write_text("".join(json.dumps(e) + "\n" for e in emails), encoding="utf-8")
    if unanalyzed:
        (indir / "emails-00001.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in unanalyzed), encoding="utf-8")
    reviewdir = tmp_path / "review"
    reviewdir.mkdir()
    with open(reviewdir / "emails-00000.spans.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(spans)
    return indir, reviewdir


def _run(indir, reviewdir, *extra):
    code = main(["--indir", str(indir), "--reviewdir", str(reviewdir), *extra])
    return code, (reviewdir / "sample.txt").read_text(encoding="utf-8")


def _headers(sample):
    return [line for line in sample.splitlines() if line.startswith("=== ")]


class TestDraw:

    def test_top_person_terms_are_guaranteed(self, tmp_path):
        #"Anh" is the most frequent name, so its one email must be drawn even at size 2
        emails = [_email(n, f"hello number {n}") for n in range(1, 30)] + [_email(99, "Hi Anh, Anh here")]
        spans = [_span(99, "Anh", "Hi Anh, Anh here")]
        _, sample = _run(*_setup(tmp_path, emails, spans), "--size", "2", "--top-person", "1")
        headers = _headers(sample)
        assert len(headers) == 2
        assert "top PERSON #1: anh" in headers[0] and "<m99@example.com>" in headers[0]

    def test_top_person_counts_only_redacted_spans(self, tmp_path):
        #kept ambiguous words ("The") outnumber every name, but aren't worth a guaranteed slot
        emails = [_email(n, f"The lions {n}") for n in range(1, 6)] + [_email(99, "Hi Anh")]
        spans = [_span(n, "The", f"The lions {n}", decision="keep", rule="ambiguous_word") for n in range(1, 6)]
        spans.append(_span(99, "Anh", "Hi Anh"))
        _, sample = _run(*_setup(tmp_path, emails, spans), "--size", "1", "--top-person", "1")
        assert "top PERSON #1: anh" in sample

    def test_random_fill_includes_emails_with_no_spans(self, tmp_path):
        #PII the recognizers missed only shows up in emails the span file never mentions
        emails = [_email(n, f"no spans in {n}") for n in range(1, 6)]
        _, sample = _run(*_setup(tmp_path, emails, []), "--size", "5")
        assert len(_headers(sample)) == 5 and "no spans in 3" in sample

    def test_same_seed_same_sample(self, tmp_path):
        emails = [_email(n, f"body {n}") for n in range(1, 50)]
        indir, reviewdir = _setup(tmp_path, emails, [])
        _, first = _run(indir, reviewdir, "--size", "10")
        _, again = _run(indir, reviewdir, "--size", "10")
        _, other = _run(indir, reviewdir, "--size", "10", "--seed", "7")
        assert first == again and first != other

    def test_skips_shards_without_a_review_file(self, tmp_path):
        emails = [_email(1, "analyzed")]
        indir, reviewdir = _setup(tmp_path, emails, [], unanalyzed=[_email(2, "never analyzed")])
        _, sample = _run(indir, reviewdir, "--size", "10")
        assert "analyzed" in sample and "never analyzed" not in sample


class TestRender:

    def test_marks_redact_and_keep_with_the_rule(self, tmp_path):
        body = "Hi Thao, see you at 8am"
        spans = [_span(1, "Thao", body),
                 _span(1, "8am", body, entity="DATE_TIME", decision="keep", rule="clock_time")]
        _, sample = _run(*_setup(tmp_path, [_email(1, body)], spans), "--size", "1", "--top-person", "0")
        assert "body: Hi [[REDACT PERSON: Thao]], see you at [[keep DATE_TIME clock_time: 8am]]" in sample
        assert "2 spans (1 redact, 1 keep)" in sample

    def test_overlapping_spans_redact_if_any_member_does(self, tmp_path):
        #apply_redactions redacts every redact span, so a keep over a redact must not read as kept
        body = "at Desert Hills Outlets today"
        spans = [_span(1, "Desert Hills Outlets", body, entity="ORGANIZATION", rule="default"),
                 _span(1, "Desert Hills", body, entity="LOCATION", decision="keep", rule="common_word")]
        _, sample = _run(*_setup(tmp_path, [_email(1, body)], spans), "--size", "1", "--top-person", "0")
        assert "body: at [[REDACT ORGANIZATION/LOCATION: Desert Hills Outlets]] today" in sample

    def test_signature_block_shows_as_one_marker(self, tmp_path):
        #the sample shows exactly what apply_redactions strips, as one block
        body = "Can we book two lions?\nThanks,\nThao Nguyen\nthao@example.com | 714-555-0100\n"
        zone = body[body.index("Thanks,"):]
        spans = [_span(1, "Thao Nguyen", body), _span(1, "thao@example.com", body, entity="EMAIL_ADDRESS"),
                 _span(1, "714-555-0100", body, entity="PHONE_NUMBER"),
                 _span(1, zone, body, entity="SIGNATURE", rule="signature_block")]
        _, sample = _run(*_setup(tmp_path, [_email(1, body)], spans), "--size", "1", "--top-person", "0")
        assert "body: Can we book two lions?\n[[REDACT SIGNATURE: Thanks,\nThao Nguyen" in sample
