#Tests for apply_redactions (Stage 1b)

#Fixtures are synthetic emails and review files written to tmp_path --> no real PII, nothing in git
#The review CSVs are written with analyze_pii's own COLUMNS, so the two scripts stay in step


import csv
import json

from analyze_pii import COLUMNS
from apply_redactions import HASH_CHARS, main

BODY = "Hi Thao, are you free? Thao is asking."
EMAIL_ID = "<m1@example.com>"


def _email(n=1, subject="Question", body=BODY, **extra):
    row = {"id": f"<m{n}@example.com>", "thrid": f"179000{n}", "date": "2025-09-01T10:00:00",
           "direction": "inbound", "labels": ["Inbox"], "from": f'"Thao Nguyen" <thao{n}@example.com>',
           "sender": f"thao{n}@example.com", "is_automated": False, "to": "shop@example.com",
           "subject": subject, "body": body, "n_words": len(body.split()),
           "body_source": "text/plain", "n_attachments": 0, "attachments": [], "offset": 100 * n}
    row.update(extra)
    return row


def _span(email_id=EMAIL_ID, field="body", entity="PERSON", start="3", end="7", text="Thao",
          decision="redact"):
    return {"email_id": email_id, "thrid": "1790001", "field": field, "entity_type": entity,
            "start": start, "end": end, "text": text, "context": f"[[{text}]]", "score": "0.85",
            "recognizer": "SpacyRecognizer", "decision": decision, "note": ""}


def _write_csv(path, spans):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(spans)


def _setup(tmp_path, emails, spans, *, original=None, view=True):
    indir = tmp_path / "extracted"
    indir.mkdir()
    (indir / "emails-00000.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in emails), encoding="utf-8")
    reviewdir = tmp_path / "review"
    reviewdir.mkdir()
    _write_csv(reviewdir / "emails-00000.spans.csv", spans)
    _write_csv(reviewdir / "emails-00000.spans.orig.csv", spans if original is None else original)
    if view:
        (reviewdir / "emails-00000.view.txt").write_text(
            "".join(f"=== {e['id']} | {e['date']} | inbound | thrid x | 0 spans\n" for e in emails),
            encoding="utf-8")
    return indir, reviewdir


def _run(tmp_path, indir, reviewdir, *extra):
    outdir = tmp_path / "redacted"
    code = main(["--indir", str(indir), "--reviewdir", str(reviewdir), "--outdir", str(outdir),
                 "--hash-key", str(tmp_path / "hash_key"), *extra])
    return code, outdir


def _out(outdir):
    path = outdir / "emails-00000.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# redacting

class TestRedacting:

    def test_redact_replaces_only_the_span_at_those_offsets(self, tmp_path, ):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span()])
        code, outdir = _run(tmp_path, indir, reviewdir)
        assert code == 0
        row = _out(outdir)[0]
        #the second "Thao" has no row of its own, so precise offsets must leave it alone
        assert row["body"] == "Hi <PERSON>, are you free? Thao is asking."
        assert row["n_words"] == len(row["body"].split())
        assert row["n_redacted"] == 1

    def test_keep_leaves_the_text_alone(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(decision="keep")])
        _, outdir = _run(tmp_path, indir, reviewdir)
        assert _out(outdir)[0]["body"] == BODY

    def test_stale_offsets_fall_back_to_text_search(self, tmp_path):
        #a hand-widened span won't slice back out, so every occurrence goes instead
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(start="99", end="103")])
        _, outdir = _run(tmp_path, indir, reviewdir)
        assert _out(outdir)[0]["body"] == "Hi <PERSON>, are you free? <PERSON> is asking."

    def test_added_row_without_offsets_redacts_every_occurrence(self, tmp_path):
        added = _span(start="", end="")
        indir, reviewdir = _setup(tmp_path, [_email()], [added], original=[])
        _, outdir = _run(tmp_path, indir, reviewdir)
        assert _out(outdir)[0]["body"] == "Hi <PERSON>, are you free? <PERSON> is asking."

    def test_subject_spans_are_separate_from_body(self, tmp_path):
        email = _email(subject="Quote for Thao")
        span = _span(field="subject", start="10", end="14")
        indir, reviewdir = _setup(tmp_path, [email], [span])
        _, outdir = _run(tmp_path, indir, reviewdir)
        row = _out(outdir)[0]
        assert row["subject"] == "Quote for <PERSON>"
        assert row["body"] == BODY

    def test_entity_type_becomes_the_tag(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(entity="LOCATION")])
        _, outdir = _run(tmp_path, indir, reviewdir)
        assert "<LOCATION>" in _out(outdir)[0]["body"]


# review-file checks

class TestChecks:

    def test_text_that_is_gone_is_an_error(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(text="Nguyen", start="", end="")])
        code, outdir = _run(tmp_path, indir, reviewdir)
        assert code == 1
        assert _out(outdir) is None

    def test_deleted_row_is_an_error(self, tmp_path):
        #deleting a row would otherwise mean no decision at all, so nothing gets redacted
        indir, reviewdir = _setup(tmp_path, [_email()], [], original=[_span()])
        code, outdir = _run(tmp_path, indir, reviewdir)
        assert code == 1
        assert _out(outdir) is None

    def test_relabelling_a_row_is_allowed(self, tmp_path):
        #the deleted-row check keys on offsets, so changing entity_type is fine
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(entity="ORGANIZATION")],
                                  original=[_span()])
        assert _run(tmp_path, indir, reviewdir)[0] == 0

    def test_unknown_decision_is_an_error(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(decision="kepe")])
        assert _run(tmp_path, indir, reviewdir)[0] == 1

    def test_empty_decision_is_an_error(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(decision="")])
        assert _run(tmp_path, indir, reviewdir)[0] == 1

    def test_unknown_email_id_is_an_error(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span(email_id="<gone@example.com>")])
        assert _run(tmp_path, indir, reviewdir)[0] == 1

    def test_missing_view_file_is_an_error(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span()], view=False)
        assert _run(tmp_path, indir, reviewdir)[0] == 1

    def test_email_missing_from_the_view_is_an_error(self, tmp_path):
        #a --limit run analyzes part of a shard; the rest must not slip out unredacted
        indir, reviewdir = _setup(tmp_path, [_email(1), _email(2)], [_span()])
        (reviewdir / "emails-00000.view.txt").write_text(
            "=== <m1@example.com> | d | inbound | thrid x | 1 spans\n", encoding="utf-8")
        assert _run(tmp_path, indir, reviewdir)[0] == 1

    def test_shard_without_a_review_file_is_skipped_not_written(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span()])
        (reviewdir / "emails-00000.spans.csv").unlink()
        code, outdir = _run(tmp_path, indir, reviewdir)
        assert code == 0
        assert _out(outdir) is None


# headers

class TestHeaders:

    def test_addresses_and_ids_are_hashed_and_from_is_dropped(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span()])
        _, outdir = _run(tmp_path, indir, reviewdir)
        row = _out(outdir)[0]
        assert "from" not in row
        assert len(row["sender"]) == HASH_CHARS and "@" not in row["sender"]
        assert len(row["id"]) == HASH_CHARS and "@" not in row["id"]
        assert row["to"] == [row["to"][0]] and "@" not in row["to"][0]
        #kept as-is: they carry no names and thread order depends on them
        assert (row["thrid"], row["labels"], row["direction"]) == ("1790001", ["Inbox"], "inbound")

    def test_same_address_hashes_the_same_way(self, tmp_path):
        emails = [_email(1), _email(2, body="Hi Thao"), ]
        emails[1]["sender"] = emails[0]["sender"]
        spans = [_span(), _span(email_id="<m2@example.com>", start="3", end="7")]
        indir, reviewdir = _setup(tmp_path, emails, spans)
        _, outdir = _run(tmp_path, indir, reviewdir)
        first, second = _out(outdir)
        assert first["sender"] == second["sender"]
        assert first["id"] != second["id"]

    def test_attachment_filenames_are_hashed_but_type_and_size_stay(self, tmp_path):
        attachment = {"filename": "Nguyen_Contract.pdf", "content_type": "application/pdf",
                      "approx_bytes": 2048}
        email = _email(n_attachments=1, attachments=[attachment])
        indir, reviewdir = _setup(tmp_path, [email], [_span()])
        _, outdir = _run(tmp_path, indir, reviewdir)
        out = _out(outdir)[0]["attachments"][0]
        assert len(out["filename"]) == HASH_CHARS and "Nguyen" not in out["filename"]
        assert (out["content_type"], out["approx_bytes"]) == ("application/pdf", 2048)

    def test_hash_key_is_written_once_and_reused(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span()])
        _, outdir = _run(tmp_path, indir, reviewdir)
        first = _out(outdir)[0]["sender"]
        key = (tmp_path / "hash_key").read_text(encoding="utf-8")
        _run(tmp_path, indir, reviewdir, "--force")
        assert _out(outdir)[0]["sender"] == first
        assert (tmp_path / "hash_key").read_text(encoding="utf-8") == key


# resuming

class TestResume:

    def test_second_run_skips_written_shards(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span()])
        _, outdir = _run(tmp_path, indir, reviewdir)
        (outdir / "emails-00000.jsonl").write_text("sentinel\n", encoding="utf-8")
        assert _run(tmp_path, indir, reviewdir)[0] == 0
        assert (outdir / "emails-00000.jsonl").read_text(encoding="utf-8") == "sentinel\n"

    def test_force_rewrites(self, tmp_path):
        indir, reviewdir = _setup(tmp_path, [_email()], [_span()])
        _, outdir = _run(tmp_path, indir, reviewdir)
        (outdir / "emails-00000.jsonl").write_text("sentinel\n", encoding="utf-8")
        _run(tmp_path, indir, reviewdir, "--force")
        assert _out(outdir)[0]["body"].startswith("Hi <PERSON>")
