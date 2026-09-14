#Tests for extract_mbox (Stage 0)

#Fixtures are synthetic mbox files written to tmp_path --> no real PII, nothing in git
#Envelope lines must match ENVELOPE_RE ("From <digits>@xxx ") or messages merge into one


import base64
import itertools

import pytest

import extract_mbox
from extract_mbox import build_row, decode_part, iter_messages, main
from shared.pipeline import Checkpoint, iter_shard_rows, shard_paths


_ids = itertools.count(1)


def _msg(body, *, subject="Question", sender="customer@example.com", labels="Inbox",
         msgid=None, content_type="text/plain; charset=utf-8", headers=()):
    msgid = msgid or f"m{next(_ids)}"
    lines = [
        f"X-Gmail-Labels: {labels}",
        f"Message-ID: <{msgid}@example.com>",
        f"From: {sender}",
        "To: shop@example.com",
        f"Subject: {subject}",
        "Date: Mon, 1 Sep 2025 10:00:00 +0000",
        f"Content-Type: {content_type}",
        *headers,
    ]
    return "\n".join(lines) + "\n\n" + body + "\n\n"


def _mbox(tmp_path, *messages):
    path = tmp_path / "test.mbox"
    text = "".join(f"From {1000 + i}@xxx Mon Sep 01 10:00:00 +0000 2025\n{m}"
                   for i, m in enumerate(messages))
    path.write_bytes(text.encode("utf-8"))
    return path


def _row(tmp_path, message, **kw):
    return build_row(next(iter_messages(str(_mbox(tmp_path, message)))), **kw)


def _run(mbox, outdir, *extra):
    assert main(["--mbox", str(mbox), "--outdir", str(outdir), *extra]) == 0
    stats = Checkpoint(str(outdir / "checkpoint.json")).load()["stats"]
    return stats, list(iter_shard_rows(str(outdir)))


#One of each outcome, with a duplicate Message-ID after the spam so it lands
#on the far side of a --limit 3 resume
MIXED = [
    _msg("Do you have availability in June?", msgid="q1"),
    _msg("Yes we do.", msgid="a1", sender="shop@example.com", labels="Sent"),
    _msg("Buy now", msgid="spam", labels="Spam"),
    _msg("Big sale", msgid="promo", headers=["List-Unsubscribe: <mailto:u@example.com>"]),
    _msg("Different body, same Message-ID", msgid="q1"),
    _msg("What are your rates?", msgid="q2"),
]


# decode_part

class TestDecodePart:

    def test_base64(self):
        assert decode_part(base64.b64encode("café".encode()), "base64", "utf-8") == "café"

    def test_quoted_printable(self):
        assert decode_part(b"caf=C3=A9", "quoted-printable", "utf-8") == "café"

    def test_wrong_declared_charset_falls_back(self):
        #mislabelled charsets are common in old mail --> fall through instead of raising
        assert decode_part("café".encode("cp1252"), "", "utf-8") == "café"

    def test_unknown_charset_name_falls_back(self):
        assert decode_part(b"hello", "", "x-made-up") == "hello"

    def test_corrupt_base64_yields_empty_not_error(self):
        assert decode_part(b"abc", "base64", "utf-8") == ""


# iter_messages

class TestIterMessages:

    def test_splits_on_envelope_lines(self, tmp_path):
        mbox = _mbox(tmp_path, _msg("one"), _msg("two"), _msg("three"))
        assert len(list(iter_messages(str(mbox)))) == 3

    def test_body_line_starting_with_from_does_not_split(self, tmp_path):
        #only Gmail's envelope format starts a message --> prose beginning with "From" stays in the body
        builders = list(iter_messages(str(_mbox(tmp_path, _msg("Thanks!\nFrom the whole team")))))
        assert len(builders) == 1
        row, _ = build_row(builders[0])
        assert "From the whole team" in row["body"]

    def test_offsets_cover_the_file_with_no_gaps(self, tmp_path):
        #end_offset is the resume point --> a gap skips a message on resume, an overlap re-reads one
        mbox = _mbox(tmp_path, _msg("one"), _msg("two"), _msg("three"))
        builders = list(iter_messages(str(mbox)))
        assert builders[0].offset == 0
        assert all(a.end_offset == b.offset for a, b in zip(builders, builders[1:]))
        assert builders[-1].end_offset == mbox.stat().st_size

    def test_start_offset_skips_earlier_messages(self, tmp_path):
        mbox = _mbox(tmp_path, _msg("one", subject="A"), _msg("two", subject="B"),
                     _msg("three", subject="C"))
        start = next(iter_messages(str(mbox))).end_offset
        subjects = [str(b.headers["Subject"]) for b in iter_messages(str(mbox), start)]
        assert subjects == ["B", "C"]


# Body extraction

class TestBodyExtraction:

    def test_attachment_bytes_skipped_but_recorded(self, tmp_path):
        payload = "JVBERi0xLjQK" * 6
        body = (
            "--MIX\nContent-Type: text/plain; charset=utf-8\n\nContract attached.\n"
            "--MIX\nContent-Type: application/pdf; name=\"contract.pdf\"\n"
            "Content-Disposition: attachment; filename=\"contract.pdf\"\n"
            "Content-Transfer-Encoding: base64\n\n"
            + f"{payload}\n" * 20 +
            "--MIX--"
        )
        row, _ = _row(tmp_path, _msg(body, content_type='multipart/mixed; boundary="MIX"'))
        assert row["body"] == "Contract attached."
        assert payload not in row["body"]
        assert row["n_attachments"] == 1
        att = row["attachments"][0]
        assert (att["filename"], att["content_type"]) == ("contract.pdf", "application/pdf")
        assert att["approx_bytes"] > 0

    def test_plain_preferred_over_html(self, tmp_path):
        body = (
            "--ALT\nContent-Type: text/plain; charset=utf-8\n\nPlain version\n"
            "--ALT\nContent-Type: text/html; charset=utf-8\n\n<p>HTML version</p>\n"
            "--ALT--"
        )
        row, _ = _row(tmp_path, _msg(body, content_type='multipart/alternative; boundary="ALT"'))
        assert (row["body"], row["body_source"]) == ("Plain version", "text/plain")

    def test_html_only_falls_back_to_rendered_text(self, tmp_path):
        row, _ = _row(tmp_path, _msg("<p>Hi there</p><p>Second line</p>",
                                     content_type="text/html; charset=utf-8"))
        assert row["body_source"] == "text/html"
        assert "<p>" not in row["body"]
        assert "Hi there" in row["body"] and "Second line" in row["body"]

    def test_base64_text_part_decoded(self, tmp_path):
        encoded = base64.b64encode(b"Hello from base64").decode()
        row, _ = _row(tmp_path, _msg(encoded, headers=["Content-Transfer-Encoding: base64"]))
        assert row["body"] == "Hello from base64"


# build_row

class TestBuildRow:

    def test_row_shape(self, tmp_path):
        #later stages read these keys --> a rename here breaks them without an error
        row, reason = _row(tmp_path, _msg("How much for a 20 minute set?",
                                          sender="Sample Person <Sample@Example.com>"))
        assert reason is None
        assert set(row) == {
            "id", "thrid", "date", "direction", "labels", "from", "sender", "is_automated",
            "to", "subject", "body", "n_words", "body_source", "n_attachments",
            "attachments", "offset",
        }
        assert row["sender"] == "sample@example.com"
        assert row["direction"] == "inbound"
        assert row["date"] == "2025-09-01T10:00:00+00:00"
        assert row["n_words"] == 7

    def test_sent_label_is_outbound(self, tmp_path):
        row, _ = _row(tmp_path, _msg("Yes we do.", labels="Sent"))
        assert row["direction"] == "outbound"

    @pytest.mark.parametrize("label", ["Spam", "Trash"])
    def test_dropped_labels(self, tmp_path, label):
        assert _row(tmp_path, _msg("hello", labels=label)) == (None, "dropped_label")

    @pytest.mark.parametrize("sender,headers", [
        ("customer@example.com", ["List-Unsubscribe: <mailto:u@example.com>"]),
        ("customer@example.com", ["Precedence: bulk"]),
        ("customer@example.com", ["Auto-Submitted: auto-generated"]),
        ("noreply@example.com", []),
        ("deals@marketing.example.com", []),
        ("leads@bark.com", []),
    ], ids=["list-unsubscribe", "precedence", "auto-submitted", "noreply", "marketing", "bark"])
    def test_automated_mail_dropped(self, tmp_path, sender, headers):
        assert _row(tmp_path, _msg("hello", sender=sender, headers=headers)) == (None, "automated")

    def test_weddingwire_lead_kept_and_unwrapped(self, tmp_path):
        #WeddingWire is an active lead source --> must get past the automated filter
        lead = "Sample Person sent you a new message:\n\nDo you travel for weddings?\n\nFor: Example Troupe"
        row, reason = _row(tmp_path, _msg(lead, sender="messages@weddingwire.com"))
        assert reason is None
        assert row["body"] == "Do you travel for weddings?"

    def test_keep_automated_flags_instead_of_dropping(self, tmp_path):
        row, reason = _row(tmp_path, _msg("Big sale", sender="noreply@example.com"),
                           keep_automated=True)
        assert reason is None
        assert row["is_automated"] is True

    def test_quoted_only_reply_is_empty_body(self, tmp_path):
        assert _row(tmp_path, _msg("> what you said before")) == (None, "empty_body")

    def test_message_without_headers_is_dropped(self, tmp_path):
        path = tmp_path / "test.mbox"
        path.write_bytes(b"From 1000@xxx Mon Sep 01 10:00:00 +0000 2025\n")
        assert build_row(next(iter_messages(str(path)))) == (None, "no_headers")


# run (end to end)

class TestRun:

    def test_mixed_mailbox(self, tmp_path):
        stats, rows = _run(_mbox(tmp_path, *MIXED), tmp_path / "out")
        assert stats == {"read": 6, "written": 3, "inbound": 2, "outbound": 1,
                         "dropped_label": 1, "automated": 1, "dup_id": 1}
        assert [r["id"] for r in rows] == ["<q1@example.com>", "<a1@example.com>", "<q2@example.com>"]

    def test_dup_body_ignores_case_and_whitespace(self, tmp_path):
        mbox = _mbox(tmp_path, _msg("Do you have  availability?"), _msg("do you have availability?"))
        stats, rows = _run(mbox, tmp_path / "out")
        assert stats["dup_body"] == 1
        assert len(rows) == 1

    def test_shards_roll_over_at_shard_size(self, tmp_path):
        mbox = _mbox(tmp_path, *(_msg(f"Question number {i}") for i in range(5)))
        _run(mbox, tmp_path / "out", "--shard-size", "2")
        sizes = [sum(1 for _ in open(p)) for p in shard_paths(str(tmp_path / "out"))]
        assert sizes == [2, 2, 1]

    def test_resume_matches_uninterrupted_run(self, tmp_path):
        #a clean stop (--limit) resumes from the checkpoint written at the end of the run
        mbox = _mbox(tmp_path, *MIXED)
        full_stats, full_rows = _run(mbox, tmp_path / "full", "--shard-size", "2")
        _run(mbox, tmp_path / "resumed", "--shard-size", "2", "--limit", "3")
        stats, rows = _run(mbox, tmp_path / "resumed", "--shard-size", "2")
        assert stats == full_stats
        assert [r["id"] for r in rows] == [r["id"] for r in full_rows]

    def test_resume_after_crash_rewrites_unflushed_rows(self, tmp_path, monkeypatch):
        #a crash skips the final flush/checkpoint --> resume starts from the last shard close,
        #so the row still in the buffer when it crashed must be written again, not marked a duplicate
        mbox = _mbox(tmp_path, *(_msg(f"Question number {i}") for i in range(5)))
        full_stats, full_rows = _run(mbox, tmp_path / "full")
        calls = itertools.count(1)
        real_build_row = extract_mbox.build_row

        def crash_on_fourth(builder, **kw):
            if next(calls) == 4:
                raise RuntimeError("simulated crash")
            return real_build_row(builder, **kw)

        monkeypatch.setattr(extract_mbox, "build_row", crash_on_fourth)
        with pytest.raises(RuntimeError):
            main(["--mbox", str(mbox), "--outdir", str(tmp_path / "resumed"), "--shard-size", "2"])
        monkeypatch.undo()
        stats, rows = _run(mbox, tmp_path / "resumed", "--shard-size", "2")
        assert stats == full_stats
        assert [r["id"] for r in rows] == [r["id"] for r in full_rows]

    def test_resume_after_ctrl_c_rereads_interrupted_message(self, tmp_path, monkeypatch):
        #Ctrl+C can land mid-message --> resume must re-read that message, not skip it
        #read overcounts by one per interrupt, so compare rows rather than stats
        mbox = _mbox(tmp_path, *MIXED)
        _, full_rows = _run(mbox, tmp_path / "full")
        calls = itertools.count(1)
        real_build_row = extract_mbox.build_row

        def interrupt_on_last(builder, **kw):
            if next(calls) == len(MIXED):
                raise KeyboardInterrupt
            return real_build_row(builder, **kw)

        monkeypatch.setattr(extract_mbox, "build_row", interrupt_on_last)
        _run(mbox, tmp_path / "resumed")
        monkeypatch.undo()
        _, rows = _run(mbox, tmp_path / "resumed")
        assert [r["id"] for r in rows] == [r["id"] for r in full_rows]

    def test_restart_ignores_previous_run(self, tmp_path):
        #a stale seen.log would mark every row as a duplicate
        mbox = _mbox(tmp_path, *MIXED)
        full, _ = _run(mbox, tmp_path / "out")
        again, rows = _run(mbox, tmp_path / "out", "--restart")
        assert again == full
        assert len(rows) == full["written"]

    def test_limit_counts_across_resumed_runs(self, tmp_path):
        #Documents a known quirk rather than asserting it is desirable
        #--limit compares against the total read restored from the checkpoint, not this run's count
        mbox = _mbox(tmp_path, *MIXED)
        first, _ = _run(mbox, tmp_path / "out", "--limit", "2")
        second, _ = _run(mbox, tmp_path / "out", "--limit", "2")
        assert (first["read"], second["read"]) == (2, 3)

    def test_missing_mbox_returns_error(self, tmp_path):
        assert main(["--mbox", str(tmp_path / "missing.mbox"), "--outdir", str(tmp_path / "out")]) == 1
