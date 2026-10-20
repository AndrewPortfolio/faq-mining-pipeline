# Tests for the test holdout: the split rule (utils/holdout.py) and its stage (pipelines/holdout.py)

# Fixtures are synthetic emails written to tmp_path --> no real data, nothing in git

import csv
import io
import json
import re

import pytest

from faq_mining.pipelines import holdout
from faq_mining.pipelines.holdout import is_auto_reply, is_reply, main, near_duplicates, pick_query
from faq_mining.utils.holdout import is_held_out, rank

# Under the default seed these hash into the holdout and t1-t9 don't
HELD = ["t20", "t28", "t37", "t42", "t53", "t66", "t77", "t83", "t91"]
# Real thread ids are 19 digits, the shape check looks for
HELD_THRID, KEPT_THRID = "1790000000000000000", "1790000000000000001"

QUESTION = "How much does a lion dance cost for a wedding?"
INQUIRY = "We're looking for a quote for a lion dance at our wedding on May 5."  # asks, with no "?"
VIETNAMESE = "Xin chào, cho tôi hỏi giá múa lân cho đám cưới vào tháng năm."


def _email(n, thrid, body, *, direction="inbound", sender="client", subject="Lion dance", day=1):
    return {
        "id": f"{n:016x}",
        "thrid": thrid,
        "date": f"2025-01-{day:02d}T10:00:00+00:00",
        "direction": direction,
        "sender": sender,
        "subject": subject,
        "body": body,
    }


# The split


def test_the_split_is_pinned_to_its_seed():
    # a changed hash would quietly hold out other threads than the ones Stage 2 skipped
    assert [f"t{n}" for n in range(100) if is_held_out(f"t{n}")] == HELD
    assert [f"t{n}" for n in range(100) if is_held_out(f"t{n}", seed=7)] != HELD


def test_about_a_tenth_is_held_out():
    share = sum(is_held_out(f"{n:019d}") for n in range(10_000)) / 10_000
    assert 0.09 < share < 0.11


# Choosing a query


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("Re: Lion dance", True),
        ("RE:Lion dance", True),
        ("Fwd: Re: Lion dance", True),
        ("Fwd: Lion dance", False),  # a client forwarding something to ask about it opens a thread
        ("Regarding our wedding", False),
        ("", False),
    ],
)
def test_is_reply(subject, expected):
    assert is_reply(subject) is expected


@pytest.mark.parametrize(
    "subject, text, expected",
    [
        ("Automatic reply: Lion dance", "I am away.", True),
        ("Lion dance", "I am out of the office until Monday.", True),
        ("Lion dance", QUESTION, False),
    ],
)
def test_is_auto_reply(subject, text, expected):
    assert is_auto_reply(subject, text) is expected


@pytest.mark.parametrize(
    "email, reason",
    [
        (_email(1, "t20", INQUIRY), ""),  # no "?" needed: every query is read by hand anyway
        (_email(1, "t20", QUESTION, subject="Re: Lion dance"), "reply"),
        (_email(1, "t20", "I am out of the office until Monday, can it wait?"), "auto_reply"),
        (_email(1, "t20", "Call me at 714-555-1234 about the wedding please"), "tripwire"),
        (_email(1, "t20", "Thanks <PERSON>!"), "no_content"),
        (_email(1, "t20", "Here is your quote.", direction="outbound", sender="atld"), "no_client_email"),
        ({**_email(1, "t20", QUESTION), "date": None}, "undated"),
    ],
)
def test_the_coarse_filter(email, reason):
    query, why = pick_query([email], {"atld"})
    assert why == reason
    assert (query is None) is bool(reason)


def test_the_query_is_the_first_client_email_and_a_draft_isnt_one():
    draft = _email(1, "t20", "Thank you for choosing us! This is the confirmation email.", sender="atld")
    asked = _email(2, "t20", INQUIRY, day=2)
    later = _email(3, "t20", "Also, is parking free at the venue?", day=3)
    query, _ = pick_query([later, asked, draft], {"atld"})  # any order: it sorts by date
    assert (query["id"], query["kind"], query["lang"]) == (asked["id"], "opener", "en")  # the draft was never sent


def test_a_client_email_after_atld_wrote_is_a_follow_up():
    sent = _email(1, "t20", "Here is the quote you asked for.", direction="outbound", sender="atld")
    asked = _email(2, "t20", "Can we pay the deposit by card?", subject="Quote", day=2)
    query, _ = pick_query([sent, asked], {"atld"})
    assert (query["id"], query["kind"]) == (asked["id"], "follow_up")


# Near-duplicates

FORM = "Name: <PERSON> Email: <EMAIL_ADDRESS> Message (Please include your date, time and location):"
OWN = "we are planning a lion dance for the grand opening of our new bakery next month with two lions"


def test_a_near_copy_in_another_thread_is_a_near_duplicate(monkeypatch):
    monkeypatch.setattr(holdout, "BOILERPLATE_EMAILS", 2)  # so three form messages share a template
    emails = [
        _email(1, "t20", f"{FORM} {OWN}"),  # a query
        _email(2, "t20", OWN, direction="outbound", sender="atld", day=2),  # its own thread doesn't count
        _email(3, "t28", f"{FORM} could you perform at our school assembly in the spring"),  # a query
        _email(4, "t1", f"{FORM} could you perform at our school assembly in the fall"),  # the same client, again
        _email(5, "t2", f"{FORM} do you also rent out costumes for a family photo shoot"),
    ]
    threads = {}
    for email in emails:
        threads.setdefault(email["thrid"], []).append(email)
    queries = [{"id": emails[0]["id"], "thrid": "t20"}, {"id": emails[2]["id"], "thrid": "t28"}]
    # the bakery one shares only the form with other threads: not a copy. The school one shares all its runs but
    # the last, 9 of 10 (4 cross from the form into its own words)
    assert near_duplicates(queries, threads) == {emails[2]["id"]: (emails[3]["id"], 0.9)}


# Whole run


def _write(tmp_path, emails):
    indir = tmp_path / "redacted"
    indir.mkdir()
    (indir / "emails-00000.jsonl").write_text("".join(json.dumps(e) + "\n" for e in emails), encoding="utf-8")
    return indir


def _queries(tmp_path, indir, *extra):
    outdir = tmp_path / "holdout"
    return main(["queries", "--indir", str(indir), "--outdir", str(outdir), *extra]), outdir


def _labels(outdir):
    return list(csv.DictReader(io.StringIO((outdir / "labels.csv").read_text(encoding="utf-8"))))


EMAILS = [
    _email(1, "t1", QUESTION),  # kept: Stage 2 embeds it, so it's never a test query
    _email(2, "t20", f"Hi!\n\n{INQUIRY}"),
    _email(3, "t28", QUESTION, subject="Re: Lion dance"),
    _email(4, "t37", "Here is the quote you asked for.", direction="outbound", sender="atld"),
    _email(5, "t37", "Can we pay the deposit by card?", subject="Quote", day=2),
    _email(6, "t42", VIETNAMESE),
]


def test_lists_each_held_out_threads_first_client_email_to_label(tmp_path):
    code, outdir = _queries(tmp_path, _write(tmp_path, EMAILS))
    assert code == 0
    rows = _labels(outdir)
    assert list(rows[0]) == holdout.LABEL_COLUMNS
    # random but fixed order: the top rows are the test, the rest the reserve
    assert [row["thrid"] for row in rows] == sorted(["t20", "t37", "t42"], key=rank)
    by_thread = {row["thrid"]: row for row in rows}
    assert (by_thread["t20"]["kind"], by_thread["t20"]["text"]) == ("opener", f"Hi! {INQUIRY}")  # on one line
    assert (by_thread["t37"]["kind"], by_thread["t37"]["id"]) == ("follow_up", EMAILS[4]["id"])
    assert by_thread["t42"]["lang"] == "vi"
    assert {row["faq"] for row in rows} == {""}  # labeled by hand
    run = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert (run["held_out_threads"], run["reply"], run["near_duplicate"], run["rows"]) == (4, 1, 0, 3)


def test_labels_are_written_once(tmp_path):
    indir = _write(tmp_path, EMAILS)
    code, outdir = _queries(tmp_path, indir)
    first = (outdir / "labels.csv").read_bytes()
    assert _queries(tmp_path, indir)[0] == 1  # it may hold labels by now
    assert _queries(tmp_path, indir, "--force")[0] == 0
    assert (outdir / "labels.csv").read_bytes() == first


# Leak check


def test_check_counts_held_out_ids_in_any_output(tmp_path, capsys):
    held, kept = _email(2, HELD_THRID, INQUIRY), _email(1, KEPT_THRID, QUESTION)
    indir = _write(tmp_path, [kept, held])
    out = tmp_path / "out"
    for folder in ("embeddings", "faqs", "clusters"):
        (out / folder).mkdir(parents=True)
    chunk = {"chunk_id": f"{held['id']}-0", "id": held["id"], "thrid": HELD_THRID, "embedding": [0.5, 0.25]}
    (out / "embeddings" / "emails-00000.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    (out / "faqs" / "read_by_hand.txt").write_text(f"=== thread {HELD_THRID}\n", encoding="utf-8")
    (out / "clusters" / "report.txt").write_text(f"=== thread {KEPT_THRID} | {kept['id']}-0\n", encoding="utf-8")

    dirs = [str(out / folder) for folder in ("embeddings", "faqs", "clusters")]
    assert main(["check", "--indir", str(indir), *dirs]) == 1
    printed = capsys.readouterr().out
    assert re.search(r"emails-00000\.jsonl\s+3\n", printed)  # the chunk id, the id and the thread
    assert re.search(r"read_by_hand\.txt\s+1\n", printed)
    assert "report.txt" not in printed
    assert main(["check", "--indir", str(indir), str(out / "clusters")]) == 0
