#Tests for seed_place_denylist (Stage 1 prep)

#Fixtures are synthetic emails, review files and word lists written to tmp_path --> no real PII


import csv
import json

from analyze_pii import COLUMNS
from seed_place_denylist import main
from shared.pii_recognizers import load_places


def _email(n, body):
    return {"id": f"<m{n}@example.com>", "thrid": f"t{n}", "subject": "Question", "body": body}


def _tag(n, body, text, recognizer="SpacyRecognizer"):
    #a review row: `recognizer` called the first `text` in `body` a LOCATION
    start = body.index(text)
    return {"email_id": f"<m{n}@example.com>", "thrid": f"t{n}", "field": "body", "entity_type": "LOCATION",
            "start": str(start), "end": str(start + len(text)), "text": text, "context": "", "score": "0.85",
            "recognizer": recognizer, "decision": "keep", "rule": "", "note": ""}


def _mentions(template, tagged, untagged, place, start=1):
    #`tagged` emails where trf called `place` a LOCATION, then `untagged` ones where it didn't
    emails, spans = [], []
    for i in range(tagged + untagged):
        body = template.format(place)
        emails.append(_email(start + i, body))
        if i < tagged:
            spans.append(_tag(start + i, body, place))
    return emails, spans


def _run(tmp_path, emails, spans, *extra, english=("orange",)):
    indir = tmp_path / "extracted"
    indir.mkdir(exist_ok=True)
    (indir / "emails-00000.jsonl").write_text("".join(json.dumps(e) + "\n" for e in emails), encoding="utf-8")
    reviewdir = tmp_path / "review"
    reviewdir.mkdir(exist_ok=True)
    with open(reviewdir / "emails-00000.spans.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(spans)
    words = tmp_path / "words.txt"
    words.write_text("".join(w + "\n" for w in english), encoding="utf-8")
    common = tmp_path / "common.txt"
    common.write_text("california\n", encoding="utf-8")
    out = tmp_path / "places.txt"
    code = main(["--indir", str(indir), "--reviewdir", str(reviewdir), "--out", str(out),
                 "--dict", str(words), "--common-location", str(common), *extra])
    return code, out


# which terms make the list

class TestSelection:

    def test_keeps_places_trf_tagged_in_most_mentions(self, tmp_path):
        emails, spans = _mentions("We perform in {} often", 2, 1, "Irvine")
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == (["irvine"], [])

    def test_drops_terms_trf_tagged_rarely(self, tmp_path):
        #"Lions" was called a place once in 18,537 mentions on the real inbox
        emails, spans = _mentions("The {} arrive at noon", 1, 3, "Lions")
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == ([], [])

    def test_needs_two_mentions(self, tmp_path):
        emails, spans = _mentions("Booked in {} today", 1, 0, "Temecula")
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == ([], [])

    def test_only_trf_votes(self, tmp_path):
        #the list's own hits from a past run can't count, or every listed term drifts to 100%
        emails, spans = _mentions("Booked in {} today", 1, 2, "Temecula")
        spans += [_tag(n, "Booked in Temecula today", "Temecula", recognizer="denylist_places") for n in (2, 3)]
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == ([], [])

    def test_region_keep_list_is_left_out(self, tmp_path):
        emails, spans = _mentions("We serve {} widely", 2, 0, "California")
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == ([], [])

    def test_short_terms_are_left_out(self, tmp_path):
        #"LA" and "CA" are the keep-list's job, and short terms match inside everything
        emails, spans = _mentions("Driving to {} soon", 2, 0, "LA")
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == ([], [])

    def test_multiword_places_count_across_line_breaks(self, tmp_path):
        #the second mention only reaches 2 (and 50%) if the line break still matches
        emails, spans = _mentions("Event in {} at noon", 1, 0, "Huntington Beach")
        emails.append(_email(9, "Event in Huntington\nBeach at noon"))
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == (["huntington beach"], [])

    def test_skips_shards_without_a_review_file(self, tmp_path):
        #an unanalyzed shard would count every mention as untagged
        (tmp_path / "extracted").mkdir()
        (tmp_path / "extracted" / "emails-00001.jsonl").write_text(
            "".join(json.dumps(_email(50 + n, "Irvine again")) + "\n" for n in range(5)), encoding="utf-8")
        emails, spans = _mentions("We perform in {} often", 2, 0, "Irvine")
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == (["irvine"], [])


# which section a term lands in

class TestSplit:

    def test_everyday_english_places_need_a_capital(self, tmp_path):
        #"orange" is a lion color in this inbox, so the list keeps only "Orange" the city
        emails, spans = _mentions("A school in the City of {} booked us", 12, 0, "Orange")
        colors = [_email(100 + n, "Eros has orange accents") for n in range(10)]
        _, out = _run(tmp_path, emails + colors, spans)
        assert load_places(str(out)) == ([], ["Orange"])

    def test_dictionary_word_used_rarely_stays_any_case(self, tmp_path):
        #in the dictionary, but never written lowercase here: not everyday English in this inbox
        emails, spans = _mentions("A school in the City of {} booked us", 2, 0, "Orange")
        _, out = _run(tmp_path, emails, spans)
        assert load_places(str(out)) == (["orange"], [])


# the file itself

class TestOutput:

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        emails, spans = _mentions("We perform in {} often", 2, 0, "Irvine")
        out = tmp_path / "places.txt"
        out.write_text("my edits\n", encoding="utf-8")
        code, _ = _run(tmp_path, emails, spans)
        assert code == 1 and out.read_text(encoding="utf-8") == "my edits\n"
        code, _ = _run(tmp_path, emails, spans, "--force")
        assert code == 0 and load_places(str(out)) == (["irvine"], [])

    def test_counts_are_written_as_comments(self, tmp_path):
        emails, spans = _mentions("We perform in {} often", 2, 1, "Irvine")
        _, out = _run(tmp_path, emails, spans)
        assert "# 2/3" in out.read_text(encoding="utf-8")
