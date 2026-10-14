#Tests for triage (Stage 4)

#Fixtures are synthetic chunks and Stage 3 files written to tmp_path --> no real data, nothing in git

import csv
import json

import pytest

from triage import decide, is_vietnamese, main, signals

MODEL = "nomic-embed-text@0a109f422b47"
VIETNAMESE = "Xin chào anh, tôi muốn đặt múa lân cho đám cưới của chúng tôi được không?"
FORM = "Name: <PERSON>\nEmail: <EMAIL_ADDRESS>\nPhone: <PHONE_NUMBER>\nMessage (Please include o: How much is it?"
FOOTER = "See you then!\nCONFIDENTIALITY NOTICE: This e-mail is for the sole use of the intended recipient."
QUESTION = "How much does a lion dance cost for a wedding?"
THANKS = "Thanks so much, sounds good!"


@pytest.mark.parametrize("texts, terms, expected", [
    ([VIETNAMESE] * 3 + [QUESTION], [], ("skip", "non_english")),
    ([FORM] * 3 + [QUESTION], [], ("skip", "template")),
    ([FOOTER, QUESTION, QUESTION], [], ("skip", "footer")),
    ([QUESTION] * 3, ["div", "div style", "line height", "wedding"], ("review", "markup")),
    ([THANKS] * 9 + [QUESTION], [], ("review", "few_questions")),
    ([QUESTION, THANKS], ["lion dance", "wedding"], ("faq", "")),
])
def test_decide(texts, terms, expected):
    assert decide(signals(texts, terms)) == expected


def test_a_real_question_saying_confidential_is_not_a_footer():
    assert signals(["Can you keep our budget confidential?"], [])["footer_share"] == 0


def test_two_markup_terms_are_not_enough():
    assert decide(signals([QUESTION], ["div", "font size", "wedding"])) == ("faq", "")


@pytest.mark.parametrize("text, expected", [
    (VIETNAMESE, True),
    ("Café crème at the château?", False),
    (QUESTION, False),
])
def test_is_vietnamese(text, expected):
    assert is_vietnamese(text) is expected


# Whole run

def _row(n, text, *, direction="inbound", thrid=None, date="2025-01-01T10:00:00+00:00"):
    return {"chunk_id": f"{n:016x}-0", "id": f"{n:016x}", "thrid": thrid or f"t{n}", "date": date,
            "direction": direction, "subject": "Question", "chunk_index": 0, "n_chunks": 1,
            "n_words": len(text.split()), "text": text, "task": "clustering", "model": MODEL,
            "embedding": [1.0, 0.0]}


def _setup(tmp_path, drop_assignment=False):
    # cluster 0 is the template, 1 the FAQ, 2 the Vietnamese one, and one Vietnamese chunk sits in noise
    groups = {0: [FORM] * 4, 1: [QUESTION] * 4, 2: [VIETNAMESE] * 4, -1: [VIETNAMESE]}
    rows, label, n = [], {}, 0
    for k, texts in groups.items():
        for text in texts:
            n += 1
            rows.append(_row(n, text))
            label[rows[-1]["chunk_id"]] = k
    rows.append(_row(900, "Yes we can, it is $300.", direction="outbound", thrid="t9",
                     date="2025-01-02T10:00:00+00:00"))
    indir = tmp_path / "embeddings"
    indir.mkdir()
    (indir / "emails-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    clusters = tmp_path / "clusters"
    clusters.mkdir()
    assigned = [{"chunk_id": cid, "cluster": k} for cid, k in label.items()][int(drop_assignment):]
    (clusters / "assignments.jsonl").write_text("".join(json.dumps(a) + "\n" for a in assigned), encoding="utf-8")
    with open(clusters / "clusters.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["cluster", "n_chunks", "n_emails", "n_threads", "top_terms", "representatives"])
        for k in (0, 1, 2):
            reps = [cid for cid, c in label.items() if c == k][:2]
            writer.writerow([k, 4, 4, 4, "some; terms", " ".join(reps)])
    (clusters / "run.json").write_text(json.dumps({"model": MODEL, "min_cluster_size": 15}), encoding="utf-8")
    return indir, clusters


def _run(tmp_path, **kw):
    indir, clusters = _setup(tmp_path, **kw)
    outdir = tmp_path / "faqs"
    code = main(["--indir", str(indir), "--clusters", str(clusters), "--outdir", str(outdir)])
    return code, outdir


def test_writes_one_decision_per_cluster(tmp_path):
    code, outdir = _run(tmp_path)
    assert code == 0
    with open(outdir / "triage.csv", encoding="utf-8") as fh:
        decisions = {int(r["cluster"]): (r["decision"], r["reason"]) for r in csv.DictReader(fh)}
    assert decisions == {0: ("skip", "template"), 1: ("faq", ""), 2: ("skip", "non_english")}
    assert json.loads((outdir / "run.json").read_text())["decisions"] == {
        "skip template": 1, "faq": 1, "skip non_english": 1}


def test_candidates_list_faqs_in_full_and_skips_one_line_each(tmp_path):
    _, outdir = _run(tmp_path)
    text = (outdir / "candidates.txt").read_text(encoding="utf-8")
    faqs, skipped = text.split("##### Skipped")
    assert "=== cluster 1 |" in faqs and QUESTION in faqs
    assert "=== cluster 0 |" not in faqs and FORM.split("\n")[0] not in text
    assert "cluster 0 | 4 chunks | template" in skipped
    assert "cluster 2 | 4 chunks | non_english" in skipped


def test_every_vietnamese_chunk_is_listed_to_read_by_hand(tmp_path):
    # the one in noise too: the model put them together by language, so the cluster isn't the point
    _, outdir = _run(tmp_path)
    text = (outdir / "read_by_hand.txt").read_text(encoding="utf-8")
    assert text.count(VIETNAMESE) == 5
    assert "cluster -1]" in text
    assert QUESTION not in text
    assert text.count("reply: Yes we can, it is $300.") == 1


def test_stale_clusters_are_refused(tmp_path):
    # assignments from another embedding run would triage the wrong texts under these numbers
    code, outdir = _run(tmp_path, drop_assignment=True)
    assert code == 1
    assert not outdir.exists()
