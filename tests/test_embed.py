#Tests for embed (Stage 2)

#Fixtures are synthetic emails written to tmp_path --> no real PII, nothing in git
#Ollama is faked by patching post_json (or urlopen, for the retry tests), so no server is needed

import csv
import io
import json
import math
import urllib.error

import pytest

import embed
from embed import (MAX_ATTEMPTS, PREFIXES, TARGET_WORDS, ContextOverflow, chunk_text, embed_pieces,
                   has_content, main, pii_hits, post_json)
from shared.pipeline import Stats

DIGEST = "0a109f422b47e3a3" + "0" * 48


def _sentences(n_words, tag="word", per_sentence=10):
    # n_words of filler prose with a full stop every per_sentence words
    return " ".join(f"{tag}{i}" + ("." if (i + 1) % per_sentence == 0 else "") for i in range(n_words))


def _sizes(chunks):
    return [len(chunk.split()) for chunk in chunks]


# Chunking

def test_short_body_is_one_unchanged_chunk():
    body = "Hi <PERSON>,\n\nHow much is a lion dance for a wedding?\n\nThanks"
    assert chunk_text(f"  {body}\n") == [body]


def test_long_body_splits_into_even_chunks():
    body = _sentences(650)
    chunks = chunk_text(body)
    assert _sizes(chunks) == [220, 220, 210]
    assert " ".join(chunks).split() == body.split()


def test_chunks_break_between_paragraphs():
    a, b, c = (_sentences(120, tag) for tag in "abc")
    assert chunk_text(f"{a}\n\n{b}\n\n{c}") == [f"{a}\n\n{b}", c]


def test_unpunctuated_text_splits_into_even_word_windows():
    assert _sizes(chunk_text(" ".join(f"w{i}" for i in range(650)))) == [217, 217, 216]


def test_no_chunk_is_over_the_target_and_no_word_is_lost():
    body = "\n\n".join([_sentences(290, "a"), _sentences(40, "b"), _sentences(500, "c"), "d1 d2"])
    chunks = chunk_text(body)
    assert max(_sizes(chunks)) <= TARGET_WORDS
    assert " ".join(chunks).split() == body.split()


def test_blank_body_has_no_chunks():
    assert chunk_text("") == []
    assert chunk_text(" \n\n ") == []


@pytest.mark.parametrize("text, expected", [
    ("", False),
    ("<PERSON>", False),
    ("Thanks <PERSON>!", False),
    ("Thank you so much", True),
    ("How much for <DATE_TIME> at <LOCATION>?", True),
])
def test_has_content(text, expected):
    assert has_content(text) is expected


# PII tripwire

@pytest.mark.parametrize("text, entity", [
    ("call me at (714)\n555-1234 tonight", "PHONE_NUMBER"),
    ("my cell is 714-555-1234", "PHONE_NUMBER"),
    ("or +1 714.555.1234", "PHONE_NUMBER"),
    ("write to thao@example.com", "EMAIL_ADDRESS"),
    ("photos at https://example.com/album", "URL"),
    ("see www.example.com", "URL"),
    ("follow us @liondance_team", "SOCIAL_HANDLE"),
])
def test_tripwire_fires(text, entity):
    assert [hit["entity_type"] for hit in pii_hits({"subject": "", "body": text})] == [entity]


@pytest.mark.parametrize("text", [
    "[cid:image001.png@01D5A2B3.C4D5E6F0]",
    "ceremony starts @5pm",
    "call <PHONE_NUMBER> or email <EMAIL_ADDRESS>, see <URL>",
    "a $1,500 deposit for 10:30-11:30 on 2024-10-05",
    "order 1234567890",
])
def test_tripwire_ignores(text):
    assert pii_hits({"subject": text, "body": text}) == []


def test_tripwire_reports_where_not_what():
    hits = pii_hits({"subject": "Re: dates", "body": "text me 714-555-1234"})
    assert hits == [{"field": "body", "entity_type": "PHONE_NUMBER", "start": 8, "end": 20}]


# Resplitting

def _overflow_above(limit, calls=None):
    # like Ollama, the whole batch fails when any one input is over the limit
    def fake(texts):
        if calls is not None:
            calls.append(len(texts))
        if any(len(text.split()) > limit for text in texts):
            raise ContextOverflow("the input length exceeds the context length")
        return [[float(len(text.split()))] for text in texts]
    return fake


def test_overflowing_input_is_halved_until_it_fits():
    text = " ".join(f"w{i}" for i in range(100))
    stats = Stats()
    [pieces] = embed_pieces([text], _overflow_above(40), stats)
    assert _sizes(t for t, _ in pieces) == [25, 25, 25, 25]
    assert " ".join(t for t, _ in pieces).split() == text.split()
    assert stats["resplits"] == 3


def test_batch_overflow_redoes_inputs_one_at_a_time():
    short_a, too_long, short_b = "a b c", " ".join(["w"] * 60), "d e f"
    calls = []
    result = embed_pieces([short_a, too_long, short_b], _overflow_above(40, calls), Stats())
    assert result[0] == [(short_a, [3.0])] and result[2] == [(short_b, [3.0])]
    assert _sizes(t for t, _ in result[1]) == [30, 30]
    assert calls[0] == 3 and set(calls[1:]) == {1}


def test_one_word_that_overflows_is_an_error():
    with pytest.raises(ContextOverflow):
        embed_pieces(["x" * 5000], _overflow_above(0), Stats())


# Talking to Ollama

class _Urlopen:
    # stands in for urllib.request.urlopen: raises or answers with each outcome in turn
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(json.dumps(outcome).encode())


def _http_error(code, body):
    return urllib.error.HTTPError("http://x/api/embed", code, "error", {}, io.BytesIO(body.encode()))


def _patch_urlopen(monkeypatch, *outcomes):
    urlopen = _Urlopen(*outcomes)
    monkeypatch.setattr(embed.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(embed.time, "sleep", lambda seconds: None)
    return urlopen


def test_network_errors_stop_after_max_attempts(monkeypatch):
    urlopen = _patch_urlopen(monkeypatch, *[urllib.error.URLError("refused")] * MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match=f"failed {MAX_ATTEMPTS} times"):
        post_json("http://x/api/tags")
    assert urlopen.calls == MAX_ATTEMPTS


def test_server_error_is_retried(monkeypatch):
    urlopen = _patch_urlopen(monkeypatch, _http_error(500, "busy"), {"models": []})
    assert post_json("http://x/api/tags") == {"models": []}
    assert urlopen.calls == 2


def test_context_overflow_is_not_retried(monkeypatch):
    urlopen = _patch_urlopen(
        monkeypatch, _http_error(400, '{"error":"the input length exceeds the context length"}'))
    with pytest.raises(ContextOverflow):
        post_json("http://x/api/embed", {"input": ["x"]})
    assert urlopen.calls == 1


def test_other_client_errors_are_not_retried(monkeypatch):
    urlopen = _patch_urlopen(monkeypatch, _http_error(404, '{"error":"model not found"}'))
    with pytest.raises(RuntimeError, match="model not found"):
        post_json("http://x/api/embed", {"input": ["x"]})
    assert urlopen.calls == 1


# End to end, with a fake Ollama

def _vector(text):
    # unit length and unique per input length, so a row's vector can be traced back to its text
    angle = len(text) / 1000
    return [math.cos(angle), math.sin(angle)]


class _FakeOllama:
    def __init__(self, digest=DIGEST):
        self.digest, self.inputs = digest, []

    def __call__(self, url, payload=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "nomic-embed-text:latest", "digest": self.digest}]}
        assert payload["truncate"] is False
        self.inputs.extend(payload["input"])
        return {"embeddings": [_vector(text) for text in payload["input"]]}


def _email(n, body):
    return {"id": f"{n:016x}", "thrid": f"t{n}", "date": "2025-09-01T10:00:00",
            "direction": "inbound", "labels": ["Inbox"], "sender": "a1b2", "to": [],
            "is_automated": False, "subject": f"Question {n}", "body": body,
            "n_words": len(body.split()), "body_source": "text/plain", "n_attachments": 0,
            "attachments": [], "offset": n, "n_redacted": 0, "signature_chars": 0}


EMAILS = [
    _email(1, "Hi, how much is a lion dance for a wedding reception?"),
    _email(2, _sentences(650)),                                 # 3 chunks
    _email(3, "Sure, call me at (714)\n555-1234 to book"),     # quarantined
    _email(4, "Thanks <PERSON>!"),                              # no content
]


def _setup(tmp_path, emails=EMAILS):
    indir = tmp_path / "redacted"
    indir.mkdir()
    (indir / "emails-00000.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in emails), encoding="utf-8")
    return indir, tmp_path / "embeddings"


def _run(monkeypatch, indir, outdir, *extra, ollama=None):
    ollama = ollama or _FakeOllama()
    monkeypatch.setattr(embed, "post_json", ollama)
    return main(["--indir", str(indir), "--outdir", str(outdir), *extra]), ollama


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_row_per_chunk_with_its_own_vector(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    code, ollama = _run(monkeypatch, indir, outdir)
    assert code == 0
    rows = _rows(outdir / "emails-00000.jsonl")
    one, two = EMAILS[0]["id"], EMAILS[1]["id"]
    assert [(r["id"], r["chunk_index"], r["n_chunks"]) for r in rows] == [
        (one, 0, 1), (two, 0, 3), (two, 1, 3), (two, 2, 3)]
    for row in rows:
        assert row["embedding"] == _vector(PREFIXES["clustering"] + row["text"])
        assert row["chunk_id"] == f"{row['id']}-{row['chunk_index']}"
        assert (row["task"], row["model"]) == ("clustering", "nomic-embed-text@0a109f422b47")
    assert all(text.startswith("clustering: ") for text in ollama.inputs)


def test_pii_hit_quarantines_the_email_without_writing_the_pii(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    _run(monkeypatch, indir, outdir)
    assert EMAILS[2]["id"] not in {r["id"] for r in _rows(outdir / "emails-00000.jsonl")}
    report = (outdir / "emails-00000.quarantine.csv").read_text(encoding="utf-8")
    assert list(csv.DictReader(io.StringIO(report))) == [
        {"shard": "emails-00000", "email_id": EMAILS[2]["id"], "field": "body",
         "entity_type": "PHONE_NUMBER", "start": "17", "end": "31"}]
    assert "555" not in report


def test_stats_count_what_was_skipped(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    _run(monkeypatch, indir, outdir)
    stats = json.loads((outdir / "checkpoint.json").read_text(encoding="utf-8"))["stats"]
    assert (stats["emails"], stats["chunks"], stats["quarantined"], stats["no_content"]) == (2, 4, 1, 1)


def test_search_document_task_uses_its_own_prefix(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    code, ollama = _run(monkeypatch, indir, outdir, "--task", "search_document")
    assert code == 0
    assert all(text.startswith("search_document: ") for text in ollama.inputs)
    assert {r["task"] for r in _rows(outdir / "emails-00000.jsonl")} == {"search_document"}


def test_rerun_skips_finished_shards(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    _run(monkeypatch, indir, outdir)
    before = (outdir / "emails-00000.jsonl").read_text(encoding="utf-8")
    code, ollama = _run(monkeypatch, indir, outdir)
    assert code == 0 and ollama.inputs == []
    assert (outdir / "emails-00000.jsonl").read_text(encoding="utf-8") == before


def test_a_different_model_needs_force(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    _run(monkeypatch, indir, outdir)
    repulled = _FakeOllama(digest="f" * 64)
    assert _run(monkeypatch, indir, outdir, ollama=repulled)[0] == 1
    assert repulled.inputs == []
    assert _run(monkeypatch, indir, outdir, "--force", ollama=repulled)[0] == 0
    assert repulled.inputs


def test_failure_keeps_finished_shards_for_the_rerun(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    (indir / "emails-00001.jsonl").write_text(
        json.dumps(_email(9, "Is parking free at the venue for the drummers?")) + "\n", encoding="utf-8")
    ollama = _FakeOllama()

    def down_for_parking(url, payload=None):
        if payload and any("parking" in text for text in payload["input"]):
            raise RuntimeError("http://x/api/embed failed 3 times")
        return ollama(url, payload)

    monkeypatch.setattr(embed, "post_json", down_for_parking)
    assert main(["--indir", str(indir), "--outdir", str(outdir)]) == 1
    assert (outdir / "emails-00000.jsonl").exists()
    assert not (outdir / "emails-00001.jsonl").exists()
    code, retry = _run(monkeypatch, indir, outdir)
    assert code == 0 and all("parking" in text for text in retry.inputs)
    assert (outdir / "emails-00001.jsonl").exists()


def test_ollama_down_fails_before_writing(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)

    def down(url, payload=None):
        raise RuntimeError("http://x/api/tags failed 3 times")

    monkeypatch.setattr(embed, "post_json", down)
    assert main(["--indir", str(indir), "--outdir", str(outdir)]) == 1
    assert not (outdir / "emails-00000.jsonl").exists()


def test_limit_writes_a_smoke_test_elsewhere(tmp_path, monkeypatch):
    indir, outdir = _setup(tmp_path)
    _run(monkeypatch, indir, outdir, "--limit", "1")
    assert not (outdir / "emails-00000.jsonl").exists()
    assert [r["id"] for r in _rows(outdir / "smoke" / "emails-00000.jsonl")] == [EMAILS[0]["id"]]
