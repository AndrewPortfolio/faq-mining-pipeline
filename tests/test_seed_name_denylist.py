#Tests for seed_name_denylist (Stage 1 prep)

#Fixtures are synthetic header rows written to tmp_path --> no real PII, nothing in git
#The word list is a tiny fake so results don't depend on the machine's /usr/share/dict/words


import json
import unicodedata
from collections import Counter

from seed_name_denylist import (MIN_LOWERCASE_EMAILS, english_in_use, fold, harvest, main,
                                name_parts, split_entries)


def _row(from_="customer@example.com", to="shop@example.com", body=""):
    return {"from": from_, "to": to, "subject": "Question", "body": body}


def _shards(tmp_path, *rows):
    indir = tmp_path / "extracted"
    indir.mkdir()
    (indir / "emails-00000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return indir


def _run(tmp_path, indir, *extra):
    words = tmp_path / "words"
    words.write_text("do\nlong\nphotography\nChan\n", encoding="utf-8")
    out = tmp_path / "pii" / "name_denylist.txt"
    code = main(["--indir", str(indir), "--out", str(out), "--dict", str(words), *extra])
    return code, out


def _entries(path):
    #section -> {name: count}; header comments come before the first section so they're skipped
    sections, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            current = sections.setdefault(line.strip("[]"), {})
        elif current is not None and line.strip():
            name, count = line.split("#")
            current[name.strip()] = int(count)
    return sections


# name parsing

class TestNameParsing:

    def test_fold_strips_vietnamese_diacritics(self):
        assert fold("Nguyễn Đức Trần") == "Nguyen Duc Tran"

    def test_name_parts_normalises_case_and_drops_initials(self):
        assert name_parts('"NGUYEN, Mei-Ling J."') == ["Nguyen", "Mei-Ling"]

    def test_name_parts_handles_decomposed_unicode(self):
        #some mail clients send NFD, which would otherwise split "Trần" mid-word
        assert name_parts(unicodedata.normalize("NFD", "Trần Thảo")) == ["Trần", "Thảo"]


# harvest

class TestHarvest:

    def test_takes_every_part_of_a_name_with_a_known_surname(self):
        assert harvest([_row(from_='"Thao Nguyen" <thao@example.com>')]) == {"Thao": 1, "Nguyen": 1}

    def test_ignores_names_without_a_known_surname(self):
        assert harvest([_row(from_='"John Smith" <john@example.com>')]) == {}

    def test_keeps_diacritic_and_folded_spellings(self):
        counts = harvest([_row(from_='"Trần Thị Mai" <mai@example.com>')])
        assert set(counts) == {"Trần", "Thị", "Mai", "Tran", "Thi"}

    def test_reads_every_recipient(self):
        row = _row(to='"Thao Nguyen" <a@example.com>, "Bob Lee" <b@example.com>')
        assert set(harvest([row])) == {"Thao", "Nguyen", "Bob", "Lee"}

    def test_counts_each_email_once(self):
        #a name in both From and To of one email is still one email
        row = _row(from_='"Thao Nguyen" <a@example.com>', to='"Thao Nguyen" <b@example.com>')
        assert harvest([row, row])["Nguyen"] == 2


# english_in_use

class TestEnglishInUse:

    def test_needs_dictionary_and_lowercase_use(self):
        #"wang" is only in the dictionary, "nguyen" is only used lowercase --> neither is ambiguous
        rows = [_row(body="do you have a song list for nguyen?")] * MIN_LOWERCASE_EMAILS
        assert english_in_use({"do", "song", "wang"}, rows) == {"do", "song"}

    def test_threshold_counts_emails_not_occurrences(self):
        rows = [_row(body="do do do")]
        assert english_in_use({"do"}, rows * (MIN_LOWERCASE_EMAILS - 1)) == set()
        assert english_in_use({"do"}, rows * MIN_LOWERCASE_EMAILS) == {"do"}

    def test_ignores_addresses_handles_and_urls(self):
        body = "song.bird@example.com, @songbird, https://example.com/song"
        rows = [_row(body=body)] * MIN_LOWERCASE_EMAILS
        assert english_in_use({"song", "songbird", "example", "com"}, rows) == set()


# split_entries

class TestSplitEntries:

    def test_seed_surnames_kept_without_hits_and_sorted_by_count(self):
        names, _ = split_entries(Counter({"Thao": 3, "Nguyen": 5}), {"Nguyen", "Wong"}, set())
        assert names == [("Nguyen", 5), ("Thao", 3), ("Wong", 0)]

    def test_english_words_go_to_ambiguous(self):
        names, ambiguous = split_entries(Counter({"Long": 2}), {"Do", "Nguyen"}, {"do", "long"})
        assert names == [("Nguyen", 0)]
        assert ambiguous == [("Long", 2), ("Do", 0)]


# run

class TestRun:

    def test_writes_names_and_ambiguous_sections(self, tmp_path):
        chatter = [_row(body="how long do you need for photography?")] * MIN_LOWERCASE_EMAILS
        indir = _shards(tmp_path, _row(from_='"Long Tran Photography" <x@example.com>'), *chatter)
        code, out = _run(tmp_path, indir)
        assert code == 0
        entries = _entries(out)
        assert entries["names"]["Tran"] == 1
        #"Chan" is capitalized in the word list --> a proper noun, not an English word
        assert entries["names"]["Chan"] == 0
        assert entries["ambiguous"] == {"Long": 1, "Photography": 1, "Do": 0}

    def test_refuses_to_overwrite_edits_without_force(self, tmp_path):
        indir = _shards(tmp_path, _row())
        out = tmp_path / "pii" / "name_denylist.txt"
        out.parent.mkdir()
        out.write_text("my edits\n", encoding="utf-8")
        assert _run(tmp_path, indir)[0] == 1
        assert out.read_text(encoding="utf-8") == "my edits\n"
        assert _run(tmp_path, indir, "--force")[0] == 0
        assert "[names]" in out.read_text(encoding="utf-8")

    def test_missing_shards_is_an_error(self, tmp_path):
        indir = tmp_path / "empty"
        indir.mkdir()
        code, out = _run(tmp_path, indir)
        assert code == 1
        assert not out.exists()
