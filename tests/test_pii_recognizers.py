#Tests for shared.pii_recognizers (Stage 1)

#Recognizers are exercised on their own (nlp_artifacts=None) so the suite never loads the trf model
#Fixtures are synthetic --> no real PII, nothing in git


from shared.pii_recognizers import (AMBIGUOUS_SCORE, NAME_SCORE, VENUE_SCORE, custom_recognizers,
                                    load_allowlist, load_denylist, load_venues, name_recognizers,
                                    pattern_recognizers, venue_recognizers)


def _hits(recognizer, text):
    results = recognizer.analyze(text, recognizer.supported_entities, nlp_artifacts=None)
    return sorted((text[r.start:r.end], round(r.score, 2)) for r in results)


def _by_entity(recognizers, entity):
    return next(r for r in recognizers if entity in r.supported_entities)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# list files

class TestLists:

    def test_denylist_sections_drop_counts_and_comments(self, tmp_path):
        path = _write(tmp_path, "names.txt", "# header comment\n"
                                             "[names]\nNguyen                  # 120\nThao # 4\n\n"
                                             "[ambiguous]\nDo                      # 6520\n")
        assert load_denylist(path) == (["Nguyen", "Thao"], ["Do"])

    def test_missing_denylist_is_not_fatal(self, tmp_path):
        #a first run can happen before the list is seeded; the built-in recognizers still work
        assert load_denylist(str(tmp_path / "absent.txt")) == ([], [])

    def test_allowlist_reads_every_line(self, tmp_path):
        path = _write(tmp_path, "allow.txt", "# never flag these\nWeddingWire\nThe Knot\n\n")
        assert load_allowlist(path) == ["WeddingWire", "The Knot"]


# name deny-list

class TestNameRecognizers:

    def test_names_match_any_case(self):
        #lowercase names are where trf fails ("pls send to huynh"), so [names] ignores case
        recognizer = name_recognizers(["Huynh"], [])[0]
        assert _hits(recognizer, "Huynh and huynh") == [("Huynh", NAME_SCORE), ("huynh", NAME_SCORE)]

    def test_ambiguous_names_need_a_capital(self):
        #otherwise "do" fires on every question in the inbox
        _, ambiguous = name_recognizers(["Nguyen"], ["Do"])
        assert _hits(ambiguous, "Do you do this?") == [("Do", AMBIGUOUS_SCORE)]

    def test_ambiguous_names_score_lower(self):
        names, ambiguous = name_recognizers(["Nguyen"], ["Do"])
        assert _hits(names, "Thao Nguyen") == [("Nguyen", NAME_SCORE)]
        assert _hits(ambiguous, "Do you know Do?") == [("Do", AMBIGUOUS_SCORE), ("Do", AMBIGUOUS_SCORE)]

    def test_partial_words_are_not_matched(self):
        recognizer = name_recognizers(["Le"], [])[0]
        assert _hits(recognizer, "Lemon Legend Le") == [("Le", NAME_SCORE)]

    def test_empty_list_makes_no_recognizer(self):
        assert name_recognizers([], []) == []


# regex recognizers

class TestStreetAddress:

    def test_matches_street_with_unit(self):
        recognizer = _by_entity(pattern_recognizers(), "STREET_ADDRESS")
        assert _hits(recognizer, "Ship to 1234 Oak St Apt 5, Santa Ana") == [("1234 Oak St Apt 5", 0.5)]

    def test_matches_lowercase_and_po_box(self):
        recognizer = _by_entity(pattern_recognizers(), "STREET_ADDRESS")
        assert _hits(recognizer, "mail it to po box 42") == [("po box 42", 0.6)]
        assert _hits(recognizer, "we're at 88 grand avenue") == [("88 grand avenue", 0.5)]

    def test_ignores_plain_numbers(self):
        recognizer = _by_entity(pattern_recognizers(), "STREET_ADDRESS")
        assert _hits(recognizer, "We need 3 hours and 250 chairs") == []


class TestSocialHandle:

    def test_matches_handles_but_not_email_addresses(self):
        recognizer = _by_entity(pattern_recognizers(), "SOCIAL_HANDLE")
        assert _hits(recognizer, "DM @thao.photo, not thao@example.com") == [("@thao.photo", 0.4)]

    def test_ignores_bare_at_sign(self):
        recognizer = _by_entity(pattern_recognizers(), "SOCIAL_HANDLE")
        assert _hits(recognizer, "meet @ 5pm @ the venue") == []


class TestVenues:

    def test_reads_file_and_matches_multi_word_names(self, tmp_path):
        path = _write(tmp_path, "venues.txt", "# venues\nRancho Las Lomas\n\nCasa Romantica\n")
        assert load_venues(path) == ["Rancho Las Lomas", "Casa Romantica"]
        recognizer = venue_recognizers(load_venues(path))[0]
        assert _hits(recognizer, "booked at rancho las lomas") == [("rancho las lomas", VENUE_SCORE)]

    def test_punctuation_matches_literally(self):
        #Presidio escapes deny-list entries itself, so "." must not behave as "any character"
        recognizer = venue_recognizers(["St. Regis"])[0]
        assert _hits(recognizer, "at St. Regis") == [("St. Regis", VENUE_SCORE)]
        assert _hits(recognizer, "at Stx Regis") == []

    def test_empty_list_makes_no_recognizer(self):
        assert venue_recognizers([]) == []


class TestCustomRecognizers:

    def test_bundles_names_venues_and_patterns(self, tmp_path):
        names = _write(tmp_path, "names.txt", "[names]\nNguyen # 3\n[ambiguous]\nDo # 9\n")
        venues = _write(tmp_path, "venues.txt", "Rancho Las Lomas\n")
        entities = {e for r in custom_recognizers(names, venues) for e in r.supported_entities}
        assert entities == {"PERSON", "LOCATION", "STREET_ADDRESS", "SOCIAL_HANDLE"}
