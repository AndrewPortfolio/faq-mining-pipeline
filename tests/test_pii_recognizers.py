#Tests for shared.pii_recognizers (Stage 1)

#Recognizers are exercised on their own (nlp_artifacts=None) so the suite never loads the trf model
#Fixtures are synthetic --> no real PII, nothing in git


from presidio_analyzer.nlp_engine import NlpArtifacts
from spacy.tokens import Doc
from spacy.vocab import Vocab

from shared.pii_recognizers import (AMBIGUOUS_ALONE, AMBIGUOUS_IN_ENTITY, AMBIGUOUS_PAIR,
                                    AMBIGUOUS_RECOGNIZER, AMBIGUOUS_WORD, CARRIER_SCORE, EIN_SCORE,
                                    ENTITIES, NAME_SCORE, POLICY_SCORE, VENUE_SCORE, custom_recognizers,
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
        assert _hits(ambiguous, "Do you do this?") == [("Do", NAME_SCORE)]

    def test_ambiguous_hits_without_a_parse_fail_closed(self):
        #no trf read to go on --> scored like a name, so the row redacts instead of quietly keeping
        _, ambiguous = name_recognizers(["Nguyen"], ["Do"])
        assert ambiguous.name == AMBIGUOUS_RECOGNIZER
        assert _hits(ambiguous, "Do you know Do?") == [("Do", NAME_SCORE), ("Do", NAME_SCORE)]

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
        assert entities == {"PERSON", "LOCATION", "STREET_ADDRESS", "SOCIAL_HANDLE",
                            "US_EIN", "INSURANCE_POLICY"}


class TestEin:

    def test_matches_dashed_ein(self):
        recognizer = _by_entity(pattern_recognizers(), "US_EIN")
        assert _hits(recognizer, "Our school tax ID is 12-3456789.") == [("12-3456789", EIN_SCORE)]

    def test_ignores_phones_zips_and_bare_digits(self):
        #the bare 9-digit form belongs to US_SSN, which already flags it at 0.05
        recognizer = _by_entity(pattern_recognizers(), "US_EIN")
        assert _hits(recognizer, "Call 714-555-0123, zip 92683-1234, ref 123456789") == []

    def test_is_in_entities(self):
        #a recognizer whose entity isn't passed to analyze() gets filtered out silently
        assert {"US_EIN", "INSURANCE_POLICY"} <= set(ENTITIES)


class TestInsurancePolicy:

    def test_labeled_number_keeps_the_label_readable(self):
        recognizer = _by_entity(pattern_recognizers(), "INSURANCE_POLICY")
        assert _hits(recognizer, "Policy No: ABC1234567 attached") == [("ABC1234567", POLICY_SCORE)]
        assert _hits(recognizer, "Certificate #: 2024-00017") == [("2024-00017", POLICY_SCORE)]

    def test_bare_carrier_number_in_a_subject_line(self):
        recognizer = _by_entity(pattern_recognizers(), "INSURANCE_POLICY")
        assert _hits(recognizer, "Re: FW: Updated General Aggregate NAEP1234567") == [
            ("NAEP1234567", CARRIER_SCORE)]

    def test_number_needs_a_digit(self):
        recognizer = _by_entity(pattern_recognizers(), "INSURANCE_POLICY")
        assert _hits(recognizer, "The policy number is pending") == []

    def test_ignores_mime_ids_and_esign_ids(self):
        #both matched a loose shape-only policy regex on the real inbox
        recognizer = _by_entity(pattern_recognizers(), "INSURANCE_POLICY")
        assert _hits(recognizer, "[cid:image001.png@01DB9E12.FB123456]") == []
        assert _hits(recognizer, "Document '2ND-TYLE-123456-01-Performance Agreement' signed") == []


# ambiguous-word rescoring

def _scored(*tokens, tagged=True):
    #(word, pos, iob) triples --> the Doc trf would hand the recognizer, built without the model
    words = [w for w, _, _ in tokens]
    spaces = [i + 1 < len(words) and words[i + 1] not in (",", ".") for i in range(len(words))]
    doc = Doc(Vocab(), words=words, spaces=spaces, ents=[e for _, _, e in tokens],
              pos=[p for _, p, _ in tokens] if tagged else None)
    artifacts = NlpArtifacts(entities=list(doc.ents), tokens=doc, tokens_indices=[t.idx for t in doc],
                             lemmas=words, nlp_engine=None, language="en")
    recognizer = name_recognizers([], ["The", "To", "My", "San", "An", "Tu"])[0]
    results = recognizer.analyze(doc.text, recognizer.supported_entities, nlp_artifacts=artifacts)
    return sorted((doc.text[r.start:r.end], round(r.score, 2)) for r in results)


class TestAmbiguousRescoring:

    def test_ordinary_words_score_low(self):
        assert _scored(("The", "DET", "O"), ("lions", "NOUN", "O")) == [("The", AMBIGUOUS_WORD)]
        #followed by a verb, not a noun: the word's own tag still catches it
        assert _scored(("To", "PART", "O"), ("confirm", "VERB", "O")) == [("To", AMBIGUOUS_WORD)]

    def test_inside_a_place_scores_lowest(self):
        assert _scored(("from", "ADP", "O"), ("San", "PROPN", "B-GPE"), ("Juan", "PROPN", "I-GPE")) == [
            ("San", AMBIGUOUS_IN_ENTITY)]

    def test_lone_proper_noun_sits_on_the_threshold(self):
        assert _scored(("Hi", "INTJ", "O"), ("An", "PROPN", "O"), (",", "PUNCT", "O")) == [("An", AMBIGUOUS_ALONE)]

    def test_proper_noun_pair_scores_as_a_name(self):
        assert _scored(("Tu", "PROPN", "O"), ("Nguyen", "PROPN", "O"), ("called", "VERB", "O")) == [
            ("Tu", AMBIGUOUS_PAIR)]

    def test_inside_trf_person_scores_as_a_name(self):
        #"My" is tagged PRON even inside "My Tran"; the PERSON entity wins
        assert _scored(("My", "PRON", "B-PERSON"), ("Tran", "PROPN", "I-PERSON")) == [("My", AMBIGUOUS_PAIR)]

    def test_untagged_parse_fails_closed(self):
        assert _scored(("The", "", "O"), ("lions", "", "O"), tagged=False) == [("The", NAME_SCORE)]

