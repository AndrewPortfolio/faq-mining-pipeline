#Tests for shared.decisions (Stage 1)

#Fixtures are synthetic --> no real PII, nothing in git


import pytest

from shared.decisions import AMBIGUOUS_THRESHOLD, DATETIME_OTHER, Rules, normalize_term
from shared.pii_recognizers import AMBIGUOUS_ALONE, AMBIGUOUS_RECOGNIZER, AMBIGUOUS_WORD

RULES = Rules(common_datetime=["morning", "October", "business hours"],
              common_location=["home"])


def _decide(entity, text, rules=RULES, score=None, recognizer=""):
    return rules.decide(entity, text, score, recognizer)


# always redact

class TestAlwaysRedact:

    @pytest.mark.parametrize("score_blind_text", ["Irene", "Do", "The", "morning"])
    def test_person_always_redacts(self, score_blind_text):
        #frequency and score don't matter anymore, and DATE_TIME's common words don't reach PERSON
        assert _decide("PERSON", score_blind_text) == ("redact", "person_always")

    def test_url_always_redacts(self):
        assert _decide("URL", "https://example.com/booking") == ("redact", "url_always")

    @pytest.mark.parametrize("entity", ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_EIN", "INSURANCE_POLICY",
                                        "ORGANIZATION", "STREET_ADDRESS"])
    def test_everything_else_defaults_to_redact(self, entity):
        assert _decide(entity, "anything") == ("redact", "default")


# ambiguous-list words

class TestAmbiguousWords:

    def test_low_score_keeps(self):
        assert _decide("PERSON", "The", score=AMBIGUOUS_WORD, recognizer=AMBIGUOUS_RECOGNIZER) == (
            "keep", "ambiguous_word")

    def test_lone_proper_noun_redacts(self):
        #"Hi An," is addressing someone: it sits exactly on the threshold, and at-or-above redacts
        assert AMBIGUOUS_ALONE >= AMBIGUOUS_THRESHOLD
        assert _decide("PERSON", "An", score=AMBIGUOUS_ALONE, recognizer=AMBIGUOUS_RECOGNIZER) == (
            "redact", "person_always")

    def test_score_gates_only_the_ambiguous_list(self):
        assert _decide("PERSON", "Thao", score=0.2, recognizer="SpacyRecognizer") == ("redact", "person_always")

    def test_missing_score_redacts(self):
        assert _decide("PERSON", "The", recognizer=AMBIGUOUS_RECOGNIZER) == ("redact", "person_always")

    def test_score_read_back_from_csv(self):
        assert _decide("PERSON", "The", score="0.2", recognizer=AMBIGUOUS_RECOGNIZER) == ("keep", "ambiguous_word")


# common words

class TestCommonWords:

    def test_match_ignores_case_and_edge_punctuation(self):
        assert _decide("DATE_TIME", "Morning!") == ("keep", "common_word")

    def test_keep_is_the_decision_not_a_flag(self):
        #a common word must suppress the redact outright, not just route the row to review
        decision, _ = _decide("DATE_TIME", "morning")
        assert decision == "keep"

    def test_whole_span_only(self):
        #"October" is generic; "October 22" is a client's event date
        assert _decide("DATE_TIME", "October") == ("keep", "common_word")
        assert _decide("DATE_TIME", "October 22") == ("redact", "calendar_date")

    def test_phrases_and_extra_spaces(self):
        assert _decide("DATE_TIME", "Business  Hours") == ("keep", "common_word")

    def test_lists_are_scoped_to_their_entity(self):
        assert _decide("LOCATION", "morning") == ("redact", "location_always")
        assert _decide("DATE_TIME", "home") == (DATETIME_OTHER, "datetime_other")


# DATE_TIME

class TestDateTime:

    @pytest.mark.parametrize("text", ["October 22", "Oct. 22nd", "22nd of October", "10/22/2027", "10/22",
                                      "10-22-2027", "2027-10-22", "the 22nd"])
    def test_calendar_dates_redact(self, text):
        assert _decide("DATE_TIME", text) == ("redact", "calendar_date")

    def test_combined_date_and_time_span_redacts_whole(self):
        #Presidio sometimes emits date + time as one span; the date takes the time with it
        assert _decide("DATE_TIME", "Saturday, October 22nd, 2027 at 6pm") == ("redact", "calendar_date")

    @pytest.mark.parametrize("text", ["8am", "8am - 10am", "12:40 - 12:55 PM", "noon", "7:02:01 pm",
                                      "9 o'clock", "10:00am on Friday"])
    def test_clock_times_without_a_date_keep(self, text):
        assert _decide("DATE_TIME", text) == ("keep", "clock_time")

    @pytest.mark.parametrize("text", ["15-20 mins", "10–15 mins", "about 15 minutes", "2 hrs", "1.5 hours",
                                      "an hour", "about an hour", "half an hour", "a few minutes",
                                      "a couple of hours", "fifteen minutes", "45 min+"])
    def test_durations_keep(self, text):
        assert _decide("DATE_TIME", text) == ("keep", "duration")

    def test_date_beats_duration(self):
        assert _decide("DATE_TIME", "October 22, 15 minutes early") == ("redact", "calendar_date")

    @pytest.mark.parametrize("text", ["2025", "1998"])
    def test_bare_years_keep(self, text):
        assert _decide("DATE_TIME", text) == ("keep", "bare_year")

    def test_year_inside_a_date_still_redacts(self):
        assert _decide("DATE_TIME", "10/22/2025") == ("redact", "calendar_date")

    @pytest.mark.parametrize("text", ["this year", "October 2027", "80th", "60th anniversary", "next week",
                                      "2025 season"])
    def test_everything_else_falls_to_the_default(self, text):
        #"80th" is an age, not a day of the month; a year with other words isn't a bare year
        assert _decide("DATE_TIME", text) == (DATETIME_OTHER, "datetime_other")


# LOCATION

class TestLocation:

    def test_everything_else_redacts(self):
        assert _decide("LOCATION", "Irvine") == ("redact", "location_always")

    def test_region_words_keep(self):
        #the common-word check runs before the location rule, so "CA" never reaches it
        rules = Rules(common_location=["ca"])
        assert _decide("LOCATION", "CA", rules) == ("keep", "common_word")


# loading

class TestLoad:

    def test_reads_files_with_comments(self, tmp_path):
        dt = tmp_path / "dt.txt"
        dt.write_text("# generic\nmorning  # most common\n\n", encoding="utf-8")
        loc = tmp_path / "loc.txt"
        loc.write_text("home\n", encoding="utf-8")
        rules = Rules.load(str(dt), str(loc))
        assert _decide("DATE_TIME", "Morning", rules) == ("keep", "common_word")
        assert _decide("LOCATION", "home", rules) == ("keep", "common_word")

    def test_missing_files_still_decide(self, tmp_path):
        rules = Rules.load(*(str(tmp_path / n) for n in ("a.txt", "b.txt")))
        assert _decide("LOCATION", "Irvine", rules) == ("redact", "location_always")
        assert _decide("DATE_TIME", "morning", rules) == (DATETIME_OTHER, "datetime_other")

    def test_normalize_term(self):
        assert normalize_term("  12pm -2pm. ") == "12pm -2pm"
        assert normalize_term("**Saturday**") == "saturday"
