# Stage 1: the redact/keep call pre-filled into every review row
# pii_recognizers decides what counts as a span; this decides what happens to it
# The rule that fired is written next to the decision, so a surprising keep can be traced back

from __future__ import annotations

import re

from shared.pii_recognizers import AMBIGUOUS_RECOGNIZER, load_entries


# Configuration

COMMON_DATETIME_PATH = "data/pii/common_words_datetime.txt"
COMMON_LOCATION_PATH = "data/pii/common_words_location.txt"

# Score doesn't gate these: a missed name or link is permanent once embedded,
# while a redacted false positive only costs a word
ALWAYS_REDACT = {"PERSON": "person_always", "URL": "url_always"}

# An ambiguous-list PERSON hit scoring below this reads as an ordinary word and is kept.
# AMBIGUOUS_ALONE sits exactly on it, so a lone proper noun ("Hi An,") still redacts
AMBIGUOUS_THRESHOLD = 0.6

# DATE_TIME that no rule below claims ("next week", "a great day")
DATETIME_OTHER = "redact"

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DAY = r"(?:[12]?\d|3[01])"  # 1-31, so "80th birthday" isn't read as a date
_ORD = r"(?:st|nd|rd|th)"

# A specific day, with or without a time attached: Presidio sometimes emits "October 22" and
# "10am" as two spans and sometimes one, so this runs on the span's text, not its boundaries
CALENDAR_RE = re.compile(
    rf"\b{_MONTH}\.?\s+{_DAY}{_ORD}?\b"           # October 22, Oct. 22nd
    rf"|\b{_DAY}{_ORD}?\s+(?:of\s+)?{_MONTH}\b"    # 22 October, 22nd of Oct
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"           # 10/22, 10/22/2027
    r"|\b\d{1,2}-\d{1,2}-\d{2,4}\b"                # 10-22-2027; the year is required or "15-20 mins" matches
    r"|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"              # 8.23.27; three parts, so "1.5 hours" stays a duration
    r"|\b\d{4}-\d{2}-\d{2}\b"                      # 2027-10-22
    rf"|\b{_DAY}{_ORD}\b",                         # the 22nd
    re.IGNORECASE)

# A length of time, not a point in it: "15-20 mins", "about an hour", "a couple of hours"
_COUNT = (r"(?:\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|"
          r"thirty|forty(?:[- ]five)?|fifty|sixty|ninety|half(?: an?)?|a half|a couple(?: of)?|"
          r"a few|few|several)")
DURATION_RE = re.compile(
    rf"\b{_COUNT}(?:\s*(?:-|\u2013|\u2014|to|or)\s*{_COUNT})?\s*\+?\s*(?:min|mins|minutes?|hrs?|hours?)\b",
    re.IGNORECASE)

# Whole span only: "2025" is context, "10/22/2025" is still a date
BARE_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

CLOCK_RE = re.compile(
    r"\b\d{1,2}(?::\d{2}){0,2}\s*(?:am|pm|a\.m\.?|p\.m\.?)(?![a-z])"
    r"|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*o'?clock\b|\bnoon\b|\bmidnight\b",
    re.IGNORECASE)

_EDGE = " \t\r\n.,;:!?()[]{}\"'*-–—"


def normalize_term(text: str) -> str:
    # lowercase, one space between words, no edge punctuation: "Morning!" and "morning" are one term
    return " ".join(text.split()).strip(_EDGE).lower()


# Rules

class Rules:

    def __init__(self, common_datetime=(), common_location=()):
        self.common = {"DATE_TIME": {normalize_term(w) for w in common_datetime} - {""},
                       "LOCATION": {normalize_term(w) for w in common_location} - {""}}

    @classmethod
    def load(cls, datetime_path: str = COMMON_DATETIME_PATH,
             location_path: str = COMMON_LOCATION_PATH) -> "Rules":
        return cls(load_entries(datetime_path), load_entries(location_path))

    def decide(self, entity_type: str, text: str, score: float | str | None = None,
               recognizer: str = "") -> tuple[str, str]:
        # The one place score gates PERSON: trf read the word as ordinary English ("The" lions)
        if (entity_type == "PERSON" and recognizer == AMBIGUOUS_RECOGNIZER and score is not None
                and float(score) < AMBIGUOUS_THRESHOLD):
            return "keep", "ambiguous_word"
        if entity_type in ALWAYS_REDACT:
            return "redact", ALWAYS_REDACT[entity_type]
        term = normalize_term(text)
        # Whole-span match, so "October" is kept but "October 22" still reaches the calendar check
        if term in self.common.get(entity_type, ()):
            return "keep", "common_word"
        if entity_type == "DATE_TIME":
            if BARE_YEAR_RE.fullmatch(term):
                return "keep", "bare_year"
            if CALENDAR_RE.search(term):
                return "redact", "calendar_date"
            if CLOCK_RE.search(term):
                return "keep", "clock_time"
            # after the calendar check, so "October 22, 15 minutes early" still redacts
            if DURATION_RE.search(term):
                return "keep", "duration"
            return DATETIME_OTHER, "datetime_other"
        if entity_type == "LOCATION":
            # any place can lead back to a client; region words already returned keep above
            return "redact", "location_always"
        return "redact", "default"
