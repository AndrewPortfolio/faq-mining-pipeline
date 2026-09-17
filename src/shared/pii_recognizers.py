# Stage 1: the recognizers Presidio runs alongside its built-ins
# Fills in the gaps the English model leaves: Vietnamese/Chinese names, US street addresses
# The name lists live in data/pii/ --> real client names, gitignored

from __future__ import annotations

import os
import re
import sys

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider


# Configuration

DEFAULT_MODEL = "en_core_web_trf"
DENYLIST_PATH = "data/pii/name_denylist.txt"
ALLOWLIST_PATH = "data/pii/allowlist.txt"
VENUELIST_PATH = "data/pii/venue_denylist.txt"

# US_DRIVER_LICENSE is left out (it fires on ordinary alphanumerics), and so is NRP:
# "Vietnamese tea ceremony" is FAQ content, not PII
ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "DATE_TIME", "ORGANIZATION",
            "URL", "CREDIT_CARD", "US_SSN", "STREET_ADDRESS", "SOCIAL_HANDLE"]

# [names] matches any case because lowercase names ("pls send to huynh") are where trf fails;
# [ambiguous] needs the capital, or everyday words (do, he, song) fire on every sentence
CASE_INSENSITIVE = re.DOTALL | re.MULTILINE | re.IGNORECASE
CASE_SENSITIVE = re.DOTALL | re.MULTILINE

NAME_SCORE = 0.85       # same as a spaCy NER hit, so neither outranks the other in review
AMBIGUOUS_SCORE = 0.35  # everyday words (Do, He, Song): still flagged, but sort to the bottom
VENUE_SCORE = 0.85      # added by hand during review, so as trustworthy as a model hit

# number + up to 4 name words + a street type, with an optional unit --> "1234 Oak St Apt 5"
_STREET_RE = (r"\b\d{1,6}\s+(?:[A-Za-z][\w.'-]*\s+){0,4}"
              r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
              r"Circle|Cir|Place|Pl|Plaza|Square|Sq|Terrace|Ter|Trail|Trl|Parkway|Pkwy|"
              r"Highway|Hwy|Way)\b\.?"
              r"(?:[,\s]+(?:Apt|Apartment|Suite|Ste|Unit|#)\s*[\w-]+)?")
_PO_BOX_RE = r"\bP\.?\s?O\.?\s?Box\s+\d+\b"

# a lone @name; the lookbehind keeps it off the domain half of an email address
_HANDLE_RE = r"(?<![\w.@])@[A-Za-z0-9_](?:[A-Za-z0-9_.]{1,28}[A-Za-z0-9_])?\b"


# Name lists

def load_list(path: str) -> dict[str, list[str]]:
    # "[section]" headers, one entry per line, "#" starts a comment (the seeder writes counts there)
    sections: dict[str, list[str]] = {}
    current = sections.setdefault("", [])
    if not os.path.exists(path):
        print(f"warning: {path} not found; continuing without it", file=sys.stderr)
        return sections
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.lstrip().startswith("["):
                current = sections.setdefault(line.strip().strip("[]"), [])
                continue
            entry = line.split("#", 1)[0].strip()
            if entry:
                current.append(entry)
    return sections


def load_denylist(path: str = DENYLIST_PATH) -> tuple[list[str], list[str]]:
    sections = load_list(path)
    return sections.get("names", []), sections.get("ambiguous", [])


def load_entries(path: str) -> list[str]:
    # flat file: every entry counts, whatever section it sits under
    return [entry for entries in load_list(path).values() for entry in entries]


def load_allowlist(path: str = ALLOWLIST_PATH) -> list[str]:
    return load_entries(path)


def load_venues(path: str = VENUELIST_PATH) -> list[str]:
    return load_entries(path)


# Recognizers

def name_recognizers(names: list[str], ambiguous: list[str]) -> list[PatternRecognizer]:
    # Two recognizers rather than one so the ambiguous list carries its own score and case rule
    recognizers = []
    for label, entries, score, flags in (("names", names, NAME_SCORE, CASE_INSENSITIVE),
                                         ("ambiguous", ambiguous, AMBIGUOUS_SCORE, CASE_SENSITIVE)):
        entries = [e for e in dict.fromkeys(entries) if e]
        if entries:
            recognizers.append(PatternRecognizer(
                supported_entity="PERSON", name=f"denylist_{label}", deny_list=entries,
                deny_list_score=score, global_regex_flags=flags))
    return recognizers


def pattern_recognizers() -> list[PatternRecognizer]:
    return [
        PatternRecognizer(supported_entity="STREET_ADDRESS", name="street_address",
                          patterns=[Pattern("street", _STREET_RE, 0.5),
                                    Pattern("po_box", _PO_BOX_RE, 0.6)]),
        PatternRecognizer(supported_entity="SOCIAL_HANDLE", name="social_handle",
                          patterns=[Pattern("handle", _HANDLE_RE, 0.4)]),
    ]


def venue_recognizers(venues: list[str]) -> list[PatternRecognizer]:
    # Presidio escapes deny-list entries itself, so "St. Regis" matches literally -- don't re.escape
    venues = [v for v in dict.fromkeys(venues) if v]
    if not venues:
        return []
    return [PatternRecognizer(supported_entity="LOCATION", name="denylist_venues", deny_list=venues,
                              deny_list_score=VENUE_SCORE, global_regex_flags=CASE_INSENSITIVE)]


def custom_recognizers(denylist_path: str = DENYLIST_PATH,
                       venuelist_path: str = VENUELIST_PATH) -> list[PatternRecognizer]:
    names, ambiguous = load_denylist(denylist_path)
    return (name_recognizers(names, ambiguous)
            + venue_recognizers(load_venues(venuelist_path))
            + pattern_recognizers())


def build_analyzer(denylist_path: str = DENYLIST_PATH, venuelist_path: str = VENUELIST_PATH,
                   model: str = DEFAULT_MODEL) -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
    }).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
    for recognizer in custom_recognizers(denylist_path, venuelist_path):
        registry.add_recognizer(recognizer)
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
