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
PLACELIST_PATH = "data/pii/place_denylist.txt"

# US_DRIVER_LICENSE is left out (it fires on ordinary alphanumerics), and so is NRP:
# "Vietnamese tea ceremony" is FAQ content, not PII
ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "DATE_TIME", "ORGANIZATION",
            "URL", "CREDIT_CARD", "US_SSN", "STREET_ADDRESS", "SOCIAL_HANDLE",
            "US_EIN", "INSURANCE_POLICY"]

# [names] matches any case because lowercase names ("pls send to huynh") are where trf fails;
# [ambiguous] needs the capital, or everyday words (do, he, song) fire on every sentence
CASE_INSENSITIVE = re.DOTALL | re.MULTILINE | re.IGNORECASE
CASE_SENSITIVE = re.DOTALL | re.MULTILINE

NAME_SCORE = 0.85       # same as a spaCy NER hit, so neither outranks the other in review

# [ambiguous] hits are rescored from trf's own read of the word in its sentence
AMBIGUOUS_RECOGNIZER = "denylist_ambiguous"
AMBIGUOUS_IN_ENTITY = 0.1   # inside a place/org/date entity: "San" in San Juan Capistrano
AMBIGUOUS_WORD = 0.2        # tagged as an ordinary word: "The" lions, "To" confirm, "My" family
AMBIGUOUS_ALONE = 0.6       # a proper noun on its own: "Hi An,"
AMBIGUOUS_PAIR = 0.85       # beside another proper noun, or inside trf's own PERSON: "Tu Nguyen"
VENUE_SCORE = 0.85      # added by hand during review, so as trustworthy as a model hit
PLACE_SCORE = 0.85      # trf's own majority call on the term, carried to the mentions it skipped
ZIP_SCORE = 0.6         # every LOCATION redacts, so this only orders review rows
EIN_SCORE = 0.4         # the shape alone is weak; "tax id" / "ein" nearby lifts it to 0.75
POLICY_SCORE = 0.6      # the "Policy No:" label is already part of the match
CARRIER_SCORE = 0.8     # a confirmed carrier prefix, found bare in subject lines

# number + up to 4 name words + a street type, with an optional unit --> "1234 Oak St Apt 5"
_STREET_RE = (r"\b\d{1,6}\s+(?:[A-Za-z][\w.'-]*\s+){0,4}"
              r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
              r"Circle|Cir|Place|Pl|Plaza|Square|Sq|Terrace|Ter|Trail|Trl|Parkway|Pkwy|"
              r"Highway|Hwy|Way)\b\.?"
              r"(?:[,\s]+(?:Apt|Apartment|Suite|Ste|Unit|#)\s*[\w-]+)?")
_PO_BOX_RE = r"\bP\.?\s?O\.?\s?Box\s+\d+\b"

# a lone @name; the lookbehind keeps it off the domain half of an email address
_HANDLE_RE = r"(?<![\w.@])@[A-Za-z0-9_](?:[A-Za-z0-9_.]{1,28}[A-Za-z0-9_])?\b"

# ##-#######; the bare 9-digit form is left to US_SSN, which already flags it at 0.05
_EIN_RE = r"\b\d{2}-\d{7}\b"

# Policy numbers share no format, so a label anchors the match; the lookbehind (Presidio compiles
# with the regex module) keeps "Policy No:" readable and redacts only the number, which needs a digit
_POLICY_RE = (r"(?<=\b(?:policy|certificate|cert)\s*(?:no\.?|number|#)\s*[:#-]?\s*)"
              r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,19}\b")
_CARRIER_RE = r"\bNAEP\d{4,10}\b"

# California's zip range is the only bare 9xxxx number this inbox writes: 12 of 12 sampled were zips
# ("Costa Mesa 92626", "Ca 92832"). The guards keep prices ($95,000), decimals and #refs out
_CA_ZIP_RE = r"(?<![\d$.,/#-])\b9[0-6]\d{3}(?:-\d{4})?\b(?![\d,.]*\d)"
# Any other state needs its code in front, or every 5-digit number would match
_STATES = ("AL|AK|AZ|AR|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|"
           "NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC")
_STATE_ZIP_RE = rf"(?<=\b(?:{_STATES})\.?,?\s{{1,3}})\d{{5}}(?:-\d{{4}})?\b"

EIN_CONTEXT = ["ein", "fein", "tax", "employer", "federal", "tin"]
POLICY_CONTEXT = ["policy", "insurance", "certificate", "coi", "coverage", "insured"]


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


def load_places(path: str = PLACELIST_PATH) -> tuple[list[str], list[str]]:
    sections = load_list(path)
    return sections.get("places", []), sections.get("capitalized", [])


# Recognizers

class AmbiguousNameRecognizer(PatternRecognizer):
    # The deny-list regex finds the word; trf's reading of it in this sentence sets the score
    # Presidio already hands every recognizer the parsed Doc, so this costs no extra model time

    def analyze(self, text, entities, nlp_artifacts=None, regex_flags=None):
        results = super().analyze(text, entities, nlp_artifacts, regex_flags)
        doc = getattr(nlp_artifacts, "tokens", None)
        for result in results:
            score = ambiguous_score(doc, result.start, result.end)
            if score is not None:  # nothing to read --> keeps NAME_SCORE, so the row redacts
                result.score = score
        return results


def ambiguous_score(doc, start: int, end: int) -> float | None:
    if doc is None:
        return None
    span = doc.char_span(start, end, alignment_mode="expand")
    if span is None or not len(span) or not span[0].pos_:
        return None
    token = span[0]
    label = next((e.label_ for e in doc.ents if e.start <= token.i < e.end), "")
    if label == "PERSON":
        return AMBIGUOUS_PAIR  # "My" is tagged PRON even inside "My Tran"; the entity wins
    if label:
        return AMBIGUOUS_IN_ENTITY
    if token.pos_ != "PROPN":
        return AMBIGUOUS_WORD
    after = doc[token.i + 1] if token.i + 1 < len(doc) else None
    return AMBIGUOUS_PAIR if after is not None and after.pos_ == "PROPN" else AMBIGUOUS_ALONE


def name_recognizers(names: list[str], ambiguous: list[str]) -> list[PatternRecognizer]:
    # Two recognizers rather than one so the ambiguous list carries its own scoring and case rule
    recognizers = []
    for label, entries, cls, flags in (("names", names, PatternRecognizer, CASE_INSENSITIVE),
                                       ("ambiguous", ambiguous, AmbiguousNameRecognizer, CASE_SENSITIVE)):
        entries = [e for e in dict.fromkeys(entries) if e]
        if entries:
            recognizers.append(cls(
                supported_entity="PERSON", name=f"denylist_{label}", deny_list=entries,
                deny_list_score=NAME_SCORE, global_regex_flags=flags))
    return recognizers


def pattern_recognizers() -> list[PatternRecognizer]:
    return [
        PatternRecognizer(supported_entity="STREET_ADDRESS", name="street_address",
                          patterns=[Pattern("street", _STREET_RE, 0.5),
                                    Pattern("po_box", _PO_BOX_RE, 0.6)]),
        PatternRecognizer(supported_entity="SOCIAL_HANDLE", name="social_handle",
                          patterns=[Pattern("handle", _HANDLE_RE, 0.4)]),
        PatternRecognizer(supported_entity="US_EIN", name="us_ein",
                          patterns=[Pattern("ein", _EIN_RE, EIN_SCORE)], context=EIN_CONTEXT),
        PatternRecognizer(supported_entity="INSURANCE_POLICY", name="insurance_policy",
                          patterns=[Pattern("labeled", _POLICY_RE, POLICY_SCORE),
                                    Pattern("carrier", _CARRIER_RE, CARRIER_SCORE)],
                          context=POLICY_CONTEXT),
        # LOCATION, so the location rule redacts it; case-sensitive so "in 92618" isn't read as Indiana
        PatternRecognizer(supported_entity="LOCATION", name="us_zip",
                          patterns=[Pattern("ca_range", _CA_ZIP_RE, ZIP_SCORE),
                                    Pattern("state_anchored", _STATE_ZIP_RE, ZIP_SCORE)],
                          global_regex_flags=CASE_SENSITIVE),
    ]


def venue_recognizers(venues: list[str]) -> list[PatternRecognizer]:
    # Presidio escapes deny-list entries itself, so "St. Regis" matches literally -- don't re.escape
    venues = [v for v in dict.fromkeys(venues) if v]
    if not venues:
        return []
    return [PatternRecognizer(supported_entity="LOCATION", name="denylist_venues", deny_list=venues,
                              deny_list_score=VENUE_SCORE, global_regex_flags=CASE_INSENSITIVE)]


def place_recognizers(places: list[str], capitalized: list[str]) -> list[PatternRecognizer]:
    # Seeded by seed_place_denylist.py; same two-list split as the names. [capitalized] places are
    # also everyday English here ("orange" is a lion color), so only the capitalized spelling counts
    recognizers = []
    for label, entries, flags in (("places", places, CASE_INSENSITIVE),
                                  ("places_capitalized", capitalized, CASE_SENSITIVE)):
        entries = [e for e in dict.fromkeys(entries) if e]
        if entries:
            recognizers.append(PatternRecognizer(
                supported_entity="LOCATION", name=f"denylist_{label}", deny_list=entries,
                deny_list_score=PLACE_SCORE, global_regex_flags=flags))
    return recognizers


def custom_recognizers(denylist_path: str = DENYLIST_PATH, venuelist_path: str = VENUELIST_PATH,
                       placelist_path: str = PLACELIST_PATH) -> list[PatternRecognizer]:
    names, ambiguous = load_denylist(denylist_path)
    return (name_recognizers(names, ambiguous)
            + venue_recognizers(load_venues(venuelist_path))
            + place_recognizers(*load_places(placelist_path))
            + pattern_recognizers())


def build_analyzer(denylist_path: str = DENYLIST_PATH, venuelist_path: str = VENUELIST_PATH,
                   placelist_path: str = PLACELIST_PATH, model: str = DEFAULT_MODEL) -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
    }).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
    for recognizer in custom_recognizers(denylist_path, venuelist_path, placelist_path):
        registry.add_recognizer(recognizer)
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
