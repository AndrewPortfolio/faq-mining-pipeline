# Stage 1b: find the signature block at the end of an email body, so it can go as one unit

# Titles, credentials and taglines are Quasi-Identifiers 
# Recognizer doesn't catch them one by one 
# Signature are marked by the redacted PII packed around them
# a name, a company and contact details within a few lines -- so the whole cluster goes
# Quoted replies are already cut at extraction, so a signature is the tail of the sender's own text
# Every rule errs toward keeping text: a question, a line in the sender's own voice and a form field
# are never stripped (measured on the inbox: 0 of 24,015 question marks lost)

from __future__ import annotations

import re


# Configuration

# The review-file row a zone becomes: analyze_pii writes it, apply_redactions deletes what it covers
SIGNATURE_ENTITY = "SIGNATURE"
SIGNATURE_RULE = "signature_block"

# The scaffold's cluster: 2+ distinct entity types, each within 200 chars of the previous one
WINDOW = 200
MIN_TYPES = 2
SIGNATURE_TYPES = {"PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "LOCATION",
                   "STREET_ADDRESS", "SOCIAL_HANDLE"}
# A signature carries contact details; a question with a name and a city doesn't
CONTACT_TYPES = {"EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "STREET_ADDRESS", "SOCIAL_HANDLE"}

# The website form packs entities like a signature does, but it is the request
FORM_MARKERS = ("Date/Time of Event", "Message (Please", "Day of Contact", "Total: $")

# Any template's event field: "Wedding Date:", "Date of wedding:", "Insurance Required?(Yes/No):"
# Contact labels ("Phone:", "Fax:", "c:") are signature material, so they're left out on purpose
EVENT_LABEL = re.compile(
    r"^[\W_]*(?:[\w’'&/-]+\s+){0,5}"
    r"(?:Date|Time|Location|Venue|Event|Performance|Guests?|Budget|Insurance(?:\s+(?:Required|needed))?|"
    r"Coordinator(?:\s+(?:Name|Contact\s+Number))?|Notes|Message|Total|Lions?)"
    r"(?:\s+of\s+(?:the\s+|your\s+|our\s+)?[\w’']+)?"
    r"\s*\??\s*(?:\([^)\n]*\))?\s*\??\s*\**\s*:", re.IGNORECASE)

# The sender's own voice: a personal pronoun (Vietnamese ones too) or a courtesy word
PERSONAL = re.compile(
    r"\b(?:i|i'm|i've|i'll|i'd|me|my|mine|we|we're|we've|we'll|we'd|us|our|ours|you|you're|you've|you'll|"
    r"you'd|your|yours|he|she|they|him|her|his|hers|them|their|theirs|sorry|thanks?|thank you|hope|"
    r"appreciate|looking forward|tôi|toi|chúng|mình|bạn|anh|chị|chi|em|con|cháu|ông|bà|cô|chú|quý)\b",
    re.IGNORECASE)
# ...except a signature's own calls to action, which say "us" too: "Follow us on Instagram!"
CALL_TO_ACTION = re.compile(r"^\W*(?:please\s+)?(?:follow|like|visit|find|connect with|check|add)\b"
                            r".{0,25}\b(?:us|our)\b", re.IGNORECASE)
# Legal and mail-client boilerplate never protects anything; it goes along with the signature
DISCLAIMER = re.compile(r"confidential|sole use of|intended (?:only )?for|intended recipient|protect your privacy|"
                        r"clicking the link does not work|unsubscribe", re.IGNORECASE)
# A signature never opens with a greeting; a zone that would is the whole message
GREETING = re.compile(r"^\W*(?:hi|hello|hey|dear|greetings|good (?:morning|afternoon|evening))\b", re.IGNORECASE)
# Where a signature starts, when the sender wrote one
SIGN_OFF = re.compile(r"^\W*(?:best(?: regards| wishes)?|kind regards|warm(?:est)? regards|warmly|regards|"
                      r"sincerely|cheers|thanks(?: so much| again)?|thank you(?: so much)?|many thanks|"
                      r"respectfully|with gratitude|talk soon|xoxo|bless(?:ings)?)\W*$", re.IGNORECASE)

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d{1,2}[.)])\s+")
_ENDS_SENTENCE = re.compile(r"[.!?:;]\s*$")
_ENDS_LINE = re.compile(r"[.!]\s*$")


# Line tests (run on the masked text, so a redacted name or a URL's "?" can't sway them)

def _sentence(line: str) -> bool:
    # prose, not a name/title/contact line: 10+ words, mostly lowercase
    words = _WORD.findall(line)
    return len(words) >= 10 and sum(w.islower() for w in words) >= 0.6 * len(words)


def _personal(line: str) -> bool:
    # a question, or a line in the sender's own voice
    if "?" in line:
        return True
    if DISCLAIMER.search(line) or CALL_TO_ACTION.search(line) or SIGN_OFF.match(line):
        return False
    if _sentence(line) and _BULLET.match(line):
        return True                                   # a signature doesn't list full sentences
    return len(_WORD.findall(line)) >= 4 and bool(PERSONAL.search(line))


def _mark(masked: str, start: int, end: int, first_contact: int) -> list[tuple[int, int, bool]]:
    # (line start, line end, protected) for each line of masked[start:end]
    marks, pos, prev, prev_kept = [], start, "", False
    for line in masked[start:end].split("\n"):
        stop = pos + len(line) + 1
        # a hard-wrapped line carries on the sentence above it, and its protection
        continues = bool(prev.strip()) and not _ENDS_SENTENCE.search(prev) and line.lstrip()[:1].islower()
        kept = (_personal(line) or bool(EVENT_LABEL.match(line))
                or (pos <= first_contact and _sentence(line) and not DISCLAIMER.search(line))
                or (continues and prev_kept))
        marks.append((pos, stop, kept))
        pos, prev, prev_kept = stop, line, kept
    return marks


# Zones

def _clusters(spans: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    # the scaffold's density scan, plus: the cluster has to hold contact details
    out, i = [], 0
    while i < len(spans):
        start, end, reach, types, j = spans[i][0], spans[i][1], spans[i][0] + WINDOW, set(), i
        while j < len(spans) and spans[j][0] <= reach:
            types.add(spans[j][2])
            end = max(end, spans[j][1])
            j += 1
        if len(types) >= MIN_TYPES and types & CONTACT_TYPES:
            while j < len(spans) and spans[j][0] - end < WINDOW:
                end = max(end, spans[j][1])
                j += 1
            out.append((start, end))
            i = j
        else:
            i += 1
    return out


def _pull_up(text: str, masked: str, s: int) -> int:
    # short lines just above the block are its sign-off, name and title: "Best,", "Kelly", "MBA"
    # A line ending in ":" introduces the value below it ("My phone number is:"), so it stays
    while s > 0 and text[s - 1] == "\n":
        top = text.rfind("\n", 0, s - 1) + 1
        line = text[top:s - 1]
        if (not line.strip() or "?" in line or EVENT_LABEL.match(line) or len(_WORD.findall(line)) > 4
                or _ENDS_LINE.search(line) or line.rstrip().endswith(":") or _personal(masked[top:s - 1])):
            break
        s = top
    return s


def _sign_off(masked: str, s: int, first_contact: int) -> int | None:
    # the last sign-off above the contact details: everything above it is the message
    anchor, pos = None, s
    for line in masked[s:first_contact].split("\n"):
        if SIGN_OFF.match(line):
            anchor = pos
        pos += len(line) + 1
    return anchor


def signature_zones(text: str, spans) -> list[tuple[int, int]]:
    # spans: (start, end, entity_type) of every redaction decided for this text
    # Returns at most one zone, (start, len(text)): the signature runs to the end of the message
    spans = sorted(spans)
    chars = list(text)
    for a, b, _ in spans:
        chars[a:b] = " " * (b - a)
    masked = "".join(chars)
    signature_spans = [s for s in spans if s[2] in SIGNATURE_TYPES]
    starts = []
    for first, last in _clusters(signature_spans):
        s = text.rfind("\n", 0, first) + 1
        e = text.find("\n", last)
        e = len(text) if e < 0 else e
        if any(m in text[s:e] for m in FORM_MARKERS):
            continue
        if any(kept for _, _, kept in _mark(masked, e, len(text), -1)):
            continue                                  # the sender keeps talking after it: not a sign-off
        first_contact = min(a for a, _, t in signature_spans if s <= a < e and t in CONTACT_TYPES)
        for _, stop, kept in _mark(masked, s, e, first_contact):
            if kept:
                s = stop                              # keep everything down to the last protected line
        s = _pull_up(text, masked, min(s, len(text)))
        contacts = [a for a, _, t in signature_spans if a >= s and t in CONTACT_TYPES]
        if not contacts:
            continue
        anchor = _sign_off(masked, s, min(contacts))
        if anchor is not None:
            s = anchor
        elif any(_personal(line) or GREETING.match(line) for line in masked[s:min(contacts)].split("\n")):
            continue                                  # a message sits above the contact details
        for a, b, _ in spans:                         # never cut through a redaction
            if a < s < b:
                s = a
        while s > 0 and text[s - 1] == "<":
            s -= 1
        starts.append(s)
    return [(min(starts), len(text))] if starts else []
