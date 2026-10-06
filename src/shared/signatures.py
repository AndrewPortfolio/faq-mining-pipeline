# Stage 1b: find the signature block at the end of an email body, so it can go as one unit

# Titles, credentials and taglines are Quasi-Identifiers 
# Recognizer doesn't catch them one by one 
# Signature are marked by the redacted PII packed around them
# a name, a company and contact details within a few lines -- so the whole cluster goes
# Without contact details, a short line opening with the sender's name near the end starts it instead,
# and the sign-off above it joins in; it stops at a P.S. or the sender talking again
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

# The tail rule, for signatures without contact details: the name line sits among the last 4 lines
# and is short once its entities are masked ("<PERSON> | SWE | City of Awesome")
TAIL_LINES = 4
NAME_WORDS = 6

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
# Where a signature starts, when the sender wrote one; phrases can chain: "Blessings, kindly,"
_SIGN_OFF_PHRASE = (r"(?:best(?: regards| wishes)?|all the best|kind(?:est)? regards|kindly|warm(?:est)? regards|"
                    r"warm wishes|warmly|warmest|regards|sincerely(?: yours)?|respectfully(?: yours)?|cheers|"
                    r"thanks(?: so much| again)?|thank you(?: so much| again| very much| kindly)?|many thanks|"
                    r"ty|thx|with (?:gratitude|love)|gratefully|love|take care|talk soon|xoxo|xo|"
                    r"have a (?:great|good|nice|wonderful|blessed) (?:day|weekend|evening|night|one)|"
                    r"(?:god |many )?bless(?:ings)?)")
SIGN_OFF = re.compile(rf"^\W*{_SIGN_OFF_PHRASE}(?:\W+(?:and\s+)?{_SIGN_OFF_PHRASE})*\W*$", re.IGNORECASE)
# A note after the signature is the sender talking again: "P.S. we can bring drums too"
PS = re.compile(r"^\W*p\.?\s?(?:p\.?\s?)?s\b", re.IGNORECASE)

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
    # a question, a P.S., or a line in the sender's own voice
    if "?" in line or PS.match(line):
        return True
    if DISCLAIMER.search(line) or CALL_TO_ACTION.search(line) or SIGN_OFF.match(line):
        return False
    if _sentence(line) and _BULLET.match(line):
        return True                                   # a signature doesn't list full sentences
    return len(_WORD.findall(line)) >= 4 and bool(PERSONAL.search(line))


def _talking(line: str) -> bool:
    # below a signature only the sender's voice, a greeting, an event field or a form is message again;
    # impersonal sentences there are taglines, as in _mark
    return (_personal(line) or bool(GREETING.match(line) or EVENT_LABEL.match(line))
            or any(m in line for m in FORM_MARKERS))


def _title_case(line: str) -> bool:
    # a few mostly-capitalized words, like a name, title or org line -- not a sentence
    words = _WORD.findall(line)
    return len(words) <= NAME_WORDS and 2 * sum(w.islower() for w in words) <= len(words)


def _name_line(text: str, masked: str, a: int, b: int, persons: set[int]) -> bool:
    # opens with a name and goes on in a few title-case words: "<PERSON> | SWE | City of Awesome"
    lead = re.search(r"[^\W_]", text[a:b])
    return (lead is not None and a + lead.start() in persons and _title_case(masked[a:b])
            and not _talking(masked[a:b]))


def _signed_line(masked: str, a: int, b: int, persons: set[int]) -> bool:
    # a sign-off with the name on the same line: "Thanks, <PERSON>"
    line = masked[a:b]
    return bool(SIGN_OFF.match(line)) and "?" not in line and any(a <= p < b for p in persons)


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


def _sign_off_above(text: str, masked: str, s: int) -> int:
    # the sign-offs right above line start s open the signature, blank lines between or not: "Kindly,\n\n<PERSON>"
    top = s
    while top > 0:
        start = text.rfind("\n", 0, top - 1) + 1
        if SIGN_OFF.match(masked[start:top - 1]) and "?" not in masked[start:top - 1]:   # \W* would eat a "?"
            s = top = start
        elif not text[start:top - 1].strip():
            top = start
        else:
            break
    return s


def _tail_zone(text: str, masked: str, spans) -> tuple[int, int] | None:
    # A signature with no contact details: a name line near the end, the sign-off above it, and the rest below
    # it down to the end or a P.S. A sign-off carrying the name ("Thanks, <PERSON>") counts too, when only
    # signature lines follow it. Only a zone when it strips words the redactions left (a title, an org)
    rows, pos = [], 0
    for line in text.split("\n"):
        if line.strip():
            rows.append((pos, pos + len(line)))
        pos += len(line) + 1
    cut = len(rows)
    while cut and _talking(masked[rows[cut - 1][0]:rows[cut - 1][1]]):
        cut -= 1                                      # a trailing P.S. isn't part of the signature
    persons = {a for a, _, t in spans if t == "PERSON"}
    for top, bottom in rows[max(0, cut - TAIL_LINES):cut]:   # top down; a candidate that fails hands on to the next
        signed = _signed_line(masked, top, bottom, persons)
        if not (signed or _name_line(text, masked, top, bottom, persons)):
            continue
        start = _sign_off_above(text, masked, top)
        if not any(b < start for _, b in rows):
            continue                                  # nothing above it: that's the whole message
        end = next((a for a, b in rows if a > top and _talking(masked[a:b])), len(text))
        if signed and not all(_title_case(masked[a:b]) for a, b in rows if top < a < end):
            continue                                  # a mid-message "Thanks <PERSON>!" with text below it
        if any(_WORD.findall(masked[a:b]) and not SIGN_OFF.match(masked[a:b]) for a, b in rows if start <= a < end):
            return start, end                         # it strips words the redactions left
    return None


def signature_zones(text: str, spans) -> list[tuple[int, int]]:
    # spans: (start, end, entity_type) of every redaction decided for this text
    # Returns at most one zone: a contact block's runs to the end of the message, the tail rule's can stop at a P.S.
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
        s = _sign_off_above(text, masked, s)          # "Best," over a blank line still opens it
        for a, b, _ in spans:                         # never cut through a redaction
            if a < s < b:
                s = a
        while s > 0 and text[s - 1] == "<":
            s -= 1
        starts.append(s)
    tail = _tail_zone(text, masked, spans)
    if starts:
        start = min(starts)
        if tail and tail[0] < start <= tail[1]:       # the tail's sign-off and name sit right above the block
            start = tail[0]
        return [(start, len(text))]
    return [tail] if tail else []
