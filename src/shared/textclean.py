"""Pure text-cleaning transforms shared across pipeline stages.

Every function here takes a string and returns a string. Nothing in this module
knows about email, MIME, or the filesystem, which is what makes it testable in
isolation -- and it needs testing, because these transforms *delete* text. When
one over-matches there is no exception, only a shorter body and quietly worse
clusters several stages downstream.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser


# --- HTML to text ----------------------------------------------------------

_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "blockquote", "section", "article", "header", "footer",
}


class _HTMLToText(HTMLParser):
    """Minimal stdlib HTML renderer. Enough for short emails, no dependency."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._suppress = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._suppress += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._suppress = max(0, self._suppress - 1)
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data):
        if not self._suppress:
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def html_to_text(html: str) -> str:
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # Malformed markup: fall back to a crude tag strip rather than losing
        # the message entirely.
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()

# --- Reply, signature and footer trimming ----------------------------------

_QUOTE_PATTERNS = [
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.I | re.M),
    re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}", re.I | re.M),
    re.compile(r"^On\b.{0,300}?\bwrote:\s*$", re.I | re.M | re.S),
    re.compile(r"^From:\s.+\r?\n(?:Sent|Date):\s", re.I | re.M),
    re.compile(r"^_{10,}\s*$", re.M),
    re.compile(r"^\s*On .{0,100}, .{0,100} <[^>]+>\s*$", re.I | re.M),
]

_FOOTER_PATTERNS = [
    re.compile(r"^.{0,80}\bThis (?:email|message) was sent (?:to|by|from)\b", re.I | re.M),
    re.compile(r"^.{0,80}\bYou(?:'re| are)? receiving this (?:email|message|notification)\b", re.I | re.M),
    re.compile(r"^.{0,80}\bYou (?:are|were) sent this (?:email|message)\b", re.I | re.M),
    re.compile(r"^.{0,80}\bTo unsubscribe\b", re.I | re.M),
    re.compile(r"^.{0,60}\bUnsubscribe\s*(?:[|<\u00b7\u2022]|$)", re.I | re.M),
    re.compile(r"^.{0,80}\bIf you (?:no longer wish|don'?t want|do not want|prefer not)\b", re.I | re.M),
    re.compile(r"^.{0,80}\bupdate your (?:email )?(?:notification )?(?:settings|preferences)\b", re.I | re.M),
    re.compile(r"^.{0,80}\bmanage your (?:email )?(?:notification )?(?:settings|preferences)\b", re.I | re.M),
    re.compile(r"^.{0,80}\bView (?:this email|this message|it) in your browser\b", re.I | re.M),
    re.compile(r"^.{0,80}\bPrivacy Policy\s*[|\u00b7\u2022-]\s*Terms\b", re.I | re.M),
    re.compile(r"^\s*\u00a9\s*\d{4}\b", re.M),
    re.compile(r"^.{0,80}\bAll rights reserved\b", re.I | re.M),
    re.compile(r"^.{0,80}\bplease feel free to give us a review\b", re.I | re.M),
    re.compile(r"^.{0,80}\brate (?:us|your experience) on (?:Yelp|Google)\b", re.I | re.M),
]

_SIG_PATTERNS = [
    re.compile(r"^-- \s*$", re.M),
    re.compile(r"^Sent from my (iPhone|iPad|Android|Samsung|mobile).*$", re.I | re.M),
    re.compile(r"^Get Outlook for (iOS|Android)\s*$", re.I | re.M),
]


def _first_match(text: str, patterns) -> int:
    cut = len(text)
    for pat in patterns:
        m = pat.search(text)
        if m and m.start() < cut:
            cut = m.start()
    return cut


# Lead-forwarding platforms wrap a short customer message in a large, constant
# boilerplate shell. Left intact, that shell dominates the embedding and forces
# every lead into one meaningless cluster, so we keep only the inner message.
_LEAD_SENDER_RE = re.compile(r"@(?:weddingwire|weddingpro|theknot|thumbtack|gigsalad|bark)\.", re.I)

_LEAD_INTRO_RE = re.compile(
    r"(?:check out their\s+message|sent you a new message|"
    r"wants to learn more about your offerings|new message from)\s*:?\s*", re.I)

# Anchored with re.M rather than a literal "\n": _LEAD_INTRO_RE's trailing
# \s* eats the newline before this marker, so a \n-anchored pattern could not
# match when the customer's message was empty -- and the wrapper's "For:" line
# became the body.
_LEAD_END_RE = re.compile(
    r"^\s*(?:For:\s|By replying\b|Reply directly\b|View (?:on|this)\b|"
    r"[A-Z][\w' ]{0,40}'s (?:wedding|event) details\b)", re.I | re.M)

_LEAD_BADGE_RE = re.compile(
    r"^\s*(?:Messaged You First|Replied to you|New Message|Message)\s*$", re.I | re.M)


def unwrap_platform_lead(text: str, sender: str) -> str:
    """Pull the customer's own words out of a platform lead notification."""
    if not sender or not _LEAD_SENDER_RE.search(sender):
        return text
    # Several intro phrases can appear in one notification ("... wants to learn
    # more about your offerings! Check out their message:"). Cut at the last of
    # them so no wrapper text survives into the body.
    matches = list(_LEAD_INTRO_RE.finditer(text))
    if not matches:
        return text
    inner = text[matches[-1].end():]
    end = _LEAD_END_RE.search(inner)
    if end:
        inner = inner[: end.start()]
    inner = _LEAD_BADGE_RE.sub("", inner).strip()
    return inner or text


def strip_quoted(text: str) -> str:
    """Drop quoted replies and signature blocks; keep only what this sender wrote."""
    text = text[: _first_match(text, _QUOTE_PATTERNS)]
    text = text[: _first_match(text, _FOOTER_PATTERNS)]
    text = text[: _first_match(text, _SIG_PATTERNS)]

    kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    text = "\n".join(kept)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
