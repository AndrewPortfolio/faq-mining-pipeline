#!/usr/bin/env python3
"""Stage 0: stream a Gmail mbox export into compact JSONL shards.

The export is ~98% base64 attachment payload. This pass walks the file once,
discards those bytes without ever decoding them, and emits one JSON row per
message into data/extracted/emails-NNNNN.jsonl.

Everything here is specific to reading mbox. The parts later stages reuse --
shard I/O, checkpointing, progress, text cleaning -- live in src/shared/.

Usage:
    python src/extract_mbox.py                 # full run, resumes if interrupted
    python src/extract_mbox.py --limit 500     # quick smoke test
    python src/extract_mbox.py --restart       # ignore checkpoint, start over
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import os
import quopri
import re
import sys
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime

from shared.pipeline import Checkpoint, Deduper, Progress, ShardWriter, Stats
from shared.textclean import html_to_text, strip_quoted, unwrap_platform_lead


# --- Configuration ---------------------------------------------------------

DEFAULT_MBOX = "data/raw/All mail Including Spam and Trash.mbox"
DEFAULT_OUTDIR = "data/extracted"
SHARD_SIZE = 2000

# Gmail labels whose messages are dropped outright. Everything else is kept
# with its labels intact so filtering stays possible downstream.
DROP_LABELS = {"Spam", "Trash"}

# A message carrying this label is our own outgoing mail.
SENT_LABEL = "Sent"

# Automated/bulk mail detection. Header signals first -- List-Unsubscribe and
# Precedence are what marketing and notification senders are actually required
# to set, so they generalise far better than matching on body text.
BULK_PRECEDENCE = {"bulk", "list", "junk", "auto_reply"}

_NOREPLY_RE = re.compile(
    r"(?:^|[.\-_+])(?:no-?reply|do-?not-?reply|donotreply|notification[s]?|"
    r"mailer-daemon|postmaster|bounce[s]?|auto-?confirm|automailer)(?:[.\-_+]|@)",
    re.I,
)

# Marketing subdomains (e.g. info@marketing.moviepass.com). Kept narrow on
# purpose: lead-forwarding platforms such as weddingwire.com send genuine
# customer enquiries and must not be caught here.
_MARKETING_DOMAIN_RE = re.compile(
    r"@(?:marketing|mktg|campaigns?|promo|newsletter|news|clicks|links)\.", re.I)

# Senders that are unambiguously machine traffic for this mailbox.
_NOISE_DOMAIN_RE = re.compile(
    r"@(?:txt\.voice\.google\.com|.*\.bounces\.google\.com|facebookmail\.com|"
    r"parastorage\.com|wix-forms\.com|messaging\.squareup\.com)$",
    re.I,
)

# Verified against 800 MB of this export: 421 '^From ' lines, 421 matches,
# zero false positives. Gmail writes the thread id as the envelope sender.
ENVELOPE_RE = re.compile(rb"^From \d+@xxx ")

PROGRESS_EVERY = 2000


# --- Part decoding ---------------------------------------------------------

def decode_part(raw: bytes, encoding: str, charset: str) -> str:
    enc = (encoding or "").strip().lower()
    if enc == "base64":
        try:
            raw = base64.b64decode(b"".join(raw.split()), validate=False)
        except (binascii.Error, ValueError):
            return ""
    elif enc == "quoted-printable":
        try:
            raw = quopri.decodestring(raw)
        except Exception:
            pass

    for cs in (charset, "utf-8", "cp1252"):
        if not cs:
            continue
        try:
            return raw.decode(cs, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue
    # latin-1 maps every byte, so this always succeeds and is the real fallback.
    return raw.decode("latin-1")


# --- Streaming mbox reader -------------------------------------------------

def _parse_headers(raw: bytes):
    return BytesParser(policy=policy.default).parsebytes(raw)


def _ct_params(value: str):
    """Split a Content-Type/Disposition value into (main, {params})."""
    parts = value.split(";")
    main = parts[0].strip().lower()
    params = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, _, v = p.partition("=")
        params[k.strip().lower()] = v.strip().strip('"').strip("'")
    return main, params


class _MessageBuilder:
    """Line-level MIME walker.

    Keeps text/* parts, and for everything else records metadata while
    discarding the bytes. Non-text lines never leave this class, which is
    where the memory and speed win comes from.
    """

    __slots__ = (
        "offset", "end_offset", "header_lines", "in_headers", "boundaries", "in_part_headers",
        "part_header_lines", "keep", "buf", "parts", "attachments",
        "cur_ct", "cur_cte", "cur_charset", "cur_name", "skip_bytes", "headers",
    )

    def __init__(self, offset: int):
        self.offset = offset
        # Byte offset just past this message; set by iter_messages once the
        # next envelope line (or EOF) is reached. This is the resume point.
        self.end_offset = offset
        self.header_lines: list[bytes] = []
        self.in_headers = True
        self.headers = None
        self.boundaries: set[bytes] = set()
        self.in_part_headers = False
        self.part_header_lines: list[bytes] = []
        self.keep = False
        self.buf: list[bytes] = []
        self.parts: list[tuple] = []
        self.attachments: list[dict] = []
        self.cur_ct = "text/plain"
        self.cur_cte = ""
        self.cur_charset = "utf-8"
        self.cur_name = None
        self.skip_bytes = 0

    # -- part bookkeeping --

    def _flush_part(self):
        if self.keep and self.buf:
            self.parts.append(
                (self.cur_ct, self.cur_cte, self.cur_charset, b"".join(self.buf))
            )
        elif self.cur_ct and not self.cur_ct.startswith("multipart/") and self.skip_bytes:
            approx = self.skip_bytes
            if (self.cur_cte or "").lower() == "base64":
                approx = int(approx * 0.75)
            self.attachments.append({
                "filename": self.cur_name,
                "content_type": self.cur_ct,
                "approx_bytes": approx,
            })
        self.buf = []
        self.skip_bytes = 0

    def _apply_part_headers(self, msg):
        ct_raw = msg.get("Content-Type", "text/plain")
        main, params = _ct_params(str(ct_raw))
        cte = str(msg.get("Content-Transfer-Encoding", "")).strip()
        disp_raw = str(msg.get("Content-Disposition", ""))
        _, disp_params = _ct_params(disp_raw) if disp_raw else ("", {})

        self.cur_ct = main
        self.cur_cte = cte
        self.cur_charset = params.get("charset", "utf-8")
        self.cur_name = disp_params.get("filename") or params.get("name")

        if main.startswith("multipart/"):
            b = params.get("boundary")
            if b:
                self.boundaries.add(b.encode("utf-8", "replace"))
            self.keep = False
        else:
            self.keep = main.startswith("text/")

    def _is_boundary(self, line: bytes) -> bool:
        if not self.boundaries:
            return False
        s = line.rstrip(b"\r\n")[2:]
        if s.endswith(b"--"):
            s = s[:-2]
        return s in self.boundaries

    # -- the hot loop --

    def feed(self, line: bytes):
        if self.in_headers:
            if line in (b"\n", b"\r\n", b""):
                self.in_headers = False
                self.headers = _parse_headers(b"".join(self.header_lines))
                self._apply_part_headers(self.headers)
            else:
                self.header_lines.append(line)
            return

        if line.startswith(b"--") and self._is_boundary(line):
            self._flush_part()
            self.in_part_headers = True
            self.part_header_lines = []
            self.keep = False
            self.cur_ct = ""
            self.cur_name = None
            return

        if self.in_part_headers:
            if line in (b"\n", b"\r\n", b""):
                self.in_part_headers = False
                self._apply_part_headers(_parse_headers(b"".join(self.part_header_lines)))
            else:
                self.part_header_lines.append(line)
            return

        if self.keep:
            self.buf.append(line)
        else:
            self.skip_bytes += len(line)

    def finish(self):
        if self.in_headers and self.header_lines:
            self.headers = _parse_headers(b"".join(self.header_lines))
            self._apply_part_headers(self.headers)
        self._flush_part()
        return self


def iter_messages(path: str, start_offset: int = 0):
    """Yield finished _MessageBuilder objects, one per message."""
    with open(path, "rb") as fh:
        if start_offset:
            fh.seek(start_offset)

        cur = None
        offset = start_offset
        for line in fh:
            if line.startswith(b"From ") and ENVELOPE_RE.match(line):
                if cur is not None:
                    cur.end_offset = offset
                    yield cur.finish()
                cur = _MessageBuilder(offset)
            elif cur is not None:
                cur.feed(line)
            offset += len(line)

        if cur is not None:
            cur.end_offset = offset
            yield cur.finish()



# --- Row construction ------------------------------------------------------

def pick_body(builder: _MessageBuilder):
    """Prefer text/plain; fall back to rendering the HTML alternative."""
    plain = [p for p in builder.parts if p[0] == "text/plain"]
    if plain:
        text = "\n".join(decode_part(p[3], p[1], p[2]) for p in plain)
        if text.strip():
            return text, "text/plain"

    html = [p for p in builder.parts if p[0] in ("text/html", "text/x-amp-html")]
    if html:
        raw = "\n".join(decode_part(p[3], p[1], p[2]) for p in html)
        return html_to_text(raw), "text/html"

    return "", "none"


def _hdr(msg, name: str) -> str:
    try:
        v = msg.get(name)
        return str(v).strip() if v is not None else ""
    except Exception:
        return ""


def sender_address(raw: str) -> str:
    m = re.search(r"[\w.+-]+@[\w.-]+", raw or "")
    return m.group(0).lower() if m else ""


def is_automated(msg, sender: str) -> bool:
    """True for bulk/marketing/notification mail, which is template-identical
    and would otherwise form the densest clusters in the corpus."""
    if msg.get("List-Unsubscribe") or msg.get("List-Id"):
        return True
    if str(msg.get("Precedence", "")).strip().lower() in BULK_PRECEDENCE:
        return True
    auto = str(msg.get("Auto-Submitted", "")).strip().lower()
    if auto and auto != "no":
        return True
    if not sender:
        return False
    return bool(
        _NOREPLY_RE.search(sender)
        or _NOISE_DOMAIN_RE.search(sender)
        or _MARKETING_DOMAIN_RE.search(sender)
    )


def build_row(builder: _MessageBuilder, keep_automated: bool = False):
    msg = builder.headers
    if msg is None:
        return None, "no_headers"

    labels = [l.strip() for l in _hdr(msg, "X-Gmail-Labels").split(",") if l.strip()]
    if DROP_LABELS.intersection(labels):
        return None, "dropped_label"

    sender = sender_address(_hdr(msg, "From"))
    automated = is_automated(msg, sender)
    if automated and not keep_automated:
        return None, "automated"

    raw_body, source = pick_body(builder)
    body = strip_quoted(unwrap_platform_lead(raw_body, sender))
    if not body:
        return None, "empty_body"

    date_raw = _hdr(msg, "Date")
    try:
        date_iso = parsedate_to_datetime(date_raw).isoformat() if date_raw else None
    except (TypeError, ValueError):
        date_iso = None

    row = {
        "id": _hdr(msg, "Message-ID") or None,
        "thrid": _hdr(msg, "X-GM-THRID") or None,
        "date": date_iso,
        "direction": "outbound" if SENT_LABEL in labels else "inbound",
        "labels": labels,
        "from": _hdr(msg, "From"),
        "sender": sender,
        "is_automated": automated,
        "to": _hdr(msg, "To"),
        "subject": _hdr(msg, "Subject"),
        "body": body,
        "n_words": len(body.split()),
        "body_source": source,
        "n_attachments": len(builder.attachments),
        "attachments": builder.attachments,
        "offset": builder.offset,
    }
    return row, None


def body_fingerprint(row: dict) -> str:
    norm = re.sub(r"\s+", " ", row["body"]).strip().lower()
    return hashlib.sha256(f"{row['subject'].strip().lower()}\x00{norm}".encode()).hexdigest()

def body_fingerprint(row: dict) -> str:
    norm = re.sub(r"\s+", " ", row["body"]).strip().lower()
    return hashlib.sha256(
        f"{row['subject'].strip().lower()}\x00{norm}".encode()
    ).hexdigest()


# --- CLI -------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 0: mbox -> JSONL shards")
    ap.add_argument("--mbox", default=DEFAULT_MBOX)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="stop after N messages read")
    ap.add_argument("--restart", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--keep-automated", action="store_true",
                    help="retain bulk/notification mail instead of dropping it")
    return ap.parse_args(argv)


def run(args) -> int:
    if not os.path.exists(args.mbox):
        print(f"error: mbox not found: {args.mbox}", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    checkpoint = Checkpoint(os.path.join(args.outdir, "checkpoint.json"))
    seen_log = os.path.join(args.outdir, "seen.log")

    stats = Stats()
    deduper = Deduper()
    start_offset, shard_index = 0, 0

    state = None if args.restart else checkpoint.load()
    if args.restart:
        checkpoint.clear()
        if os.path.exists(seen_log):
            os.remove(seen_log)
    elif state:
        start_offset = state["offset"]
        shard_index = state["next_shard"]
        stats.update(state.get("stats", {}))
        deduper.load(seen_log)
        print(f"resuming at byte {start_offset:,} (shard {shard_index}, "
              f"{len(deduper.prints):,} rows already seen)")

    writer = ShardWriter(args.outdir, "emails", args.shard_size, shard_index)
    progress = Progress(total=os.path.getsize(args.mbox), every=PROGRESS_EVERY,
                        start=start_offset)
    resume_offset = start_offset

    def persist(offset: int) -> None:
        """Record everything needed to resume exactly here."""
        deduper.append(seen_log)
        checkpoint.save(offset=offset, next_shard=writer.index, stats=dict(stats))

    try:
        for builder in iter_messages(args.mbox, start_offset):
            stats["read"] += 1
            resume_offset = builder.end_offset

            row, reason = build_row(builder, keep_automated=args.keep_automated)
            if reason:
                stats[reason] += 1
            else:
                dup = deduper.check(row["id"], body_fingerprint(row))
                if dup:
                    stats[dup] += 1
                else:
                    stats["written"] += 1
                    stats[row["direction"]] += 1
                    if writer.add(row):
                        # A shard just closed; resume must restart *after* this
                        # message, not at it, or the next run re-reads it.
                        persist(builder.end_offset)

            progress.tick(builder.end_offset, note=f"kept {stats['written']:,}")

            if args.limit and stats["read"] >= args.limit:
                break
    except KeyboardInterrupt:
        print("\ninterrupted; flushing partial shard", file=sys.stderr)

    writer.flush()
    persist(resume_offset)

    print(f"\ndone in {progress.elapsed_min():.1f} min -> {args.outdir}")
    print(stats.render())
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
