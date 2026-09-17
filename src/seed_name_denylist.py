# Stage 1 prep: seed the editable name deny-list used by the PII recognizer
# spaCy's English NER misses a lot of Vietnamese/Chinese names, so Presidio also gets an explicit list
# From/To display names are known people, so harvesting them needs no NER guesswork
# Writes data/pii/name_denylist.txt --> holds real client names
# Usage:
#     python src/seed_name_denylist.py           # refuses to overwrite edited list in data 
#     python src/seed_name_denylist.py --force   # regenerate from scratch

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter
from email.utils import getaddresses
from typing import Iterable

from shared.pipeline import iter_shard_rows, shard_paths


# Configuration

DEFAULT_INDIR = "data/extracted"
DEFAULT_OUT = "data/pii/name_denylist.txt"
DEFAULT_DICT = "/usr/share/dict/words"

# Diacritics folded (Trần -> Tran) so one entry covers both spellings when matching headers
VIETNAMESE_SURNAMES = {
    "Bui", "Cao", "Chau", "Dam", "Dang", "Diep", "Dinh", "Do", "Doan", "Duong", "Giang", "Ha",
    "Ho", "Hoang", "Hua", "Huynh", "Kha", "Khuu", "Kieu", "La", "Lai", "Lam", "Le", "Luong",
    "Luu", "Ly", "Mac", "Mai", "Ngo", "Nguyen", "Pham", "Phan", "Phung", "Quach", "Ta", "Tang",
    "Thach", "Thai", "Tieu", "Ton", "Tong", "Tran", "Trieu", "Trinh", "Truong", "Ung", "Vo",
    "Vu", "Vuong",
}

# Mandarin pinyin plus the Cantonese/Hokkien/Wade-Giles spellings common in the US
CHINESE_SURNAMES = {
    "Bai", "Cai", "Cao", "Chan", "Chang", "Chen", "Cheng", "Cheung", "Chiang", "Chien", "Chiu",
    "Choi", "Chong", "Chow", "Chu", "Chua", "Chung", "Deng", "Ding", "Dong", "Du", "Duan",
    "Fang", "Feng", "Fong", "Fu", "Fung", "Gao", "Goh", "Gu", "Guo", "Han", "Hao", "He", "Hsieh",
    "Hsu", "Hu", "Huang", "Hui", "Hung", "Jiang", "Jin", "Koh", "Kong", "Kuo", "Kwan", "Kwok",
    "Kwong", "Lau", "Law", "Lee", "Lei", "Leung", "Li", "Liang", "Liao", "Lim", "Lin", "Liu",
    "Lo", "Loh", "Lu", "Lui", "Luo", "Ma", "Mak", "Mao", "Meng", "Ng", "Ong", "Pan", "Peng",
    "Poon", "Qian", "Qin", "Qiu", "Ren", "Shen", "Shi", "Siu", "Song", "Su", "Sun", "Tam", "Tan",
    "Tang", "Tay", "Teo", "Tian", "Tsai", "Tsang", "Tse", "Wang", "Wei", "Wong", "Wu", "Xiao",
    "Xie", "Xiong", "Xu", "Yang", "Yao", "Ye", "Yee", "Yeung", "Yin", "Yip", "Yu", "Yuan",
    "Zeng", "Zhang", "Zhao", "Zheng", "Zhong", "Zhou", "Zhu",
}

SURNAMES = VIETNAMESE_SURNAMES | CHINESE_SURNAMES

# Picked from this inbox: at 10 the ambiguous surnames are real words (Do, He, Song, Law);
# lower thresholds start pulling in Lee and Chang
MIN_LOWERCASE_EMAILS = 10

# A name part: 2+ letters with an optional inner hyphen/apostrophe (Mei-Ling); initials drop out
_NAME_PART_RE = re.compile(r"[^\W\d_]{2,}(?:[-'][^\W\d_]+)*")
_SPLIT_RE = re.compile(r"[\s,;\"()<>]+")
_ADDRESS_RE = re.compile(r"\S*@\S+|https?://\S+|www\.\S+")
_WORD_RE = re.compile(r"[^\W\d_]+")

HEADER = """\
# Name deny-list for the PII recognizer
# Holds real client names: keep it under data/, which is gitignored
# One name per line, matched case-sensitively; anything after "#" is ignored
# The number is how many emails had the name in a From/To display name (0 = built-in surname, not seen)
# Prune anything that isn't a person's name, e.g. "Photography" from "Tran Photography"
# [names]      flagged as PERSON
# [ambiguous]  also an everyday English word in this inbox (Do, He, Song): flagged at a lower score so review can sort them out
# The seed script won't overwrite this file without --force, so edits here are safe
"""


# Name parsing

def fold(text: str) -> str:
    # đ has no Unicode decomposition, so NFKD alone would leave it in
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def name_parts(display: str) -> list[str]:
    # NFC first: decomposed accents aren't letters, so "Trần" would otherwise split mid-word
    # Title-case so "NGUYEN THAO" and "Nguyen Thao" land on one entry
    display = unicodedata.normalize("NFC", display)
    return [p.title() for p in _SPLIT_RE.split(display) if _NAME_PART_RE.fullmatch(p)]


def harvest(rows: Iterable[dict], surnames: set[str] = SURNAMES) -> Counter:
    # emails per name part, for From/To display names that contain a known surname
    # Given names ride along with the surname because greetings use them alone ("Hi Thao")
    counts: Counter = Counter()
    for row in rows:
        found: set[str] = set()
        for display, _ in getaddresses([row.get("from") or "", row.get("to") or ""]):
            parts = name_parts(display)
            if any(fold(p) in surnames for p in parts):
                # both spellings: a body can drop the diacritics its header used, or add them
                found.update(parts)
                found.update(fold(p) for p in parts)
        counts.update(found)
    return counts


def load_english(path: str) -> set[str]:
    # lowercase entries only: a capitalized entry is a proper noun, which won't read as an ordinary word
    if not os.path.exists(path):
        print(f"warning: {path} not found; nothing will be marked ambiguous", file=sys.stderr)
        return set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return {w for w in (line.strip() for line in fh) if w.islower()}


def english_in_use(english: set[str], rows: Iterable[dict],
                   min_emails: int = MIN_LOWERCASE_EMAILS) -> set[str]:
    # Needs both signals: the full dictionary alone flags archaic entries like "wang" and "yang",
    # and inbox usage alone flags names written in lowercase
    usage: Counter = Counter()
    for row in rows:
        # addresses, handles and URLs spell names in lowercase (thao.nguyen@...), so cut them first
        text = _ADDRESS_RE.sub(" ", f"{row.get('subject') or ''} {row.get('body') or ''}")
        usage.update({w for w in _WORD_RE.findall(text) if w.islower()})
    return {w for w, n in usage.items() if n >= min_emails and w in english}


def split_entries(counts: Counter, surnames: set[str], english: set[str]):
    # Seed surnames stay in with zero hits: bodies mention people (partners, parents) who never emailed
    entries = Counter(dict.fromkeys(surnames, 0))
    entries.update(counts)
    names, ambiguous = [], []
    for part, n in sorted(entries.items(), key=lambda kv: (-kv[1], kv[0])):
        (ambiguous if part.lower() in english else names).append((part, n))
    return names, ambiguous


def render(names, ambiguous) -> str:
    def lines(entries):
        return "".join(f"{part:<24}# {n}\n" for part, n in entries)
    return f"{HEADER}\n[names]\n{lines(names)}\n[ambiguous]\n{lines(ambiguous)}"


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1 prep: seed the name deny-list")
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dict", default=DEFAULT_DICT, help="word list used to mark ambiguous names")
    ap.add_argument("--force", action="store_true", help="overwrite an existing (possibly edited) list")
    return ap.parse_args(argv)


def run(args) -> int:
    if not shard_paths(args.indir):
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1
    if os.path.exists(args.out) and not args.force:
        print(f"error: {args.out} exists and may hold your edits; pass --force to regenerate",
              file=sys.stderr)
        return 1

    counts = harvest(iter_shard_rows(args.indir))
    english = english_in_use(load_english(args.dict), iter_shard_rows(args.indir))
    names, ambiguous = split_entries(counts, SURNAMES, english)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(render(names, ambiguous))
    os.replace(tmp, args.out)

    # counts only: the list is real client names, keep them out of terminal scrollback
    print(f"{len(names):,} names, {len(ambiguous):,} ambiguous "
          f"({len(counts):,} harvested from headers, {len(SURNAMES):,} seed surnames) -> {args.out}")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
