# Stage 2: chunk the redacted emails and embed them with Ollama (nomic-embed-text)
# Reads data/redacted/ and writes data/embeddings/<task>/emails-NNNNN.jsonl, one per input shard
# Each row is one chunk with its text, its email's metadata and its vector, so they can't drift apart
# An email the PII tripwire fires on is never embedded; emails-NNNNN.quarantine.csv says where the hit is
# Usage (all Stage 2; --task only changes the prefix and the output folder):
#     .venv/bin/python src/embed.py                          # -> data/embeddings/clustering/, Stage 3's input
#     .venv/bin/python src/embed.py --task search_document   # -> data/embeddings/search_document/, the RAG index
#     .venv/bin/python src/embed.py --limit 200              # smoke test into <outdir>/smoke
#     .venv/bin/python src/embed.py --force                  # rebuild shards already embedded

from __future__ import annotations

import argparse
import collections
import csv
import functools
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

from shared.pipeline import Checkpoint, Progress, Stats, read_shard, shard_paths, write_atomic


# Configuration

DEFAULT_INDIR = "data/redacted"
DEFAULT_OUTROOT = "data/embeddings"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_URL = "http://localhost:11434"

# nomic-embed-text is trained with a task prefix, and one text embeds differently under each
# (cos 0.93 measured), so every task is its own vector set. search_query: is for the responder's
# incoming messages, never for stored chunks
PREFIXES = {"clustering": "clustering: ", "search_document": "search_document: "}

# ~370 tokens at the measured ~1.2 tokens/word. The installed model's real context is 2048 tokens
# (its Modelfile's num_ctx 8192 is ignored); chunks stay small for retrieval precision, see CLAUDE.md
TARGET_WORDS = 300
MIN_WORDS = 3          # real words a chunk needs once its <ENTITY> tags are gone
BATCH_SIZE = 32        # throughput is flat from 8 to 64 (~27 inputs/s), and a small batch is cheap to redo
REQUEST_TIMEOUT = 120  # the slowest measured batch took 2.5 s, a cold model load 1.1 s
MAX_ATTEMPTS = 3       # per request, then the run stops; finished shards are kept

QUARANTINE_COLUMNS = ["shard", "email_id", "field", "entity_type", "start", "end"]


# Chunking

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TAG_RE = re.compile(r"<[A-Z_]+>")
_WORD_RE = re.compile(r"[^\W\d_]{2,}")  # letters only, accented ones included


def _units(body: str) -> list[tuple[str, bool]]:
    # (text, starts a paragraph) pieces of at most TARGET_WORDS: paragraphs, then sentences for a
    # paragraph that's too long, then even word windows for a sentence that still is
    units = []
    for paragraph in _PARAGRAPH_RE.split(body):
        if len(paragraph.split()) <= TARGET_WORDS:
            if paragraph.strip():
                units.append((paragraph.strip(), True))
            continue
        first = True
        for sentence in _SENTENCE_RE.split(paragraph):
            words = sentence.split()
            if not words:
                continue
            size = math.ceil(len(words) / math.ceil(len(words) / TARGET_WORDS))
            for i in range(0, len(words), size):
                units.append((" ".join(words[i:i + size]), first))
                first = False
    return units


def chunk_text(body: str) -> list[str]:
    # ~92% of bodies fit in one chunk and pass through unchanged
    total = len(body.split())
    if total <= TARGET_WORDS:
        return [body.strip()] if total else []
    # aim for even chunks, so 310 words split 155/155 rather than 300/10
    target = math.ceil(total / math.ceil(total / TARGET_WORDS))
    chunks, current, size = [], "", 0
    for text, new_paragraph in _units(body):
        n = len(text.split())
        if current and size + n > TARGET_WORDS:
            chunks.append(current)
            current, size = "", 0
        if current:
            current += ("\n\n" if new_paragraph else " ") + text
        else:
            current = text
        size += n
        if size >= target:
            chunks.append(current)
            current, size = "", 0
    if current:
        chunks.append(current)
    return chunks


def has_content(text: str) -> bool:
    # "<PERSON>" or "Thanks <PERSON>" would embed to a vector about nothing
    return len(_WORD_RE.findall(_TAG_RE.sub(" ", text))) >= MIN_WORDS


# PII tripwire

# High-precision patterns for PII Stage 1 has already missed in this data, line-wrapped phone numbers
# above all. Names and addresses can't be caught this way and stay Stage 1's job
PII_PATTERNS = {
    "PHONE_NUMBER": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?!\d)"),
    # an Outlook inline-image id (image001.png@01D5A2B3.C4D5E6F0) has an address's shape but no person
    "EMAIL_ADDRESS": re.compile(r"[\w.+-]+@(?![0-9A-Fa-f]{8}\.[0-9A-Fa-f]{8}\b)[\w-]+(?:\.[\w-]+)+"),
    "URL": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
    # must start with a letter, so "@5pm" isn't a handle
    "SOCIAL_HANDLE": re.compile(r"(?<![\w@.<])@[A-Za-z][A-Za-z0-9_.]{1,29}"),
}


def pii_hits(row: dict) -> list[dict]:
    # where each hit is, never what it is, so the quarantine file can't leak what it reports
    hits = []
    for field in ("subject", "body"):
        text = row.get(field) or ""
        for entity, pattern in PII_PATTERNS.items():
            hits.extend({"field": field, "entity_type": entity, "start": m.start(), "end": m.end()}
                        for m in pattern.finditer(text))
    return hits


def render_quarantine(hits: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=QUARANTINE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(hits)
    return buf.getvalue()


# Ollama

class ContextOverflow(Exception):
    # an input longer than the model's context; it only surfaces because requests send truncate=false
    pass


def post_json(url: str, payload: dict | None = None) -> dict:
    # GET when there's no payload. Connection errors, timeouts and 5xx are retried, MAX_ATTEMPTS
    # calls in all; an overflow or any other 4xx is raised at once, since repeating it can't help
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    failure = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            print(f"  {url} failed ({failure}); retry {attempt - 1} of {MAX_ATTEMPTS - 1}",
                  file=sys.stderr)
            time.sleep(2 * (attempt - 1))
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            message = err.read().decode("utf-8", "replace")
            if err.code == 400 and "context length" in message:
                raise ContextOverflow(message) from None
            if err.code < 500:
                raise RuntimeError(f"Ollama answered {err.code} at {url}: {message}") from None
            failure = f"HTTP {err.code}: {message}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            failure = str(getattr(err, "reason", err))
    raise RuntimeError(f"{url} failed {MAX_ATTEMPTS} times, last: {failure}. Is Ollama running?")


def model_digest(url: str, model: str) -> str:
    # goes in the config guard, so vectors from a re-pulled model can't mix with the old ones
    name = model if ":" in model else f"{model}:latest"
    for entry in post_json(f"{url}/api/tags").get("models", []):
        if name in (entry.get("name"), entry.get("model")):
            return entry["digest"]
    raise RuntimeError(f"{name} isn't pulled into Ollama: run `ollama pull {model}`")


def unit(vector: list[float]) -> list[float]:
    # Ollama already returns unit vectors (measured), so this only fixes one that isn't, which keeps
    # Stage 3's Euclidean-as-cosine assumption off the server. Untouched floats keep their short
    # JSON form; rescaled ones would print ~17 digits each
    norm = math.hypot(*vector)
    return vector if not norm or abs(norm - 1.0) < 1e-4 else [x / norm for x in vector]


def embed_batch(texts: list[str], *, url: str, model: str, prefix: str) -> list[list[float]]:
    # truncate=false matters: Ollama's default silently cuts anything past 2048 tokens, so an
    # overflow would never reach embed_pieces to be split
    reply = post_json(f"{url}/api/embed", {"model": model, "input": [prefix + t for t in texts],
                                           "truncate": False})
    vectors = reply.get("embeddings") or []
    if len(vectors) != len(texts):
        raise RuntimeError(f"Ollama sent {len(vectors)} vectors for {len(texts)} inputs")
    return [unit(v) for v in vectors]


def embed_pieces(texts: list[str], embed, stats: Stats) -> list[list[tuple[str, list[float]]]]:
    # One list of (text, vector) per input, nearly always a single pair. A too-long input fails its
    # whole batch without saying which one, so the batch is redone one input at a time, and an input
    # that still overflows is halved by words until both halves fit
    try:
        return [[pair] for pair in zip(texts, embed(texts))]
    except ContextOverflow:
        pass
    if len(texts) > 1:
        return [embed_pieces([text], embed, stats)[0] for text in texts]
    words = texts[0].split()
    if len(words) < 2:
        raise ContextOverflow(f"one {len(texts[0])}-character word is longer than the model's context")
    stats["resplits"] += 1
    half = len(words) // 2
    return [embed_pieces([" ".join(words[:half])], embed, stats)[0]
            + embed_pieces([" ".join(words[half:])], embed, stats)[0]]


# Shards

def embed_shard(rows: list[dict], shard: str, embed, batch_size: int, stats: Stats,
                *, task: str, model: str) -> tuple[list[dict], list[dict]]:
    # returns (output rows, quarantine rows)
    pending, quarantine = [], []  # pending holds (row index, chunk text)
    for index, row in enumerate(rows):
        hits = pii_hits(row)
        if hits:
            stats["quarantined"] += 1
            quarantine.extend({"shard": shard, "email_id": row.get("id") or "", **hit} for hit in hits)
            continue
        chunks = [chunk for chunk in chunk_text(row.get("body") or "") if has_content(chunk)]
        if not chunks:
            stats["no_content"] += 1
            continue
        pending.extend((index, chunk) for chunk in chunks)

    pieces = []
    for start in range(0, len(pending), batch_size):
        batch = [text for _, text in pending[start:start + batch_size]]
        pieces.extend(embed_pieces(batch, embed, stats))

    # a resplit can turn one chunk into several, so chunk numbers are given out only now
    by_row: dict[int, list] = collections.defaultdict(list)
    for (index, _), pairs in zip(pending, pieces):
        by_row[index].extend(pairs)
    out = []
    for index, pairs in by_row.items():
        row = rows[index]
        for i, (text, vector) in enumerate(pairs):
            out.append({
                "chunk_id": f"{row.get('id')}-{i}",
                "id": row.get("id"),
                "thrid": row.get("thrid"),
                "date": row.get("date"),
                "direction": row.get("direction"),
                "subject": row.get("subject"),
                "chunk_index": i,
                "n_chunks": len(pairs),
                "n_words": len(text.split()),
                "text": text,
                "task": task,
                "model": model,
                "embedding": vector,
            })
        stats["emails"] += 1
        stats["chunks"] += len(pairs)
    return out, quarantine


def _rows_in(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


# CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2: chunk redacted emails and embed them with Ollama")
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--task", choices=sorted(PREFIXES), default="clustering",
                    help="nomic task prefix; each task is its own vector set")
    ap.add_argument("--outdir", help=f"default: {DEFAULT_OUTROOT}/<task>")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--url", default=DEFAULT_URL, help="Ollama server")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: embed the first N emails into <outdir>/smoke")
    ap.add_argument("--force", action="store_true", help="rebuild shards already embedded")
    return ap.parse_args(argv)


def run(args) -> int:
    paths = shard_paths(args.indir)
    if not paths:
        print(f"error: no shards in {args.indir}", file=sys.stderr)
        return 1

    # A shard cut short by --limit would look finished to the next run, smoke tests go elsewhere
    outdir = args.outdir or os.path.join(DEFAULT_OUTROOT, args.task)
    if args.limit:
        outdir = os.path.join(outdir, "smoke")
        print(f"smoke test: {args.limit} emails -> {outdir}")
    os.makedirs(outdir, exist_ok=True)

    try:
        digest = model_digest(args.url, args.model)
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    config = {"model": args.model, "digest": digest, "task": args.task,
              "target_words": TARGET_WORDS, "min_words": MIN_WORDS}

    checkpoint = Checkpoint(os.path.join(outdir, "checkpoint.json"))
    rebuild = args.force or bool(args.limit)  # a smoke test always starts over
    if rebuild:
        checkpoint.clear()
    state = checkpoint.load() or {}
    if state.get("config", config) != config:
        # vectors from another model, task or chunking in one set would quietly skew Stage 3
        print(f"error: {outdir} was built with {state['config']}, not {config}; "
              f"rerun with --force to rebuild every shard", file=sys.stderr)
        return 1
    stats = Stats()
    stats.update(state.get("stats", {}))

    todo = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(outdir, f"{name}.jsonl")
        if os.path.exists(out_path) and not rebuild:
            continue
        todo.append((name, path, out_path))
    if len(todo) < len(paths):
        print(f"skipping {len(paths) - len(todo)} shard(s) already embedded")

    embed = functools.partial(embed_batch, url=args.url, model=args.model, prefix=PREFIXES[args.task])
    model = f"{args.model}@{digest[:12]}"
    total = sum(_rows_in(path) for _, path, _ in todo)
    progress = Progress(total=min(total, args.limit) if args.limit else total, every=1,
                        unit="emails", scale=1.0)
    done = 0
    try:
        for name, path, out_path in todo:
            rows = read_shard(path)
            if args.limit:
                rows = rows[:max(0, args.limit - done)]
                if not rows:
                    break
            out, quarantine = embed_shard(rows, name, embed, args.batch_size, stats,
                                          task=args.task, model=model)
            # the quarantine list goes first: the shard's output is what marks it finished
            quarantine_path = os.path.join(outdir, f"{name}.quarantine.csv")
            if quarantine:
                write_atomic(quarantine_path, render_quarantine(quarantine))
            elif os.path.exists(quarantine_path):
                os.remove(quarantine_path)
            write_atomic(out_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
            stats["shards"] += 1
            done += len(rows)
            checkpoint.save(config=config, last_shard=name, stats=dict(stats))
            progress.tick(done, note=f"{name}: {len(out):,} chunks")
    except (RuntimeError, ContextOverflow) as err:
        print(f"\nerror: {err}\nfinished shards are saved; rerun to resume", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; finished shards are saved", file=sys.stderr)

    print(f"\ndone in {progress.elapsed_min():.1f} min -> {outdir}")
    print(stats.render())
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
