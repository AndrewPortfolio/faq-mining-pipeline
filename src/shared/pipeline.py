"""Stage-agnostic plumbing shared by every pipeline stage.

Stage 0 (mbox extraction), Stage 1 (Presidio redaction) and Stage 2 (Ollama
embedding) all do the same four things: read input, process it, write JSONL
shards, and be resumable. That loop lives here once so the three stages cannot
drift apart.

Nothing in this module knows what a row contains.
"""

from __future__ import annotations

import glob
import json
import os
import time
from collections import Counter
from typing import Any, Iterator


# --- Reading ---------------------------------------------------------------

SHARD_PREFIX = "emails"


def shard_paths(indir: str, prefix: str = SHARD_PREFIX) -> list[str]:
    """Sorted shard paths, matching ShardWriter's naming exactly.

    The '-' is load-bearing: a bare '*.jsonl' glob would also match sidecar
    files written alongside the shards, and a stage would silently ingest its
    own dedupe log as data.
    """
    return sorted(glob.glob(os.path.join(indir, f"{prefix}-*.jsonl")))


def iter_shard_rows(indir: str, prefix: str = SHARD_PREFIX, skip_shards: int = 0) -> Iterator[dict]:
    """Yield every row across a directory of JSONL shards, in shard order.

    skip_shards resumes past shards a previous run already finished.
    """
    for path in shard_paths(indir, prefix)[skip_shards:]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def count_rows(indir: str, prefix: str = SHARD_PREFIX) -> int:
    total = 0
    for path in shard_paths(indir, prefix):
        with open(path, encoding="utf-8") as fh:
            total += sum(1 for line in fh if line.strip())
    return total


# --- Writing ---------------------------------------------------------------

class ShardWriter:
    """Buffers rows and writes fixed-size JSONL shards atomically.

    Each shard is written to a .part file and renamed into place, so a crash
    mid-write can never leave a half-written shard that later parses as valid.
    """

    def __init__(self, outdir: str, prefix: str = SHARD_PREFIX,
                 shard_size: int = 2000, start_index: int = 0):
        self.outdir = outdir
        self.prefix = prefix
        self.shard_size = shard_size
        self.index = start_index
        self._buf: list[str] = []
        os.makedirs(outdir, exist_ok=True)

    @property
    def pending(self) -> int:
        return len(self._buf)

    def add(self, row: dict) -> bool:
        """Buffer a row. Returns True if that filled a shard and flushed it."""
        self._buf.append(json.dumps(row, ensure_ascii=False) + "\n")
        if len(self._buf) >= self.shard_size:
            self.flush()
            return True
        return False

    def flush(self) -> str | None:
        """Write the buffered rows as the next shard. No-op when empty."""
        if not self._buf:
            return None
        path = os.path.join(self.outdir, f"{self.prefix}-{self.index:05d}.jsonl")
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("".join(self._buf))
        os.replace(tmp, path)
        self.index += 1
        self._buf = []
        return path


# --- Resuming --------------------------------------------------------------

class Checkpoint:
    """Atomically-persisted resume state, written next to the output shards."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict | None:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, **state: Any) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


# --- Reporting -------------------------------------------------------------

class Stats(Counter):
    """Counter that renders as an aligned summary block.

    Being a Counter means an unseen key is 0 rather than a KeyError, so adding
    a new drop reason to a stage cannot crash a long run partway through.
    """

    def render(self, indent: str = "  ") -> str:
        if not self:
            return f"{indent}(nothing recorded)"
        width = max(len(k) for k in self)
        return "\n".join(f"{indent}{k:<{width}}  {v:,}" for k, v in self.items())


class Progress:
    """Periodic throughput/ETA line.

    `position` and `total` are in whatever unit the stage measures itself in --
    bytes of an mbox for Stage 0, rows for later stages.
    """

    def __init__(self, total: float, every: int = 2000,
                 start: float = 0.0, unit: str = "MB", scale: float = 1e6):
        self.total = max(total, 1)
        self.every = every
        self.start = start
        self.unit = unit
        self.scale = scale
        self.ticks = 0
        self.t0 = time.time()

    def tick(self, position: float, note: str = "") -> str | None:
        """Count one item; emit a line every `every` items. Returns it, or None."""
        self.ticks += 1
        if self.ticks % self.every:
            return None
        elapsed = time.time() - self.t0
        rate = (position - self.start) / elapsed / self.scale if elapsed else 0.0
        remaining = (self.total - position) / (rate * self.scale) if rate else 0.0
        line = (f"{self.ticks:>7,} | {100.0 * position / self.total:5.1f}% | "
                f"{rate:6.1f} {self.unit}/s | ETA {remaining / 60:5.1f} min"
                + (f" | {note}" if note else ""))
        print(line, flush=True)
        return line

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60


# --- Deduplication ---------------------------------------------------------

class Deduper:
    """Two-key dedupe (a strong id plus a content fingerprint).

    Resume correctness depends on this state surviving a restart: it is what
    stops a re-read message from being written twice.

    State persists to an append-only log. Rewriting the full key set on every
    flush would make a run quadratic in its own output -- on a 29k-row corpus
    that is 4 MB rewritten once per shard.
    """

    def __init__(self):
        self.ids: set[str] = set()
        self.prints: set[str] = set()
        self._new_ids: list[str] = []
        self._new_prints: list[str] = []

    def check(self, key: str | None, fingerprint: str) -> str | None:
        """Return a drop reason, or None if this row is new. Records it if new."""
        if key and key in self.ids:
            return "dup_id"
        if fingerprint in self.prints:
            return "dup_body"
        if key:
            self.ids.add(key)
            self._new_ids.append(key)
        self.prints.add(fingerprint)
        self._new_prints.append(fingerprint)
        return None

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from an interrupted append
                self.ids.update(rec.get("ids", ()))
                self.prints.update(rec.get("prints", ()))
        self._new_ids.clear()
        self._new_prints.clear()

    def append(self, path: str) -> None:
        """Persist only the keys seen since the last append."""
        if not self._new_ids and not self._new_prints:
            return
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ids": self._new_ids, "prints": self._new_prints}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._new_ids = []
        self._new_prints = []
