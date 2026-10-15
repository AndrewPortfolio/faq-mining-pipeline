# Read input, proccess it, write JSONL shards

# Stage 0 (mbox extraction), Stage 1 (Presidio redaction) and Stage 2 (Ollama
# embedding) uses this pipeline


from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator
from typing import Any

# Old names, kept for anything still importing them; the refactor's Cleanup deletes them
from faq_mining.utils.reporting import Progress, Stats  # noqa: F401

# Reading

SHARD_PREFIX = "emails"


def shard_paths(indir: str, prefix: str = SHARD_PREFIX) -> list[str]:
    # sorted shard paths that guards against other jsonl files from being read as shards
    return sorted(glob.glob(os.path.join(indir, f"{prefix}-*.jsonl")))


def iter_shard_rows(indir: str, prefix: str = SHARD_PREFIX, skip_shards: int = 0) -> Iterator[dict]:
    # get every row across a directory of JSONL shards in order
    # Skips shards a previous run already finished
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


def read_shard(path: str) -> list[dict]:
    # one whole shard, for stages that write one output per input shard
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# Write


def write_atomic(path: str, text: str) -> None:
    # temp file + rename, so a crash never leaves a half-written file behind
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)


class ShardWriter:
    # Buffer rows (holds in mem) and writes (to disk) a fixed-size JSONL
    # Won't leave a shard half written (unless crash --> next run restarts from the beginning)

    def __init__(self, outdir: str, prefix: str = SHARD_PREFIX, shard_size: int = 2000, start_index: int = 0):
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
        # Buffer a row to memory --> once buf holds shard_size rows call flush
        self._buf.append(json.dumps(row, ensure_ascii=False) + "\n")
        if len(self._buf) >= self.shard_size:
            self.flush()
            return True
        return False

    def flush(self) -> str | None:
        # writes all buffered rows as the next jsonl shard
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


# Resuming


class Checkpoint:
    # Atomically-persisted resume state
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


# Deduplication


class Deduper:
    # Two-key dedupe (a strong id plus a content fingerprint)
    # Appends to a log so each save doesn't rewrite every key
    # rewriting would get slower as the output grows

    def __init__(self):
        self.ids: set[str] = set()
        self.prints: set[str] = set()
        self._new_ids: list[str] = []
        self._new_prints: list[str] = []

    def check(self, key: str | None, fingerprint: str) -> str | None:
        # return a drop reason, only appends new key (ID + Fingerprint)
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
        # persist only: it saves only the new ids and fingerprints
        if not self._new_ids and not self._new_prints:
            return
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ids": self._new_ids, "prints": self._new_prints}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._new_ids = []
        self._new_prints = []
