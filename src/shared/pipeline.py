# Read input, proccess it, write JSONL shards

#Stage 0 (mbox extraction), Stage 1 (Presidio redaction) and Stage 2 (Ollama
#embedding) uses this pipeline


from __future__ import annotations

import glob
import json
import os
import time
from collections import Counter
from typing import Any, Iterator


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


# Write

class ShardWriter:

    # Buffer rows (holds in mem) and writes (to disk) a fixed-size JSONL
    # Won't leave a shard half written (unless crash --> next run restarts from the beginning)

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


# Reporting 

class Stats(Counter):
    #Counter that renders as an aligned summary block 
    # Because it's a Counter, a missing key counts as 0,
    # so a new drop reason can't crash a long run with a KeyError

    def render(self, indent: str = "  ") -> str:
        if not self:
            return f"{indent}(nothing recorded)"
        width = max(len(k) for k in self)
        return "\n".join(f"{indent}{k:<{width}}  {v:,}" for k, v in self.items())


class Progress:
   # Gives periodic updates on progress
   # In Stage 0, position is in bytes while `every` counts messages,
   # so the two use different units

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
        # count one item, emit a line every `every` items
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


# Deduplication 

class Deduper:
    #Two-key dedupe (a strong id plus a content fingerprint)
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
