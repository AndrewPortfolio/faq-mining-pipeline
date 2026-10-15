# Read and write JSONL shards, plus the checkpoint a stage resumes from

import glob
import json
import os
from collections.abc import Iterator
from typing import Any

SHARD_PREFIX = "emails"


def shard_paths(indir: str, prefix: str = SHARD_PREFIX) -> list[str]:
    # sorted shard paths that stops other jsonl files from being read as shards
    return sorted(glob.glob(os.path.join(indir, f"{prefix}-*.jsonl")))


def iter_shard_rows(indir: str, prefix: str = SHARD_PREFIX, skip_shards: int = 0) -> Iterator[dict[str, Any]]:
    # get every row across a directory of JSONL shards in order
    # skips shards a previous run already finished
    for path in shard_paths(indir, prefix)[skip_shards:]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def count_rows(indir: str, prefix: str = SHARD_PREFIX) -> int:
    # total non-blank rows across every shard, used as Progress's total
    total = 0
    for path in shard_paths(indir, prefix):
        with open(path, encoding="utf-8") as fh:
            total += sum(1 for line in fh if line.strip())
    return total


def read_shard(path: str) -> list[dict[str, Any]]:
    # one whole shard, for stages that write one output per input shard
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# The rename is atomic, so a crash leaves either the old or new file, never half a file
def write_atomic(path: str, text: str) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)


# Checkpoint stores atomically-persisted resume state in one JSON file
class Checkpoint:
    def __init__(self, path: str) -> None:
        self.path = path

    # unreadable checkpoint means start over not crash
    def load(self) -> dict[str, Any] | None:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, **state: object) -> None:
        text = json.dumps(state)
        write_atomic(self.path, text)

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


# buffers rows and writes them out as fixed-size shards
class ShardWriter:
    # Stores the settings and creates outdir if missing
    def __init__(self, outdir: str, prefix: str = SHARD_PREFIX, shard_size: int = 2000, start_index: int = 0) -> None:
        self.outdir = outdir
        self.prefix = prefix
        self.shard_size = shard_size
        self.index = start_index
        self._buf: list[str] = []
        os.makedirs(outdir, exist_ok=True)

    @property
    def pending(self) -> int:
        return len(self._buf)

    # ensure_ascii=False so vietnamese stays readable
    def add(self, row: dict[str, Any]) -> bool:
        self._buf.append(json.dumps(row, ensure_ascii=False) + "\n")
        if len(self._buf) >= self.shard_size:
            self.flush()
            return True
        return False

    # Once buff is full or finished with shards --> flush and write rows to json
    def flush(self) -> str | None:
        if not self._buf:
            return None

        path = os.path.join(self.outdir, f"{self.prefix}-{self.index:05d}.jsonl")
        buf_s = "".join(self._buf)
        write_atomic(path, buf_s)
        self.index += 1
        self._buf = []
        return path
