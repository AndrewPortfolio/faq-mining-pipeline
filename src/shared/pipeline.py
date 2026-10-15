# Read input, proccess it, write JSONL shards

# Stage 0 (mbox extraction), Stage 1 (Presidio redaction) and Stage 2 (Ollama
# embedding) uses this pipeline


from __future__ import annotations

import json
import os

# Old names, kept for anything still importing them; the refactor's Cleanup deletes them
from faq_mining.storage.shards import (  # noqa: F401
    SHARD_PREFIX,
    Checkpoint,
    ShardWriter,
    count_rows,
    iter_shard_rows,
    read_shard,
    shard_paths,
    write_atomic,
)
from faq_mining.utils.reporting import Progress, Stats  # noqa: F401

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
