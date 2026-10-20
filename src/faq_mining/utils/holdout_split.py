# The test holdout's rule: about a tenth of all threads, picked by a seeded hash of the thread id
# Stage 2 never embeds them, so no later stage (clusters, triage, FAQs, the RAG index) can see them
# A hash rather than a stored list: every stage gets the same answer, with no file to keep in sync or run first

import hashlib

SEED = 42
SHARE = 0.10  # 732 of 7,171 threads on 2026-10-20, ~380 of them with a client email to label


def rank(thrid: str, seed: int = SEED) -> float:
    # a fixed, evenly spread number in [0, 1) per thread. sha256, not hash(): Python salts hash() per process
    digest = hashlib.sha256(f"{seed}:{thrid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_held_out(thrid: str, seed: int = SEED, share: float = SHARE) -> bool:
    return rank(thrid, seed) < share
