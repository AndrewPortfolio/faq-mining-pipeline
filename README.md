# Faq-mining-pipeline
Given email text data, this program will clean the data of footers, spam, trash, and marketing/automated emails, and strip PII. It will then be converted into vector embeddings using Ollama (nomic-embed-text), and use HDBSCAN to find FAQs. 

## Stage 0: Data Extraction + Sharding 
Reads the Gmail Takeout mbox and writes client conversations to `data/extracted/emails-NNNNN.jsonl`.

- **Filters out:** Spam/Trash (Gmail labels), automated mail (bulk/list headers, no-reply and marketing senders), and retired Bark leads.
- **Cleans bodies:** skips attachment payloads, falls back to HTML→text when there's no plain text, strips quoted replies/signatures/footers, and unwraps WeddingWire lead templates.
- **Dedupes** by Message-ID and by a subject+body fingerprint.
- **Shards** at 2,000 rows each. Each shard is written to a temp file and renamed, so a crash never leaves a half-written shard.
- **Resumable:** a checkpoint lets an interrupted run pick up where it stopped (`--restart` to start over, `--limit N` for a quick test).

Output still contains PII. It's removed in Phase 1.


## Stage 1: Presidio PII Redaction
Reads the shards from Stage 0 (`data/extracted/`) and produces `data/redacted/emails-NNNNN.jsonl`.
Two separate files, so PII aren't removed without my review.

### 1a. `analyze_pii.py` — flag, don't touch
Runs Presidio (spaCy `en_core_web_trf`) over each email's subject and body and writes, a review per shard. DOES NOT WRITE OVER EMAIL TEXT

- **Flags:** PERSON, EMAIL_ADDRESS, PHONE_NUMBER, DATE_TIME, LOCATION, ORGANIZATION, URL,
  CREDIT_CARD, US_SSN, STREET_ADDRESS, SOCIAL_HANDLE (`@handle`).
- **Custom recognizers** (`src/shared/pii_recognizers.py`) fill gaps the English model has:
  a Vietnamese/Chinese name deny-list (`data/pii/name_denylist.txt`), a venue deny-list
  (`data/pii/venue_denylist.txt`), a US street-address pattern, and `@handle` matching.
- **`data/pii/allowlist.txt`** suppresses known non-PII (WeddingWire, Venmo, …) before it
  ever reaches the review file.
- **Outputs per shard**, in `data/review/`: `emails-NNNNN.spans.csv` (manually edit 
  each row. Each row has a `decision` column, pre-filled `redact`, can flip to `keep` for false
  positives), a `.spans.orig.csv` snapshot used to catch deleted rows, and a `.view.txt`
  with every email in the shard shown with its spans marked to see what
  *wasn't* flagged.
- **Resumable per shard:** skips a shard once its review file exists, and never overwrites
  once edited, even with `--force`.

### 1b. `apply_redactions.py` — redact on rows labeled `redact` in 1a
Reads edited review files and writes the redacted shards. Refuses to write a shard
(with a clear error, nothing written) if a row was deleted from the CSV instead of marked
`keep`, a `decision` isn't `redact`/`keep`, a span's text no longer matches, or part of the
shard was never analyzed — each of those would otherwise let PII through silently.

- **Redacts:** every `redact` span is replaced with a `<ENTITY_TYPE>` tag (e.g. `<PERSON>`).
- **Hashes, not removes:** `sender` and `to` addresses and attachment filenames are hashed
  with a keyed HMAC (`data/pii/hash_key`) so the same person hashes the same way across
  shards; `id` (Message-ID) is hashed with plain SHA-256. `from` is dropped — its only
  extra content over `sender` was the display name.
- **Kept as-is:** `thrid`, `date`, `direction`, `labels` — nothing here identifies a
  person, and Stage 2 clustering needs them.
