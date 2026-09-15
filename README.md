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

