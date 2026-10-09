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

- **Entities:** PERSON, EMAIL_ADDRESS, PHONE_NUMBER, DATE_TIME, LOCATION, ORGANIZATION, URL,
  CREDIT_CARD, US_SSN, STREET_ADDRESS, SOCIAL_HANDLE (`@handle`), US_EIN, INSURANCE_POLICY,
  ORDER_ID, plus whole SIGNATURE blocks (below). `US_DRIVER_LICENSE` and `NRP` are left out on
  purpose: the first fires on ordinary alphanumerics, the second would redact FAQ content
  ("Vietnamese tea ceremony").
- **Custom recognizers** (`src/shared/pii_recognizers.py`) fill gaps the English model has:
  a Vietnamese/Chinese name deny-list (`data/pii/name_denylist.txt`), a venue deny-list
  (`data/pii/venue_denylist.txt`), a place deny-list seeded by `seed_place_denylist.py`
  (`data/pii/place_denylist.txt`), plus patterns for street addresses, `@handle`s, zip codes,
  venue names, EINs, policy numbers, order/invoice/tracking IDs, and month-first dotted dates.
  Names that are also English words ("An", "To", "My") are scored from how spaCy reads them in
  the sentence, so "Hi An," redacts and "An apple" doesn't.
- **Pre-filled decisions** (`src/shared/decisions.py`): each row's `redact`/`keep` and the `rule`
  that fired are written to the CSV so a surprising `keep` can be traced.
  - Always redacted: names, URLs, every location (region words in
    `data/pii/common_words_location.txt` excepted), and specific calendar dates.
  - Kept: clock times, durations ("15 mins"), bare years, and common words in
    `data/pii/common_words_datetime.txt`. Could be FAQ content and don't identify anyone.
- **Signature blocks** (`src/shared/signatures.py`): the tail of a message packed with contact
  details (or a sign-off + name line) is flagged as one `SIGNATURE` row, because titles,
  credentials and taglines identify someone even after their name is gone. Questions, lines in the
  sender's own voice, and form fields are never included.
- **`data/pii/allowlist.txt`** suppresses known non-PII (WeddingWire, Venmo, …) before it
  ever reaches the review file.
- **Outputs per shard**, in `data/review/`: `emails-NNNNN.spans.csv` (edit
  each row. Each row has a `decision` column and a `rule` column, `decision` can flip to `keep`
  for false positives), a `.spans.orig.csv` snapshot used to catch deleted rows, and a `.view.txt`
  with every email in the shard shown with its spans marked to see what
  *wasn't* flagged.
- **Resumable per shard:** skips a shard once its review file exists, and never overwrites
  once edited, even with `--force`.

### 1b. `apply_redactions.py` — redact on rows labeled `redact` in 1a
Reads edited review files and writes the redacted shards. Refuses to write a shard
(clear error, nothing written) if a row was deleted from the CSV instead of marked
`redact`/`keep`, a `decision` isn't `redact`/`keep`, or a span's text no longer matches,
since each would let PII through silently.

- **Redacts:** every `redact` span is replaced with a `<ENTITY_TYPE>` tag (e.g. `<PERSON>`);
  a `SIGNATURE` span is deleted outright and the blank lines it leaves are tidied.
- **Hashes, not removes:** `sender` and `to` addresses and attachment filenames are hashed
  with a keyed HMAC (`data/pii/hash_key`) so the same person hashes the same way across
  shards; `id` (Message-ID) is hashed with plain SHA-256. `from` is dropped — its only
  extra content over `sender` was the display name.
- **Kept as-is:** `thrid`, `date`, `direction`, `labels` — nothing here identifies a
  person, and Stage 2 clustering needs them.

### 1c. `pii_sample.py` — confidence read before the real run
Draws whole emails (one per top PERSON term, the rest uniformly) into `data/review/sample.txt`
to catch PII no recognizer flagged and FAQ content wrongly marked `redact`. Uses the whole
email, not just spans, because a missed PII never becomes a span.

### Stage 1 workflow
1. `python src/analyze_pii.py` — writes the review files (~1.1 h for all shards)
2. Review `data/review/*.spans.csv` and `sample.txt`
3. `python src/apply_redactions.py` — writes `data/redacted/`

Presidio is done: Stage 2 (embedding) reads `data/redacted/`.


## Stage 2: Chunking + Embedding
Reads the redacted shards (`data/redacted/`) and writes `data/embeddings/<task>/emails-NNNNN.jsonl`,
one per input shard. Each row is one chunk: its text, its email's `id`/`thrid`/`date`/`direction`/
`subject`, and its vector, kept in one row so they can't drift apart.

- **Model:** Ollama `nomic-embed-text` (768-d). Its real context is 2048 tokens (the Modelfile's
  `num_ctx 8192` is ignored), and Ollama silently cuts longer input by default, so every request
  sends `truncate: false`: an overflow comes back as an error and that chunk is split in half instead.
- **Task prefixes:** the model is trained with them, and the same text embeds differently under each,
  so each task is its own vector set: `--task clustering` (default, for Stage 3) and
  `--task search_document` (the RAG index). The responder adds `search_query:` to incoming messages.
- **Chunks:** ~300 words, split at paragraphs, then sentences, then words, so most emails stay one
  chunk. A chunk with fewer than 3 real words once its `<ENTITY>` tags are gone is skipped. Only the
  body is embedded: most subjects are `Re:` threads or form templates.
- **PII tripwire:** a phone number (line-wrapped ones too), email address, URL or `@handle` that
  Stage 1 missed sends the whole email to quarantine, so it's never embedded.
  `emails-NNNNN.quarantine.csv` lists where each hit is (never its text) so it can be fixed in Stage 1.
- **Resumable per shard:** a shard whose output exists is skipped. `checkpoint.json` records the model
  digest, task and chunk settings, and a run with different ones refuses to mix them in (`--force`
  rebuilds). `--limit N` writes a smoke test to `<outdir>/smoke`.

### Stage 2 workflow
1. `ollama pull nomic-embed-text` (once), with Ollama running
2. `python src/embed.py` — clustering vectors (~8 min on an M4)
3. `python src/embed.py --task search_document` — RAG vectors (same chunks, same `chunk_id`s)


## Stage 3: UMAP + HDBSCAN Clusters
Reads the clustering vectors (`data/embeddings/clustering/`) and groups the inbound (client) chunks into
question clusters, the FAQ candidates. Only inbound chunks are clustered; your replies are looked up through the
thread instead. Writes to `data/clusters/` (gitignored; the empty top-level `clusters/` folder isn't, so it can go).

- **UMAP → 5 dimensions first:** HDBSCAN's density estimates break down in 768 dimensions, so UMAP squeezes the
  vectors into 5 that keep each chunk's neighbours (cosine, 15 neighbours, `min_dist` 0, fixed seed).
- **HDBSCAN:** (scikit-learn's) finds the dense groups. Chunks that fit no group are noise (`-1`), which are one-off
  questions and are never forced into a cluster. Defaults: `--min-cluster-size 15 --min-samples 5 --method eom`. **Chose HDBSCAN over other algorithms like K-Means because this data is unstructured and I don't know/have a predefined number (k) of clusters (FAQs)** this is what I am trying to find so hdbscan works perfectly for my use case. 
- **Identical texts are clustered once** and their label is copied to every copy, so sizes still count them.
- **Refuses mixed input:** every row must be the same model's `clustering:` vectors.
- **Outputs:**
  - `assignments.jsonl`: each inbound chunk's cluster and membership strength.
  - `clusters.csv`: one row per cluster, biggest first: sizes, top terms (c-TF-IDF), most typical chunks.
  - `report.txt`: every cluster with its top terms, the 5 chunks closest to its centre, and up to 2 outbound
    replies from the same threads.
  - `clusters.html`: opens offline in a browser. A bar chart of the 20 biggest clusters and a map of every chunk
    (its own 2-D UMAP run), where hovering or clicking highlights a cluster, plus a table of all clusters. It holds
    cluster sizes and top terms only, never message text.
  - `run.json`: the settings and input used.
- **Repeatable:** the same settings give byte-identical output.

Installed for this stage: `umap-learn` 0.5.12 and `scikit-learn` 1.9.1 (they bring numba, llvmlite, scipy,
pynndescent; numpy is unchanged).

### Stage 3 workflow
1. `python src/cluster.py --sweep` — one UMAP run, 16 HDBSCAN settings compared, nothing written (~30 s)
2. `python src/cluster.py` — defaults gave 206 clusters with 35.5% noise (~25 s, including the map's UMAP run); **TODO:** add flags to arbitrarly choose a chunk in cluster for future testing to determine prod model
3. `data/clusters/clusters.html` shows the overview of the clusters and `data/clusters/report.txt` contains what each cluster says
