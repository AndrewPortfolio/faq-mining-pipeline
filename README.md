# Faq-mining-pipeline [![.github/workflows/ci.yml](https://github.com/AndrewPortfolio/faq-mining-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewPortfolio/faq-mining-pipeline/actions/workflows/ci.yml)  
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
- **Loose phone/URL/email shapes** (`loose_phone`, `loose_url`, `loose_email`): Presidio's own recognizers reject a
  number glued to a letter or wrapped onto the next line, and an address with no real TLD. These match the Stage 2
  tripwire's shapes instead, so whatever the tripwire would quarantine gets redacted here first. They score under
  Presidio's, so a span both find stays credited to Presidio.
- **Invisible characters are stripped first** (`textclean.strip_invisible`): a zero-width character inside a word
  hides it from every recognizer. Every script that lines review offsets up with shard text reads through
  `analyze_pii.read_rows`, so the offsets agree.
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
1. `PYTHONHASHSEED=0 python src/analyze_pii.py` — writes the review files (~1.1 h for all shards). The fixed seed
   keeps tied spans in the same order between runs, so two runs' review files can be diffed.
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
- **ATLD's own unsent mail is skipped:** Stage 0 calls anything without Gmail's `Sent` label inbound, so drafts
  (canned replies included) and old synced notes passed for client messages. An "inbound" email from whoever sent
  the Sent mail is skipped (101 emails).
- **Quoted mail Stage 0 missed is cut:** HTML→text sometimes leaves a quote header mid-line
  (`…thanks! On <DATE_TIME> at 07:23 PM, <PERSON> <<EMAIL_ADDRESS>> wrote: Hello…`), where Stage 0's
  line-anchored patterns can't see it. Everything from the header on is dropped (229 emails), often ATLD's own
  reply quoted back. The tripwire still reads the whole email first.
- **Resumable per shard:** a shard whose output exists is skipped. `checkpoint.json` records the model
  digest, task, chunk settings and cleaning version, and a run with different ones refuses to mix them in (`--force`
  rebuilds). `--limit N` writes a smoke test to `<outdir>/smoke`.

### Stage 2 workflow
1. `ollama pull nomic-embed-text` (once), with Ollama running
2. `python src/embed.py` — clustering vectors (~8 min on an M4)
3. `python src/embed.py --task search_document` — RAG vectors (same chunks, same `chunk_id`s)


## Stage 3: UMAP + HDBSCAN Clusters
Reads the clustering vectors (`data/embeddings/clustering/`) and groups the inbound (client) chunks into
question clusters, the FAQ candidates. Only inbound chunks are clustered; your replies are looked up through the
thread instead. Writes to `data/clusters/` (gitignored).

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
2. `python src/cluster.py` — defaults gave 204 clusters with 32.1% noise (~25 s, including the map's UMAP run); **TODO:** add flags to arbitrarly choose a chunk in cluster for future testing to determine prod model
3. `data/clusters/clusters.html` shows the overview of the clusters and `data/clusters/report.txt` contains what each cluster says


## Stage 4: Noise Triage
Reads Stage 3's output (`data/clusters/`) and the chunk texts, and sorts every cluster into an FAQ candidate, one to
review, or noise to skip. Writes to `data/faqs/`. Nothing is deleted: a skipped cluster is still listed with its reason.

- **Rules read a cluster's text, never its number:** every Stage 3 run renumbers the clusters biggest-first.
- **Skipped** (each threshold sits in a wide gap measured on the first run):
  - `non_english`: half its chunks are Vietnamese. nomic-embed-text is mostly English, so it groups them by language.
  - `template`: half its chunks hold the website form or the filled-in confirmation template (`FORM_MARKERS`).
  - `footer`: 30% of its chunks hold a confidentiality-notice phrase.
- **Review**, never skipped: clusters whose top terms are HTML (`markup`, which can hide a real topic), and clusters
  where ≤10% of chunks ask a question (`few_questions`, mostly thanks and confirmations). Question share has no clean
  cutoff, and some inquiries are written as statements.
- **Refuses stale input:** if `data/clusters/` wasn't built from the current embeddings, it stops and asks for a rerun
  of `cluster.py`.
- **Outputs:**
  - `triage.csv`: each cluster's decision, reason, and the shares behind it.
  - `candidates.txt`: FAQ candidates, then the review groups, in `report.txt`'s format; skipped clusters one line each.
  - `read_by_hand.txt`: every Vietnamese chunk wherever it landed, by thread, with the reply it got.
  - `run.json`: the Stage 3 run it triaged and the thresholds.

### Stage 4 workflow
1. `python src/triage.py` after every `cluster.py` run (~4 s). The current run gives 153 FAQ candidates, 32 to review,
   19 skipped. Known gap: a 255-chunk "sorry / thanks / yes" cluster passes as a candidate, because 23% of its chunks
   ask a question.
2. Read `candidates.txt` to name FAQs, and `read_by_hand.txt` for the Vietnamese ones.

## Refactor 
### Storage Layer `src/faq_mining/shards.py`
Moves the shard read/write helpers and `Checkpoint` out of `shared/pipeline.py` into `faq_mining/storage/shards.py`, as part of the layered refactor. `Deduper` stays in `shared/pipeline.py` for now.

## Changes
- New `faq_mining/storage/shards.py`: `shard_paths`, `iter_shard_rows`, `count_rows`, `read_shard`, `write_atomic`, `ShardWriter`, `Checkpoint`.
- `ShardWriter.flush` and `Checkpoint.save` now call `write_atomic`; before, each repeated its own temp-file-and-rename code.
- Repointed every caller (9 scripts and `tests/test_extract_mbox.py`) to the new module. `apply_redactions`, `pii_sample` and `seed_place_denylist` now import `write_atomic` from storage, not through `analyze_pii`.
- `shared/pipeline.py` re-exports the moved names so open branches keep working. The refactor's Cleanup PR removes them.

## Verification
- 403 tests pass; ruff is clean.
- A smoke run on 500 real messages (extract, PII analysis, redaction and the denylist/sample helpers) produces output identical to main's.
- `embed`, `cluster` and `triage` changed only their imports, and their tests import them.

## How `shards.py` works
- **Reading:** `shard_paths` lists a folder's `emails-NNNNN.jsonl` files in shard order and ignores other files. The 5-digit padding makes name order match number order. `iter_shard_rows` streams rows one at a time across every shard. `read_shard` loads one whole shard, for stages that write one output per input shard, like `embed`. `count_rows` gives the total for `analyze_pii`'s progress display.
- **Writing:** almost every output file goes through `write_atomic`, including shards, checkpoints, CSVs and reports. It writes to a `.part` file and then renames it, so a file is either complete or missing, never half-written. `ShardWriter` buffers rows and writes a shard each time `shard_size` rows have built up, plus one last partial shard at the end. `add()` returns `True` when a shard closes.
- **Resuming:** `Checkpoint` saves a dict as JSON, loads it, or deletes it, and each stage chooses what to store. `extract_mbox` saves its mbox byte position and next shard number whenever a shard closes, so rows still in the buffer when a run dies are re-read. `analyze_pii` and `embed` skip any shard whose output file already exists, which is safe because of `write_atomic`. They use the checkpoint to carry their stats across restarts.

For deeper details about the PR and how everything works: https://github.com/AndrewPortfolio/faq-mining-pipeline/pull/16#issue-5875451374 
