# RAG Chunking/Embedding Pipeline

## Project Context
This file governs an
offline Python tool: parsing the Google Takeout export of ATLD's inbox,
chunking it, and generating embeddings for the RAG email/Instagram
responder.

## How I Want Help (I'm a junior dev leveling up)
- Explain the *why* for nontrivial choices, not the *what* — No more than 1 sentence (keep it simple)
    - If something needs more than 1 sentence than 2 is the hard max 
- Name the alternative and the trade-off when one exists, don't just pick
  silently.
- Don't re-explain things already established here
- Concise inline explanations; flag if something needs more room instead of
  writing a lecture unasked.

## Editing Style
- Minimal, targeted diffs. Multiple changes → separate, clearly located
  edits, not a full-file rewrite.
- This is offline/local tooling, not user-facing — fine to favor simplicity
  over robustness 

## Review Gates (do not skip)
- Never commit or push without being asked to.
- Never add a new dependency (tokenizer libs, HTTP clients, etc.) without
  flagging it first — check what's already used before reaching for pip
  install.
- Don't guess the shard schema or the installed embedding model's real
  context/token limit — confirm both empirically before writing logic
  against them.
- If inline PII (names/emails/phones/addresses in body text, not just
  headers) turns up anywhere in the pipeline, stop and flag it — don't let
  it reach an embedding call. Once something's embedded it's effectively
  permanently retrievable.
- No unbounded retry loops on any network call (Ollama or otherwise) — cap
  and log on failure.

## Stack Reminders
- Python, stdlib-first — the stripper already avoids new dependencies where
  it can (e.g. HTML→text fallback via stdlib, not a library); match that
  instinct here.
- Embeddings via local Ollama, model `nomic-embed-text` — confirm the
  installed tag and its real context window with `ollama show`; don't
  assume a number from memory or documentation alone.
- Output is local JSONL shards only, one per input shard, same
  offset-checkpoint + atomic-write convention as the stripper. 
- Batch embedding calls through `/api/embed` (array `input`), not the
  legacy singular `/api/embeddings`.

## Trade-off Awareness Specific to This Pipeline
- Chunk size is deliberately conservative (~300 words) rather than sized to
  the model's actual ceiling — optimizing for retrieval precision over
  fewer, larger chunks. A bigger chunk fits more content per embedding
  call, but blends multiple topics into one vector and hurts retrieval
  quality; don't "fix" this by growing the chunk size back up without cause.
- No tokenizer dependency for exact token counting — a conservative word
  target plus a catch-and-resplit retry on overflow instead. Revisit only
  if the retry path starts firing often enough to matter.
- PII gets stripped in this pipeline, before embedding — not deferred to a
  later review pass on the FAQ/RAG content, because by then it's already
  baked into a stored vector.