## Stage 0:
    - Data extraction of 25 GB mbox file into 15 .jsonl shards

## Stage 1: 
    - PII redaction

## Stage 2: 
    - Chunking + embedding
    - Test holdout (README Stage 2b): embed.py skips ~10% of threads, picked by a hash of `thrid`, so they're never embedded, clustered or triaged
        - After every re-embed / recluster / retriage: `holdout check` should find 0 held-out ids

## Stage 3:
    - UMAP + HDBSCAN for clustering

## Stage 4: 
    - Triage noisy clusters
    - Read `candidates.txt` to name the rest of the FAQs, `read_by_hand.txt` to review possible FAQs



## Deprecated
    - Pivot: after finding 6 FAQs --> no need for RAG --> store FAQs in table in DB easy lookup 

### Stage 5 (DEPRECATED):
    - Group questions by answer
        - For each FAQ cluster write the answer for each question 
            - answer | escalate --> answer is known or escalate to a human rep
        - Group similar answers into one entry 
    - Phrasings
        - Add phrasings to each group 
            - Ex. Main Topic: What's in the show?
                - phrasings: [text: is a drummer included? from: email_<number>, text: "how many performers for instruments?" from: email_<number>]
    - Test Production Model
        - Once the FAQ entries are final: `holdout queries` --> hand-label `faq` for the top ~300 rows of `data/holdout/labels.csv`, the rest is the reserve
            - Not earlier: seeing the test questions while writing FAQs inflates the score
        - Test queries: the labeled rows, openers and follow-ups scored separately
        - 3 Production models testing for embeddings:
            1. nomic-embed-text (768) — control, already installed + wired
            2. embeddinggemma:300m (768) — multilingual, same dim as nomic
                - prefixes: qwen `Instruct: <task>\nQuery:` (queries only)
            3. qwen3-embedding:0.6b (1024) — highest multilingual MTEB (64.33), 32K ctx
                -  prefixes: gemma `task: search result | query: ` / `title: none | text: `; 
        - choose 1 of the 3 based on test results 
    
## Stage 6 (DEPRECATED):
    - Production (high-level overview)
        - Incoming email
            - Clean for query 
        - Embed query
        - Retrieval 
            - Algorithms:
                - Approximate Nearest Neighbor (ANN)
                - possibly HNSW or IVF
                - Product Quantiazation (PQ) for compressing the vector if needed
            - Returns top k vectors
        - Reply
            - Evauluated and then logged asynchronously to track accuracy, cost, and silent degradation over time 
            - Depending on evaluation: send answer | escalate to human representative

                