## Stage 4: 
    - Test dataset to decide on a production model: ~10% of threads are held out by a hash of `thrid` and never embedded (Test Holdout in the README)
        - Once the FAQs exist: `holdout queries` --> hand-label `faq` for the top ~300 rows of `data/holdout/labels.csv`, the rest is the reserve
    - Read `candidates.txt` to name the rest of the FAQs, `read_by_hand.txt` to review possible FAQs

## Stage 5:
    - Group questions by answer
        - For each FAQ cluster write the answer for each question 
            - answer | escalate --> answer is known or escalate to a human rep
        - Group similar answers into one entry 
    - Phrasings
        - Add phrasings to each group 
            - Ex. Main Topic: What's in the show?
                - phrasings: [text: is a drummer included? from: email_<number>, text: "how many performers for instruments?" from: email_<number>]
    - Test Production Model
        - Test queries: the labeled rows of `data/holdout/labels.csv`, openers and follow-ups scored separately
        - 3 Production models testing for embeddings:
            1. nomic-embed-text (768) — control, already installed + wired
            2. embeddinggemma:300m (768) — multilingual, same dim as nomic
                - prefixes: qwen `Instruct: <task>\nQuery:` (queries only)
            3. qwen3-embedding:0.6b (1024) — highest multilingual MTEB (64.33), 32K ctx
                -  prefixes: gemma `task: search result | query: ` / `title: none | text: `; 
        - choose 1 of the 3 based on test results 
    
## Stage 6:
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
                
