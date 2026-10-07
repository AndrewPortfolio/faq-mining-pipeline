# TODO 
- [ ] No batching, I have a 25 GB mbox
- [ ] Missing the nomic task prefix. nomic-embed-text is trained with task prefixes and this project expects: clustering
- [ ] No timeout= on requests.post. Default is infinite --> Ollama call runs forever
- [ ] Needs a main() behind if __name__ == "__main__": FOR embed.py 
- [ ] No L2 normalization HDBSCAN defaults to Euclidean --> fix: normalize unit length
- [ ] Nothing guarantees vectors and email_ids stay aligned. Fix make them one artifact 
- [ ] Output paths ignore layout 

# 09/26/26
- [x] Hardened analyze PII to find EIN and policy numbers 
    - These slipped through the first pass after data review 
    - Ambiguous names like "The", "To", "An", etc. checks have been hardend to check for context --> no longer redacts false positives

# 9/28/26
- [x] Remove Locations/Venues from data
    - was brainstorming ideas on how to remove locations/venues 
        - Remove all vs remove at freq k (arbritrary number) --> Remove all locations: Venues, places, zip 

# 9/29/26
- [x] Implement Signature Block Removal 
    - Reasoning: Lot's of quasi-identifier information found in these blocks 
- [x] Review the rest of sample.txt

# 9/30/26
- [x] Run analyze_pii.py
- [x] Review csv files in /data/reviews
- [x] Run apply_redactions.py 

# 10/6/26 
- [x] Vector Embeddings 