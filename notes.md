# TODO 
- [ ] No batching, I have a 25 GB mbox
- [ ] Missing the nomic task prefix. nomic-embed-text is trained with task prefixes and this project expects: clustering
- [ ] No timeout= on requests.post. Default is infinite --> Ollama call runs forever
- [ ] Needs a main() behind if __name__ == "__main__": FOR embed.py 
- [ ] No L2 normalization HDBSCAN defaults to Euclidean --> fix: normalize unit length
- [ ] Nothing guarantees vectors and email_ids stay aligned. Fix make them one artifact 
- [ ] Output paths ignore layout 

# 09/26/26
- [ ] Run analyze_pii.py
- [ ] Review csv files in /data/reviews
- [ ] Run apply_redactions.py 