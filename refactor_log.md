### Refactor Tracker
This file is for me to keep track of what i need to do. 
Done on a separate worktree named `faq-mining-pipeline-refactor`


## Goal:
1. Have full understanding and mastery over the codebase
2. Optimize/find bugs in the codebase

## Rules: 
1. **Fix as I find bugs/optimizations; pytest is the control**
2. **Copy data constants, rewrite the logic** 
3. **Keep public names** 
4. **Type-hint for every function** 
5. **Leave a shim; Cleanup deletes it** When a chunk moves a name that other modules import, the old module imports it back from its new home instead of losing it. That keeps open feature branches and usual commands working.


### Chunk 0 Setup - 402 pytest passed
## Chunk 0a: 
- setting up pyproject.toml file, and initializing data for smoke runs on this worktree
- **Bug**: during the smoke diff before and after shards: `emails-00000.spans.csv` and `emails-00000.spans.csv` differed by about 456 spans
    the fault was in `src/shared/pii_recognizer` function `build_analyzer`. The objects are hashed by memory address so it ran in a diff order every process.
    **What changed**: credit for decision was made --> should've been denylist_names that got credit for catching names, but went to SpacyRecognizer instead
    No redaction decisions were effected, just what mechanism got named in the csv for making the decision. 
    **Fix**: OrderedRegistry class sorts the recognizers Presidio returns back in order they were registred --> Presidio first, so spaCy wins every tie

