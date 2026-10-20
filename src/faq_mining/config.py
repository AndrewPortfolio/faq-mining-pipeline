# Every data path the stages read and write, relative to the repo root, so commands run from there
# Plain strings rather than pathlib.Path: argparse defaults stay the exact strings they always were

DATA_DIR = "data"

MBOX = f"{DATA_DIR}/raw/All mail Including Spam and Trash.mbox"

# One folder per stage
EXTRACTED_DIR = f"{DATA_DIR}/extracted"
REVIEW_DIR = f"{DATA_DIR}/review"
REDACTED_DIR = f"{DATA_DIR}/redacted"
EMBEDDINGS_DIR = f"{DATA_DIR}/embeddings"
CLUSTERS_DIR = f"{DATA_DIR}/clusters"
FAQS_DIR = f"{DATA_DIR}/faqs"
HOLDOUT_DIR = f"{DATA_DIR}/holdout"
# Stage 2's clustering task writes here, and Stages 3 and 4 read it
CLUSTERING_EMBEDDINGS_DIR = f"{EMBEDDINGS_DIR}/clustering"

# Hand-kept lists and the hash key --> real client names, gitignored
PII_DIR = f"{DATA_DIR}/pii"
NAME_DENYLIST = f"{PII_DIR}/name_denylist.txt"
VENUE_DENYLIST = f"{PII_DIR}/venue_denylist.txt"
PLACE_DENYLIST = f"{PII_DIR}/place_denylist.txt"
ALLOWLIST = f"{PII_DIR}/allowlist.txt"
COMMON_DATETIME = f"{PII_DIR}/common_words_datetime.txt"
COMMON_LOCATION = f"{PII_DIR}/common_words_location.txt"
HASH_KEY = f"{PII_DIR}/hash_key"

# The system dictionary the seed tools read
ENGLISH_WORDS = "/usr/share/dict/words"
