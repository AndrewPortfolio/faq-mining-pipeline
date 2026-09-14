import json
import numpy as np
import requests


def embed_texts(texts, model="nomic-embed-text"):
    url = "http://localhost:11434/api/embed"

    # 2. Pass the list of texts all at once into 'input'
    payload = {"model": model, "input": texts}

    response = requests.post(url, json=payload)
    response.raise_for_status()  # Safe practice: crash early if Ollama isn't running

    # 3. Extract the plural 'embeddings' array
    return response.json()["embeddings"]


# --- Workflow Pipeline ---
# After redaction is done and approved:
vectors = embed_texts(redacted_texts)

# Convert to a clean NumPy array and save efficiently
np.save("embeddings.npy", np.array(vectors))

# Save your mapping tracking IDs
with open("email_ids.json", "w") as f:
    json.dump(email_ids, f)
