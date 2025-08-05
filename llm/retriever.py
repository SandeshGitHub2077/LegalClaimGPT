import json
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io_helpers import save_json
from utils.io_helpers import save_json

load_dotenv()

INPUT_PATH = "data/processed/cases.json"
INDEX_PATH = "data/embeddings/faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast + compact
EMBED_FIELD = "full_text"  # Can switch to "summary" later

def load_cases():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(cases, embed_model):
    texts = []
    metadata = []

    for case in tqdm(cases, desc="📚 Encoding cases"):
        text = case.get(EMBED_FIELD)
        if not text:
            continue
        texts.append(text)
        metadata.append({
            "case_id": case["case_id"],
            "case_name": case["case_name"],
            "source_url": case.get("source_url", "")
        })

    embeddings = embed_model.encode(texts, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, metadata

def save_index(index, metadata):
    os.makedirs(INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_PATH, "index.faiss"))
    with open(os.path.join(INDEX_PATH, "index.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def main():
    print("🔍 Loading cases...")
    cases = load_cases()
    model = SentenceTransformer(MODEL_NAME)

    print("🔧 Building FAISS index...")
    index, metadata = build_index(cases, model)

    print("💾 Saving index...")
    save_index(index, metadata)
    print(f"✅ Saved FAISS index to {INDEX_PATH}")

if __name__ == "__main__":
    main()
