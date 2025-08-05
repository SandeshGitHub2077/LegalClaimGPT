import json
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io_helpers import save_json

load_dotenv()

# Paths
INPUT_PATH = "data/processed/cases.json"
OUTPUT_PATH = "data/processed/summaries.json"
INDEX_PATH = "data/embeddings/faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding + LLM clients
embedder = SentenceTransformer(MODEL_NAME)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

SUMMARY_PROMPT = """
You are a legal assistant. Summarize the following legal case in 3–5 sentences using simple language.
Highlight the key legal issue, type of injury, and outcome if available.

CASE:
\"\"\"{text}\"\"\"
"""

def load_faiss_index():
    index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
    with open(os.path.join(INDEX_PATH, "index.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def summarize_case(text):
    prompt = SUMMARY_PROMPT.format(text=text[:3000])

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("⚠️ Summarization failed:", e)
        return ""

def main():
    print("📂 Loading cases and FAISS index...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    index, _ = load_faiss_index()

    summaries = []
    print(f"🧠 Summarizing {len(cases)} cases using RAG + LLaMA-3...")
    for case in tqdm(cases):
        text = case.get("full_text", "")
        if not text:
            continue

        summary = summarize_case(text)
        summaries.append({
            **case,
            "summary": summary
        })

    save_json(summaries, OUTPUT_PATH)
    print(f"✅ Saved {len(summaries)} summaries to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
