import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import re

def extract_valid_json(text):
    text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"{[\s\S]*?}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None
    return None

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY
)

INPUT_PATH = "../data/raw/cases.json"
OUTPUT_PATH = "../data/processed/cases.json"

PROMPT_TEMPLATE = """
You are a legal-medical assistant. Given the following case text, extract and generate the following details as JSON:
1. Top 1–2 injury types
2. Approximate total medical bills (USD)
3. Approximate lost wages (USD)
4. A reasonable settlement amount (USD)
5. Plaintiff's age (between 20 and 70)
6. Plaintiff's gender (Male/Female)

Respond only with the JSON object.

CASE TEXT:
\"\"\"{case_text}\"\"\" 
"""

RELEVANT_KEYWORDS = [
    "injury", "accident", "negligence", "pain", "suffering",
    "medical malpractice", "damages", "fracture", "hospital", "treatment",
    "settlement", "claimant", "plaintiff", "liability"
]
IRRELEVANT_KEYWORDS = [
    "criminal", "sexual", "tenant", "appeal denied", "parole",
    "fraud", "custody", "disciplinary", "wage", "discrimination"
]

def is_relevant_text(text):
    text = text.lower()
    pos = sum(kw in text for kw in RELEVANT_KEYWORDS)
    neg = any(kw in text for kw in IRRELEVANT_KEYWORDS)
    return pos >= 2 and not neg

def label_case(case_text):
    prompt = PROMPT_TEMPLATE.format(case_text=case_text[:3000])

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()

        if "```" in reply:
            lines = reply.splitlines()
            inside_code = False
            cleaned = []
            for line in lines:
                if line.strip().startswith("```"):
                    inside_code = not inside_code
                    continue
                if inside_code:
                    cleaned.append(line)
            reply = "\n".join(cleaned).strip()

        parsed = extract_valid_json(reply)

        if "injury_types" in parsed and "injuries" not in parsed:
            parsed["injuries"] = parsed.pop("injury_types")

        return parsed

    except Exception as e:
        print("⚠️ LLM error or JSON parse failed:", str(e))
        print("🔁 Raw reply:", reply)
        return {}

def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cases(cases, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

def enrich_cases():
    raw_cases = load_cases(INPUT_PATH)
    enriched = []

    print(f"✨ Labeling {len(raw_cases)} cases using LLM...")
    for case in tqdm(raw_cases):
        text = case.get("full_text", "").strip()

        if len(text) < 100:
            print(f"⚠️ Empty or very short text for case: {case.get('case_id') or 'Unknown'} (len={len(text)})")
            continue

        if not is_relevant_text(text):
            fallback_name = case.get("case_name") or case.get("case_id") or text[:50]
            print(f"⏩ Skipping irrelevant case: {fallback_name}...")
            continue

        labels = label_case(text)
        enriched_case = {**case, **labels}
        enriched.append(enriched_case)
        time.sleep(1)

    save_cases(enriched, OUTPUT_PATH)
    print(f"✅ Saved enriched cases to {OUTPUT_PATH}")

if __name__ == "__main__":
    enrich_cases()
