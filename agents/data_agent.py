import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io_helpers import save_json

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("COURTLISTENER_API_KEY")
assert API_KEY, "COURTLISTENER_API_KEY not set in .env"

# Constants
BASE_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
HEADERS = {"Authorization": f"Token {API_KEY}"}
SAVE_PATH = "../data/raw/cases.json"

INCLUDE_KEYWORDS = [
    "personal injury", "negligence", "medical malpractice", "wrongful death",
    "slip and fall", "trip and fall", "car accident", "motor vehicle",
    "tort", "fracture", "spinal cord", "brain injury", "premises liability",
    "burn injury", "back injury", "bodily harm"
]
EXCLUDE_KEYWORDS = [
    "criminal", "murder", "homicide", "sexual assault", "real estate",
    "landlord", "tenant", "eviction", "contract", "loan", "bankruptcy",
    "labor", "overtime", "minimum wage", "retaliation", "discrimination"
]

def is_likely_personal_injury(text):
    if not text:
        return False
    text = text.lower()
    pi_keywords = [
        "personal injury", "negligence", "medical malpractice", "slip and fall",
        "wrongful death", "pain and suffering", "bodily harm", "trauma",
        "injury", "accident", "damages"
    ]
    exclusion_keywords = [
        "sexual assault", "criminal", "child custody", "termination of parental rights",
        "unpaid wages", "overtime", "foreclosure", "arbitration",
        "disciplinary", "custody", "appeal denied", "drug trafficking", "head shop"
    ]
    return sum(kw in text for kw in pi_keywords) >= 2 and not any(kw in text for kw in exclusion_keywords)

def is_likely_case_name(name):
    if not name:
        return False
    name = name.lower()
    return any(good in name for good in INCLUDE_KEYWORDS) and not any(bad in name for bad in EXCLUDE_KEYWORDS)

def fetch_opinions(limit=30, court="ca9"):
    print(f"🔍 Fetching {court.upper()} cases (limit: {limit})...")
    page_size = min(limit, 100)
    params = {"court": court, "page_size": page_size}
    results = []
    next_url = BASE_URL

    while next_url and len(results) < limit:
        try:
            response = requests.get(next_url, headers=HEADERS, params=params)
            if response.status_code != 200:
                print(f"❌ Error fetching data: {response.status_code}")
                break

            data = response.json()
            for result in tqdm(data["results"], desc="🔎 Filtering PI cases"):
                if len(results) >= limit:
                    break
                case_text = result.get("plain_text", "") or ""
                case_name = result.get("caseName", "") or ""

                if not case_text.strip():
                    print(f"⚠️ Empty plain_text for case ID: {result.get('id')} — skipping.")
                    continue

                if is_likely_personal_injury(case_text) or is_likely_case_name(case_name):
                    results.append({
                        "case_id": result.get("id"),
                        "case_name": case_name,
                        "jurisdiction": result.get("court", ""),
                        "date_filed": result.get("date_filed"),
                        "source_url": result.get("absolute_url", ""),
                        "full_text": case_text.strip()
                    })

            next_url = data.get("next")
            params = {}  # Clear after first request

        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            break

    return results

if __name__ == "__main__":
    cases = fetch_opinions(limit=30, court="ca9")
    save_json(cases, SAVE_PATH)
    print(f"✅ Saved {len(cases)} cases to ../{SAVE_PATH}")
