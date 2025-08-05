import os
import json
import pandas as pd

def load_json(path):
    """Load a JSON file (list of dicts)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    """Save a list of dicts to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved JSON to {path}")

def load_csv(path):
    """Load a CSV file into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    return pd.read_csv(path)

def save_csv(df, path):
    """Save a DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved CSV to {path}")
