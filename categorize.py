import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    # "http://localhost:11434"
) + "/api/generate" 
MODEL      = "llama3.1:8b"                                
INPUT_CSV  = "./combined_df.csv"
OUTPUT_CSV = "./knowledge_base.csv"
INPUT_COL  = "query"  

CATEGORIES = [
    "Family Issues",
    "Relationship Issues",
    "Work & Career Stress",
    "Academic Pressure",
    "Grief & Loss",
    "Anxiety & Panic",
    "Depression & Mood",
    "Self-Esteem & Identity",
    "Trauma & Abuse",
    "Loneliness & Social Isolation",
    "Financial Stress",
    "Health & Medical Concerns",
    "Addiction & Substance Use",
    "Anger & Conflict",
    "General Mental Health",
]

SYSTEM_PROMPT = f"""You are a mental health data analyst.
Your job is to classify a user's query into exactly ONE category from the list below.
Reply with ONLY the category name — no explanation, no punctuation, nothing else.

Categories:
{chr(10).join(f"- {c}" for c in CATEGORIES)}

If the query does not clearly fit any category, reply with: General Mental Health
"""

# ── Ollama call ───────────────────────────────────────────────────────────────
def get_category(query: str, retries: int = 3) -> str:
    payload = {
        "model": MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser query: {query}\n\nCategory:",
        "stream": False,
        "options": {"temperature": 0},
    }
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
   
            for cat in CATEGORIES:
                if cat.lower() in raw.lower():
                    return cat
          
            return raw if len(raw) < 60 else "General Mental Health"
        except requests.exceptions.RequestException as e:
            print(f"\n[Attempt {attempt+1}] Error: {e}")
            time.sleep(2)
    return "General Mental Health"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Number of Rows: {len(df)}")

    df = df.head(20)
    print(f"Found {len(df)} rows. Starting categorization with model '{MODEL}' ...\n")

    if INPUT_COL not in df.columns:
        raise ValueError(f"Column '{INPUT_COL}' not found. Available: {list(df.columns)}")

    print(f"Found {len(df)} rows. Starting categorization with model '{MODEL}' ...\n")

    categories = []
    for query in tqdm(df[INPUT_COL], desc="Categorizing"):
        category = get_category(str(query))
        categories.append(category)

    df["category"] = categories
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved to {OUTPUT_CSV}")

    # Quick summary
    print("\nCategory distribution:")
    print(df["category"].value_counts().to_string())

if __name__ == "__main__":
    main()