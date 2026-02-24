import pandas as pd
import requests
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL"
    # , "http://localhost:11434" 
    ) + "/api/generate"
MODEL      = "llama3.2:3b"
INPUT_CSV  = "./combined_df.csv"
OUTPUT_CSV = "./knowledge_base.csv"
INPUT_COL  = "query"
MAX_WORKERS = 5 

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
def get_category(index: int, query: str, retries: int = 3) -> tuple[int, str]:
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
                    return index, cat
            return index, raw if len(raw) < 60 else "General Mental Health"
        except requests.exceptions.RequestException as e:
            print(f"\n[Row {index} | Attempt {attempt+1}] Error: {e}")
            time.sleep(2)
    return index, "General Mental Health"

def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows in dataset: {len(df)}")

    if INPUT_COL not in df.columns:
        raise ValueError(f"Column '{INPUT_COL}' not found. Available: {list(df.columns)}")

    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        df["category"] = pd.NA
        df.loc[done_df.index, "category"] = done_df["category"].values
        already_done = set(df[df["category"].notna()].index.tolist())
        nan_count = len(done_df) - len(already_done)
        print(f"Resuming — {len(already_done)} valid rows, {nan_count} NaN rows will be retried, {len(df) - len(already_done)} total remaining.\n")
    else:
        already_done = set()
        df["category"] = pd.NA
        print(f"Fresh start — processing {len(df)} rows.\n")

    pending = [
        (i, str(row[INPUT_COL]))
        for i, row in df.iterrows()
        if i not in already_done
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_category, i, q): i for i, q in pending}
        with tqdm(total=len(pending), desc="Categorizing") as pbar:
            for future in as_completed(futures):
                idx, category = future.result()
                df.at[idx, "category"] = category
                df.to_csv(OUTPUT_CSV, index=False)
                pbar.update(1)

    print(f"\nDone! Saved to {OUTPUT_CSV}")
    print("\nCategory distribution:")
    print(df["category"].value_counts().to_string())

if __name__ == "__main__":
    main()