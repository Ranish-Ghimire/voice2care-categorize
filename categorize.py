import pandas as pd
import requests
import time
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

OLLAMA_URL  = os.getenv(
    "OLLAMA_BASE_URL",
    # "http://localhost:11434"
    ) + "/api/generate"
MODEL       = "llama3.2:3b"
INPUT_CSV   = "./combined_df.csv"
OUTPUT_CSV  = "./knowledge_base.csv"
INPUT_COL   = "query"
EMOTION_COL = "emotion"
MAX_WORKERS = 5
SAVE_EVERY  = 10

CATEGORY_NAMES = [
    "Family Issues",
    "Relationship Issues",
    "Marriage & Couples",
    "Sexual Identity & Orientation",
    "Gender Identity",
    "Self-Esteem & Identity",
    "Body Image & Eating Disorders",
    "Anxiety & Panic",
    "Depression & Mood",
    "Grief & Loss",
    "Trauma & PTSD",
    "OCD & Intrusive Thoughts",
    "Psychosis & Severe Mental Illness",
    "Work & Career Stress",
    "Academic Pressure",
    "Financial Stress",
    "Life Transitions & Existential Crisis",
    "Loneliness & Social Isolation",
    "Addiction & Substance Use",
    "Self-Harm & Suicidal Ideation",
    "Anger & Conflict",
    "Health & Medical Concerns",
    "Sleep & Fatigue",
    "Social Skills & Relationships",
    "Positive Growth & Resilience",
    "General Mental Health",
]

EMOTION_GUIDANCE = {
    "happy":   "The user feels positive. Look for recovery, gratitude, achievements, or breakthroughs.",
    "sad":     "The user feels sad or despairing. Look for grief, loneliness, depression, relationship pain, family conflict, or self-worth issues.",
    "angry":   "The user feels angry or frustrated. Look for conflict, injustice, relationship issues, work stress, family problems, or abuse.",
    "fear":    "The user feels fearful or anxious. Look for panic, health worries, financial stress, academic pressure, trauma, or identity fears.",
    "disgust": "The user feels disgust or aversion. Look for self-loathing, toxic relationships, addiction, abuse, or shame.",
    "neutral": "The user has a neutral tone. Focus entirely on the topic and content of the query.",
    "surprise":"The user feels shocked. Look for unexpected life events, sudden relationship changes, identity revelations, or trauma triggers.",
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert mental health data analyst.

Your task: Classify the user's query into ONE or TWO categories from the list below.
- Assign TWO categories only when the query clearly has two distinct themes of equal importance.
- Assign ONE category when one theme dominates, even if another is lightly mentioned.
- ALWAYS put the most dominant/central theme as category_1.
- category_2 should only be used when a second theme is strongly present, not just hinted at.

EMOTION CONTEXT: The detected emotion is "{emotion}".
Hint: {emotion_hint}

CATEGORIES:
{categories}

RESPONSE FORMAT — reply with valid JSON only, no explanation, no markdown:
{{"category_1": "<primary category>", "category_2": "<secondary category or null>"}}

CLASSIFICATION RULES:
1. category_1 is always required. category_2 is null unless a strong second theme exists.
2. Both values must be exact category names from the list above, or null for category_2.
3. Never assign the same category to both fields.
4. Prioritize safety — if suicidal ideation or self-harm is present, it must be category_1.
5. Pay special attention to:
   - Sexuality/orientation confusion → "Sexual Identity & Orientation"
   - Gender identity/dysphoria → "Gender Identity"
   - Suicidal thoughts or self-harm → "Self-Harm & Suicidal Ideation" (always category_1)
   - Death of a loved one → "Grief & Loss"
   - Past abuse/childhood trauma/PTSD symptoms → "Trauma & PTSD"
   - Romantic partner problems → "Relationship Issues" or "Marriage & Couples"

Examples:
Query: "I lost my wife two weeks ago and I feel helpless and alone. The hospice staff used to visit but now I only get phone calls and miss the human connection."
Emotion: sad
→ {{"category_1": "Grief & Loss", "category_2": "Loneliness & Social Isolation"}}

Query: "I've been having flashbacks of childhood abuse and I can't trust anyone anymore."
Emotion: fear
→ {{"category_1": "Trauma & PTSD", "category_2": "Self-Esteem & Identity"}}

Query: "I am confused about my sexuality and fear judgment from my family."
Emotion: fear
→ {{"category_1": "Sexual Identity & Orientation", "category_2": "Family Issues"}}

Query: "My heart races every time I think about my upcoming exam."
Emotion: fear
→ {{"category_1": "Academic Pressure", "category_2": null}}

Query: "I feel like nobody would miss me if I was gone and I've been hurting myself."
Emotion: sad
→ {{"category_1": "Self-Harm & Suicidal Ideation", "category_2": "Depression & Mood"}}
"""

def build_prompt(query: str, emotion: str) -> str:
    emotion = emotion.lower().strip() if isinstance(emotion, str) else "neutral"
    hint = EMOTION_GUIDANCE.get(emotion, EMOTION_GUIDANCE["neutral"])
    category_list = "\n".join(f"- {c}" for c in CATEGORY_NAMES)
    system = SYSTEM_PROMPT_TEMPLATE.format(
        emotion=emotion,
        emotion_hint=hint,
        categories=category_list,
    )
    return f"{system}\n\nUser query: {query}\n\nJSON:"

def parse_response(raw: str) -> tuple[str, str | None]:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        cat1 = data.get("category_1", "").strip()
        cat2 = data.get("category_2", None)

        if cat1 not in CATEGORY_NAMES:
            for c in CATEGORY_NAMES:
                if c.lower() in cat1.lower():
                    cat1 = c
                    break
            else:
                cat1 = "General Mental Health"

        if cat2 and cat2 != "null":
            if cat2 not in CATEGORY_NAMES:
                for c in CATEGORY_NAMES:
                    if c.lower() in cat2.lower():
                        cat2 = c
                        break
                else:
                    cat2 = None
            if cat2 == cat1: 
                cat2 = None
        else:
            cat2 = None

        return cat1, cat2

    except (json.JSONDecodeError, AttributeError):
        for c in CATEGORY_NAMES:
            if c.lower() in raw.lower():
                return c, None
        return "General Mental Health", None

def get_category(index: int, query: str, emotion: str, retries: int = 3) -> tuple[int, str, str | None]:
    payload = {
        "model": MODEL,
        "prompt": build_prompt(query, emotion),
        "stream": False,
        "options": {"temperature": 0},
    }
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            cat1, cat2 = parse_response(raw)
            return index, cat1, cat2
        except requests.exceptions.RequestException as e:
            print(f"\n[Row {index} | Attempt {attempt + 1}] Error: {e}")
            time.sleep(3)
    return index, "General Mental Health", None

save_lock = threading.Lock()
save_counter = [0]

def save_result(df: pd.DataFrame, idx: int, cat1: str, cat2: str | None):
    with save_lock:
        df.at[idx, "category_1"] = cat1
        df.at[idx, "category_2"] = cat2 
        save_counter[0] += 1
        if save_counter[0] % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False)

def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows in dataset: {len(df)}")

    if INPUT_COL not in df.columns:
        raise ValueError(f"Column '{INPUT_COL}' not found. Available: {list(df.columns)}")

    if EMOTION_COL not in df.columns:
        print(f"Warning: '{EMOTION_COL}' column not found. Using 'neutral' for all rows.")
        df[EMOTION_COL] = "neutral"

    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        df["category_1"] = pd.NA
        df["category_2"] = pd.NA
        df.loc[done_df.index, "category_1"] = done_df["category_1"].values if "category_1" in done_df.columns else pd.NA
        df.loc[done_df.index, "category_2"] = done_df["category_2"].values if "category_2" in done_df.columns else pd.NA
        # A row is "done" only if category_1 has a valid value
        already_done = set(df[df["category_1"].notna()].index.tolist())
        nan_count = len(done_df) - len(already_done)
        print(f"Resuming — {len(already_done)} valid rows done, {nan_count} failed rows retrying, {len(df) - len(already_done)} total remaining.\n")
    else:
        already_done = set()
        df["category_1"] = pd.NA
        df["category_2"] = pd.NA
        print(f"Fresh start — processing {len(df)} rows.\n")

    pending = [
        (i, str(row[INPUT_COL]), str(row[EMOTION_COL]))
        for i, row in df.iterrows()
        if i not in already_done
    ]

    if not pending:
        print("All rows already categorized!")
        return

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(get_category, i, q, e): i for i, q, e in pending}
            with tqdm(total=len(pending), desc="Categorizing") as pbar:
                for future in as_completed(futures):
                    idx, cat1, cat2 = future.result()
                    save_result(df, idx, cat1, cat2)
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
    finally:
        with save_lock:
            df.to_csv(OUTPUT_CSV, index=False)
        done = df["category_1"].notna().sum()
        dual = df["category_2"].notna().sum()
        print(f"Saved. {done}/{len(df)} rows categorized ({dual} with dual categories).")

    print(f"\nDone! Saved to {OUTPUT_CSV}")
    print("\nCategory 1 distribution:")
    print(df["category_1"].value_counts().to_string())
    print("\nCategory 2 distribution (dual-theme rows only):")
    print(df["category_2"].dropna().value_counts().to_string())

if __name__ == "__main__":
    main()