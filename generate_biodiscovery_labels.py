import pandas as pd
import json
import time
import anthropic

# === CONFIG ===
ANNOTATION_FILE = "datasets/flybase/fbgn_annotation_ID_fb_2025_03.tsv"
SNAPSHOTS_FILE = "datasets/flybase/gene_snapshots_fb_2025_03.tsv"
OUTPUT_FILE = "train.json"
API_KEY_FILE = "Claude_api_key.txt"
MODEL_NAME = "claude-3-5-sonnet-20241022"  # Updated to current model
RATE_LIMIT_SECONDS = 1  # pause between API calls

# === Load Anthropic API key and initialize client ===
with open(API_KEY_FILE, "r") as f:
    api_key = f.read().strip()

client = anthropic.Anthropic(api_key=api_key)

# === Load FlyBase data ===
print("Loading FlyBase annotation file...")
annotation_df = pd.read_csv(ANNOTATION_FILE, sep="\t", comment="#", header=None)
print(annotation_df.head(10))
print("Annotation columns:", annotation_df.columns.tolist())
print(f"Loaded {len(annotation_df)} rows from annotation file.")

print("\nLoading FlyBase gene snapshots...")
snapshots_df = pd.read_csv(
    SNAPSHOTS_FILE, sep="\t", comment="#", names=["FBgn", "Symbol", "Summary"]
)
print(f"Loaded {len(snapshots_df)} rows from snapshots file.")
print("Snapshots columns:", snapshots_df.columns.tolist())

# === Check if annotation file has the expected FBgn column ===
# FlyBase annotation files typically have FBgn in the first column
if annotation_df.shape[1] >= 1:
    annotation_df.columns = ["FBgn"] + [f"col_{i}" for i in range(1, annotation_df.shape[1])]
    print(f"Assigned column names to annotation file: {annotation_df.columns.tolist()}")
else:
    print("Warning: Annotation file structure unexpected")

# === Merge annotation and snapshots on FBgn ID ===
merged_df = pd.merge(
    annotation_df, snapshots_df, left_on="FBgn", right_on="FBgn", how="inner"
)
print(f"Merged dataset contains {len(merged_df)} genes.")

# === Function to query Claude ===
def get_label(summary):
    if not summary or pd.isna(summary):
        return "Unknown function"
    
    prompt = f"""Summarize the biological function of this Drosophila gene in one short sentence.

Description:
{summary}

Label:"""
    
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=100,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        # Fixed: Use .content[0].text instead of ["completion"]
        label = response.content[0].text.strip()
        return label
    except Exception as e:
        print(f"Claude API error: {e}")
        return "Unknown function"

# === Generate train data ===
train_data = []
print("\nGenerating labels using Claude...")

for i, row in merged_df.iterrows():
    symbol = row["Symbol"]
    summary = row["Summary"]
    label = get_label(summary)
    
    train_data.append({
        "gene": symbol,
        "text": summary if pd.notna(summary) else "No description available.",
        "label": label,
    })
    
    print(f"[{i+1}/{len(merged_df)}] {symbol}: {label}")
    time.sleep(RATE_LIMIT_SECONDS)

# === Save to JSON ===
with open(OUTPUT_FILE, "w") as f:
    json.dump(train_data, f, indent=2)

print(f"\nDone! Wrote {len(train_data)} entries to {OUTPUT_FILE}.")