import pandas as pd
import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = Path("/Users/rachurivijay/Desktop/Capstone/teamross/capstone3/results/media/media_RUs.csv")  #change path accordingly
OUTPUT_JSON = Path("/Users/rachurivijay/Desktop/Capstone/teamross/capstone3/results/media/RUs_response_percentages.json")  ##change path accordingly

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Expected columns: RUs_id, question_id, response (and maybe timestamp)
if "question_id" not in df.columns or "response" not in df.columns:
    raise ValueError("CSV must contain 'question_id' and 'response' columns.")

# -----------------------------
# Compute percentages
# -----------------------------
output_data = []

for qid, group in df.groupby("question_id"):
    total = len(group)
    counts = group["response"].value_counts(normalize=True) * 100
    counts = counts.round(1)

    # Build dictionary for this question
    entry = {
        "id": int(qid),
        "RU_distribution": {opt: float(pct) for opt, pct in counts.items()}  
    }
    output_data.append(entry)

# -----------------------------
# Save JSON
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ RU response percentages saved to: {OUTPUT_JSON}")
print(json.dumps(output_data, indent=2))
