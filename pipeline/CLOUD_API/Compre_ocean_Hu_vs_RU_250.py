import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy, ttest_ind
import matplotlib.pyplot as plt
import re
import sys

# -----------------------------------------------------
# 1. Path setup (portable on all systems)
# -----------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
for parent in CURRENT_FILE.parents:
    if (parent / "pipeline").is_dir():
        ROOT = parent
        break
else:
    ROOT = CURRENT_FILE.parents[2]

print(f"✅ Using project root: {ROOT}\n")

HUMAN_DIR = ROOT / "human"
RESULTS_DIR = ROOT / "results" / "ocean_results"

PHASE1 = HUMAN_DIR / "human_responses_phase1.csv"
RUS100 = RESULTS_DIR / "cloudresponses_100.csv"
RUS250 = RESULTS_DIR / "cloudresponses_250.csv"

# -----------------------------------------------------
# 2. File checks
# -----------------------------------------------------
for path in [PHASE1, RUS100, RUS250]:
    if not path.exists():
        print(f"⚠️ Missing file: {path}")
print()

# -----------------------------------------------------
# 3. Helper functions
# -----------------------------------------------------
def filt_ocean(df):
    """Keep only OCEAN-related items."""
    return df[df["question_id"].str.startswith(("O", "C", "E", "A", "N"))].copy()

def load_human(path, limit):
    """Load first N humans (by human_id)."""
    df = pd.read_csv(path)
    df = filt_ocean(df)
    if "human_id" in df.columns:
        df = df[df["human_id"] <= limit]
        unique = df["human_id"].nunique()
    else:
        df = df.iloc[:limit]
        unique = "N/A"
    print(f"📘 {path.name}: loaded {unique} humans, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def load_rus(path, limit):
    """Load first N RUs (by RU_id or RU_###)."""
    if not path.exists():
        print(f"❌ File not found: {path}")
        return pd.DataFrame(columns=["question_id", "response_num"])

    df = pd.read_csv(path)
    df = filt_ocean(df)

    if "RU_id" in df.columns:
        df["rus_num"] = df["RU_id"].apply(lambda x: int(re.sub(r"[^0-9]", "", str(x))) if re.search(r"\d+", str(x)) else None)
        df = df[df["rus_num"] <= limit]
        unique = df["rus_num"].nunique()
    else:
        df = df.iloc[:limit]
        unique = "N/A"

    print(f"🤖 {path.name}: loaded {unique} RUs, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def safe_prob(x, eps=1e-6):
    x = np.clip(np.array(x, float), eps, None)
    return x / x.sum()

def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def kl_div(p, q):
    return entropy(p, q)

def compute(df1, df2):
    if df1.empty or df2.empty:
        print("⚠️ One or both DataFrames are empty — skipping compute.")
        return {m: np.nan for m in ["Correlation", "KL Divergence", "JS Divergence", "t-statistic", "p-value"]}

    m = df1.merge(df2, on="question_id", suffixes=("_H", "_R"))
    if m.empty:
        print("⚠️ No overlapping questions between datasets.")
        return {m: np.nan for m in ["Correlation", "KL Divergence", "JS Divergence", "t-statistic", "p-value"]}

    x, y = m["response_num_H"], m["response_num_R"]
    corr, _ = pearsonr(x, y)
    p, q = safe_prob(x), safe_prob(y)
    kl, js = kl_div(p, q), js_div(p, q)
    t_stat, p_val = ttest_ind(x, y, equal_var=False)
    return {
        "Correlation": corr,
        "KL Divergence": kl,
        "JS Divergence": js,
        "t-statistic": t_stat,
        "p-value": p_val
    }

# -----------------------------------------------------
# 4. Load and compute
# -----------------------------------------------------
h_100 = load_human(PHASE1, 100)
r_100 = load_rus(RUS100, 100)

h_250 = load_human(PHASE1, 250)
r_250 = load_rus(RUS250, 250)

m100 = compute(h_100, r_100)
m250 = compute(h_250, r_250)

# -----------------------------------------------------
# 5. Save CSV
# -----------------------------------------------------
output_csv = RESULTS_DIR / "human_vs_RUS_100_250_metrics.csv"
pd.DataFrame({
    "Metric": list(m100.keys()),
    "Human100_vs_RUS100": list(m100.values()),
    "Human250_vs_RUS250": list(m250.values())
}).to_csv(output_csv, index=False)
print(f"✅ Metrics saved to: {output_csv}")

# -----------------------------------------------------
# 6. Plot with value labels
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
metrics = ["Correlation", "KL Divergence", "JS Divergence"]
x = np.arange(len(metrics))
w = 0.35

bars1 = plt.bar(x - w/2, [m100[m] for m in metrics], width=w,
                label="H100 ↔ R100", color="#1F77B4", edgecolor="black")
bars2 = plt.bar(x + w/2, [m250[m] for m in metrics], width=w,
                label="H250 ↔ R250", color="#FF7F0E", edgecolor="black")

def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            plt.text(bar.get_x() + bar.get_width()/2,
                     h + (0.01 if h >= 0 else -0.02),
                     f"{h:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

add_labels(bars1)
add_labels(bars2)

plt.xticks(x, metrics, rotation=15, ha="right")
plt.ylabel("Value")
plt.title("Human vs RUs (OCEAN) — 100 & 250 Sample Comparison", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

output_png = RESULTS_DIR / "human_vs_RUS_100_250_metrics.png"
plt.savefig(output_png, dpi=300)
plt.close()
print(f"✅ Graph saved with value labels: {output_png}")
