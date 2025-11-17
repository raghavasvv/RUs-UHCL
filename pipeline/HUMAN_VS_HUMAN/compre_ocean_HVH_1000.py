import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy, ttest_ind
import matplotlib.pyplot as plt

# =====================================================
# 1. Portable Root Detection
# =====================================================
CURRENT_FILE = Path(__file__).resolve()

# Walk upward until we find the project root (contains "human" and "results")
for parent in CURRENT_FILE.parents:
    if (parent / "human").is_dir() and (parent / "results").is_dir():
        ROOT = parent
        break
else:
    raise RuntimeError("❌ Could not find project root (missing 'human' or 'results' folder).")

print(f"✅ Using project root: {ROOT}")

HUMAN_DIR = ROOT / "human"
PHASE1 = HUMAN_DIR / "human_responses_phase1.csv"
PHASE2 = HUMAN_DIR / "human_responses_phase2.csv"

# =====================================================
# 2. Helper Functions
# =====================================================
def filt_ocean(df):
    """Keep only OCEAN-related items."""
    return df[df["question_id"].str.startswith(("O", "C", "E", "A", "N"))].copy()

def load_mean(path, limit):
    """Load first N humans (by human_id) and compute mean per question."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")
    df = pd.read_csv(path)
    df = filt_ocean(df)

    # Identify correct ID column
    col = None
    for c in df.columns:
        if c.lower() in ["human_id", "humanid", "respondent_id", "participant", "id"]:
            col = c
            break

    if col:
        df = df[df[col] <= limit]
        unique = df[col].nunique()
    else:
        df = df.iloc[:limit]
        unique = "N/A"

    print(f"📘 {path.name}: loaded {unique} humans, {len(df)} rows")
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
    merged = df1.merge(df2, on="question_id", suffixes=("_1", "_2"))
    if merged.empty:
        print("⚠️ No overlapping OCEAN questions between datasets.")
        return {m: np.nan for m in ["Correlation", "KL Divergence", "JS Divergence", "t-statistic", "p-value"]}
    x, y = merged["response_num_1"], merged["response_num_2"]
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

# =====================================================
# 3. Load and Compute
# =====================================================
h1_1000 = load_mean(PHASE1, 1000)
h2_1000 = load_mean(PHASE2, 1000)
metrics = compute(h1_1000, h2_1000)

# =====================================================
# 4. Save Metrics CSV
# =====================================================
out_csv = HUMAN_DIR / "human_vs_human_metrics_1000.csv"
pd.DataFrame({
    "Metric": list(metrics.keys()),
    "Human1000_vs_Human1000": list(metrics.values())
}).to_csv(out_csv, index=False)
print(f"✅ Saved metrics to: {out_csv}")

# =====================================================
# 5. Visualization
# =====================================================
plt.figure(figsize=(8, 5))
metrics_list = ["Correlation", "KL Divergence", "JS Divergence"]
values = [metrics[m] for m in metrics_list]

bars = plt.bar(metrics_list, values,
               color=["#4C9F70", "#F6C90E", "#3498DB"],
               edgecolor="black", alpha=0.9)

for bar, val in zip(bars, values):
    if not np.isnan(val):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

plt.title("Human vs Human (1000 Participants)", fontsize=13, fontweight="bold")
plt.ylabel("Metric Value")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

out_png = HUMAN_DIR / "human_vs_human_metrics_1000.png"
plt.savefig(out_png, dpi=300)
plt.close()
print(f"✅ Graph saved with value labels: {out_png}")
