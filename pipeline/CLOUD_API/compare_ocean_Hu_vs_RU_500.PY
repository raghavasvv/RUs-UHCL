import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy, ttest_ind
import matplotlib.pyplot as plt

# -----------------------------------------------------
# File paths (same structure, just 500 version)
# -----------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # points to capstone3/
HUMAN_DIR = ROOT / "human"

HUMAN_FILE = HUMAN_DIR / "human_responses_phase1.csv"
RUS_FILE = ROOT / "results" / "ocean_results" / "cloud_responses_500.csv"  # ✅ verified style
RESULTS_DIR = ROOT / "results" / "ocean_results"

# -----------------------------------------------------
# Data preparation
# -----------------------------------------------------
def filter_ocean_items(df):
    """Select only OCEAN-related questions."""
    return df[df["question_id"].str.startswith(("O", "C", "E", "A", "N"))].copy()

def load_human_data(path):
    """Load human responses and compute mean per question."""
    df = pd.read_csv(path)
    df = filter_ocean_items(df)
    df["response_num"] = df["response_num"].astype(float)
    participant_count = df["RU_id"].nunique() if "RU_id" in df.columns else "N/A"
    print(f"{path.name}: loaded {participant_count} human participants, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def load_rus_data(path):
    """Load RUS (Reflector Unit) responses and compute mean per question."""
    df = pd.read_csv(path)
    df = filter_ocean_items(df)
    df["response_num"] = df["response_num"].astype(float)
    rus_count = df["RU_id"].nunique() if "RU_id" in df.columns else "N/A"
    print(f"{path.name}: loaded {rus_count} RUS units, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

# -----------------------------------------------------
# Statistical computation
# -----------------------------------------------------
def safe_distribution(values, eps=1e-6):
    arr = np.clip(np.array(values, dtype=float), eps, None)
    return arr / arr.sum()

def kl_divergence(p, q):
    return entropy(p, q)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def compute_metrics(df1, df2):
    merged = df1.merge(df2, on="question_id", suffixes=("_human", "_RUS"))
    x, y = merged["response_num_human"], merged["response_num_RUS"]

    correlation, _ = pearsonr(x, y)
    p, q = safe_distribution(x), safe_distribution(y)
    kl, js = kl_divergence(p, q), js_divergence(p, q)
    t_stat, p_val = ttest_ind(x, y, equal_var=False)

    return {
        "Correlation": correlation,
        "KL Divergence": kl,
        "JS Divergence": js,
        "t-statistic": t_stat,
        "p-value": p_val
    }

# -----------------------------------------------------
# Execution
# -----------------------------------------------------
human_df = load_human_data(HUMAN_FILE)
rus_df = load_rus_data(RUS_FILE)

metrics = compute_metrics(human_df, rus_df)

# -----------------------------------------------------
# Save results
# -----------------------------------------------------
output_csv = RESULTS_DIR / "human_vs_cloud500_RUS_metrics.csv"
pd.DataFrame({
    "Metric": metrics.keys(),
    "Human500_vs_RUS500": metrics.values()
}).to_csv(output_csv, index=False)
print(f"📊 Metrics saved to: {output_csv}")

# -----------------------------------------------------
# Visualization
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
metric_names = ["Correlation", "KL Divergence", "JS Divergence"]
metric_values = [metrics[m] for m in metric_names]

colors = ["#2E8B57", "#1F77B4", "#F6C90E"]

bars = plt.bar(metric_names, metric_values,
               color=colors, edgecolor="black", alpha=0.9)

for bar, val in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color="#2C2C2C")

plt.title("500 Humans vs 500 RU's (Cloud)", fontsize=14, fontweight="bold", color="#2C3E50")
plt.ylabel("Metric Value", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

output_png = RESULTS_DIR / "human_vs_cloud500_RUS_metrics.png"
plt.savefig(output_png, dpi=300)
plt.close()
print(f"📈 Graph saved to: {output_png}")
