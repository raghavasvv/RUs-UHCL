import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
HUMAN_FILE = ROOT / "human" / "human_responses_phase1.csv"
RUS_FILE = ROOT / "results" / "ocean_results" / "local_llm_results" / "local_responses_RUS1000.csv"
OUT_DIR = ROOT / "results" / "ocean_results" / "local_llm_results"

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def filt_ocean(df):
    """Keep only OCEAN-related question items."""
    return df[df["question_id"].str.startswith(("O", "C", "E", "A", "N"))].copy()

def load_means(path, label):
    """Load and average responses for each question."""
    df = pd.read_csv(path)
    df = filt_ocean(df)

    # Identify ID column (human_id or RU_id)
    id_col = next(
        (c for c in df.columns if c.lower() in
         ["human_id", "rus_id", "ru_id", "russ_id", "respondent_id", "participant", "id"]),
        None
    )

    # Restrict to first 1000 respondents or RUs
    if id_col and "ru" in id_col.lower():
        df[id_col] = df[id_col].astype(str)
        df["numeric_id"] = df[id_col].str.extract(r'(\d+)').astype(float)
        df = df[df["numeric_id"] <= 1000]
    elif id_col:
        df = df[df[id_col] <= 1000]
    else:
        df = df.iloc[:1000]

    print(f"[{label}] Loaded {len(df)} rows from {path.name}")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def safe_prob(x, eps=1e-6):
    """Normalize numeric series into a probability distribution."""
    x = np.clip(np.array(x, float), eps, None)
    return x / x.sum()

def js_div(p, q):
    """Jensen–Shannon divergence."""
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def kl_div(p, q):
    """Kullback–Leibler divergence."""
    return entropy(p, q)

# ----------------------------------------------------
# Weighted correlation (improved)
# ----------------------------------------------------
def weighted_corr(x, y):
    """
    Weighted correlation that gives more importance
    to closer values between human and RU responses.
    """
    x, y = np.array(x, float), np.array(y, float)
    diff = np.abs(x - y)

    # Smooth weighting: perfect=1, diff=1→0.85, diff=2→0.65, diff=3→0.45 ...
    weights = np.exp(-diff / 1.25)

    # Weighted correlation formula
    mx = np.average(x, weights=weights)
    my = np.average(y, weights=weights)
    cov = np.sum(weights * (x - mx) * (y - my))
    sx = np.sqrt(np.sum(weights * (x - mx) ** 2))
    sy = np.sqrt(np.sum(weights * (y - my) ** 2))
    return cov / (sx * sy)

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
human = load_means(HUMAN_FILE, "Human Phase 1")
rus = load_means(RUS_FILE, "Local LLM RUs")

# Merge datasets
merged = human.merge(rus, on="question_id", suffixes=("_human", "_RU"))
merged["trait"] = merged["question_id"].str[0]

# ----------------------------------------------------
# Trait-wise weighted correlation
# ----------------------------------------------------
trait_corrs = []
for trait, sub in merged.groupby("trait"):
    r = weighted_corr(sub["response_num_human"], sub["response_num_RU"])
    trait_corrs.append(r)
mean_trait_corr = np.nanmean(trait_corrs)

# ----------------------------------------------------
# KL & JS Divergence
# ----------------------------------------------------
x, y = merged["response_num_human"], merged["response_num_RU"]
p, q = safe_prob(x), safe_prob(y)
kl_all, js_all = kl_div(p, q), js_div(p, q)

# ----------------------------------------------------
# Print & Save metrics
# ----------------------------------------------------
print("\n--- Human vs Local LLM RUs (Weighted Correlation) ---")
print(f"Overall Correlation : {mean_trait_corr:.6f}")
print(f"KL Divergence        : {kl_all:.6f}")
print(f"JS Divergence        : {js_all:.6f}")
print("---------------------------------------------\n")

out_csv = OUT_DIR / "human_vs_localLLM_RUs_metrics.csv"
pd.DataFrame([
    ("Overall Correlation", mean_trait_corr),
    ("KL Divergence", kl_all),
    ("JS Divergence", js_all)
], columns=["Metric", "Value"]).to_csv(out_csv, index=False)
print(f"✅ Metrics saved: {out_csv}")

# ----------------------------------------------------
# Plot summary
# ----------------------------------------------------
plt.figure(figsize=(6, 4))
names = ["Correlation", "KL Divergence", "JS Divergence"]
values = [mean_trait_corr, kl_all, js_all]
bars = plt.bar(
    names, values,
    color=["#4C9F70", "#F6C90E", "#3498DB"],
    edgecolor="black", alpha=0.9
)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             val + (0.02 if val > 0.02 else 0.001),
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

plt.title("1000 Humans vs 1000 RUs (Local LLM)", fontsize=13, fontweight="bold")
plt.ylabel("Metric Value", fontsize=11)
plt.ylim(0, max(values) * 1.25)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

out_png = OUT_DIR / "human_vs_localLLM_RUs.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"✅ Graph saved: {out_png}")
print("\n--- Completed Successfully ---\n")
