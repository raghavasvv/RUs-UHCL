import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# =====================================================
# 1. Portable Path Setup
# =====================================================
CURRENT_FILE = Path(__file__).resolve()
for parent in CURRENT_FILE.parents:
    if (parent / "human").is_dir() and (parent / "results").is_dir():
        ROOT = parent
        break
else:
    raise RuntimeError("❌ Could not find project root (missing 'human' or 'results' folder).")

print(f"\n✅ Using project root: {ROOT}")

# =====================================================
# 2. File Paths
# =====================================================
HUMAN_METRICS = ROOT / "human" / "human_vs_human_metrics_1000.csv"
CLOUD_METRICS = ROOT / "results" / "ocean_results" / "human_vs_cloud1000_metrics.csv"

OUT_CSV = ROOT / "results" / "ocean_results" / "normalized_accuracy_cloud1000.csv"
OUT_PNG = ROOT / "results" / "ocean_results" / "normalized_accuracy_cloud1000.png"

print("\n--- Stanford-Style Normalized Accuracy (1000 Humans vs 1000 Cloud Agents) ---")
print(f"Using Human metrics file : {HUMAN_METRICS}")
print(f"Using Cloud metrics file : {CLOUD_METRICS}\n")

# =====================================================
# 3. Load Metric Files
# =====================================================
if not HUMAN_METRICS.exists():
    raise FileNotFoundError(f"❌ Missing: {HUMAN_METRICS}")
if not CLOUD_METRICS.exists():
    raise FileNotFoundError(f"❌ Missing: {CLOUD_METRICS}")

human_df = pd.read_csv(HUMAN_METRICS)
cloud_df = pd.read_csv(CLOUD_METRICS)

# =====================================================
# 4. Extract Correlations
# =====================================================
human_corr = float(human_df.loc[human_df["Metric"] == "Correlation", human_df.columns[-1]])
cloud_corr = float(cloud_df.loc[cloud_df["Metric"] == "Correlation", cloud_df.columns[-1]])

# =====================================================
# 5. Compute Normalized Accuracy
# =====================================================
normalized = cloud_corr / human_corr if human_corr != 0 else float("nan")

print(f"Human internal consistency (H↔H): {human_corr:.3f}")
print(f"Human vs Cloud correlation (H↔A): {cloud_corr:.3f}")
print(f"Normalized accuracy:              {normalized:.3f}\n")

# =====================================================
# 6. Save to CSV
# =====================================================
df = pd.DataFrame({
    "Comparison": ["Human1000 vs Cloud1000"],
    "Human↔Human_Corr": [human_corr],
    "Human↔Cloud_Corr": [cloud_corr],
    "Normalized_Accuracy": [normalized]
})
df.to_csv(OUT_CSV, index=False)
print(f"✅ Results saved to: {OUT_CSV}")

# =====================================================
# 7. Visualization
# =====================================================
plt.figure(figsize=(6, 4))
bars = plt.bar(["1000 Humans vs Cloud Agents"], [normalized],
               color="#2E8B57", edgecolor="black", alpha=0.9)

for bar, val in zip(bars, [normalized]):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

plt.title("Normalized Accuracy – 1000 Humans vs Cloud Agents", fontsize=12, fontweight="bold")
plt.ylabel("Normalized Accuracy")
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"📈 Graph saved to: {OUT_PNG}\n")
