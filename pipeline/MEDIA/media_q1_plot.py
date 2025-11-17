import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]   # ✅ one level higher
RESULTS_DIR = ROOT / "results" / "media"

CSV_2020 = RESULTS_DIR / "media_q1_RUs_2020.csv"
CSV_2025 = RESULTS_DIR / "media_q1_RUs_2025.csv"
OUT_PNG = RESULTS_DIR / "media_q1_trend_chart.png"

# -----------------------------
# Load Data
# -----------------------------
df_2020 = pd.read_csv(CSV_2020)
df_2025 = pd.read_csv(CSV_2025)

# count responses
count_2020 = df_2020["response"].value_counts(normalize=True) * 100
count_2025 = df_2025["response"].value_counts(normalize=True) * 100

# ensure consistent order
labels = ["Right direction", "Wrong direction", "No opinion"]
year_2020 = [count_2020.get(label, 0) for label in labels]
year_2025 = [count_2025.get(label, 0) for label in labels]

# -----------------------------
# DataFrame for plotting
# -----------------------------
trend_df = pd.DataFrame({
    "Category": labels,
    "2020": year_2020,
    "2025": year_2025
})

# -----------------------------
# Plot (Pew-style)
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(["2020", "2025"], [year_2020[0], year_2025[0]], marker="o", label="Right direction", color="#1B998B", linewidth=3)
plt.plot(["2020", "2025"], [year_2020[1], year_2025[1]], marker="o", label="Wrong direction", color="#E9C46A", linewidth=3)
plt.plot(["2020", "2025"], [year_2020[2], year_2025[2]], marker="o", label="No opinion", color="#A8A8A8", linewidth=3, linestyle="--")

for year, yvals in zip(["2020", "2025"], [year_2020, year_2025]):
    plt.text(year, yvals[0] + 1, f"{yvals[0]:.0f}%", ha="center", color="#1B998B", fontweight="bold")
    plt.text(year, yvals[1] + 1, f"{yvals[1]:.0f}%", ha="center", color="#E9C46A", fontweight="bold")
    plt.text(year, yvals[2] + 1, f"{yvals[2]:.0f}%", ha="center", color="#6E6E6E")

plt.title("Trend in RU's' Perception of U.S. Higher Education (2020–2025)", fontsize=13, weight="bold")
plt.ylabel("% of RU's")
plt.ylim(0, 80)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"✅ Trend chart saved to: {OUT_PNG}")
print(trend_df.round(1))
