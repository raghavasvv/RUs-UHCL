import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
ROOT = Path("/Users/rachurivijay/Desktop/Capstone/teamross/capstone3")
RESULTS = ROOT / "results" / "ocean_results"

FILES = {
    "100/250": RESULTS / "ocean_results_250_RUS.csv",
    "500": RESULTS / "human_vs_cloud500_RUS_metrics.csv",
    "1000": RESULTS / "human_vs_cloud1000_RUS_metrics.csv"
}

# -----------------------------------------------------
# Extract metrics
# -----------------------------------------------------
metrics = {"Correlation": {}, "KL Divergence": {}, "JS Divergence": {}}

for label, path in FILES.items():
    if not path.exists():
        print(f"⚠️ Missing file: {path}")
        continue

    df = pd.read_csv(path)

    if label == "100/250":
        for metric in metrics.keys():
            m100 = float(df.loc[df["Metric"] == metric, "Human100_vs_RUS100"])
            m250 = float(df.loc[df["Metric"] == metric, "Human250_vs_RUS250"])
            metrics[metric]["100"] = m100
            metrics[metric]["250"] = m250
    else:
        for metric in metrics.keys():
            val = float(df.loc[df["Metric"] == metric].iloc[0, 1])
            metrics[metric][label] = val

# -----------------------------------------------------
# UHCL Hawks Color Palette
# -----------------------------------------------------
UHCL_BLUE = "#0057B8"     # UHCL blue (shirt)
UHCL_GREEN = "#006400"    # UHCL green (shoes)
UHCL_GOLD = "#FF8C00"     # UHCL gold (beak)
UHCL_TEXT = "#002D62"     # Navy outline tone

# -----------------------------------------------------
# Plot function (clean, UHCL Hawks theme)
# -----------------------------------------------------
def plot_metric(metric_name, ylabel, color):
    data = metrics[metric_name]
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=color, edgecolor="black", alpha=0.9)

    ymax = max(values) * 1.18  # More headroom for labels
    plt.ylim(0, ymax)

    # Add metric labels above bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + (ymax * 0.03),
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=UHCL_TEXT
        )

    plt.title(
        f"{metric_name} of Human vs RU's (Cloud) Split: 100, 250, 500, 1000",
        fontsize=13, fontweight="bold", color=UHCL_TEXT
    )
    plt.ylabel(ylabel, fontsize=11, color=UHCL_TEXT)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(pad=2.0)

    out_file = RESULTS / f"{metric_name.replace(' ', '_')}_UHCL_Hawks_Final.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_file.name}")

# -----------------------------------------------------
# Generate all 3 graphs (clean, mascot color theme)
# -----------------------------------------------------
plot_metric("Correlation", "Correlation Value", UHCL_BLUE)
plot_metric("KL Divergence", "KL Divergence Value", UHCL_GREEN)
plot_metric("JS Divergence", "JS Divergence Value", UHCL_GOLD)

print("\n🎯 All UHCL Hawks–themed charts generated cleanly (no overlaps, no interpretation box).")
