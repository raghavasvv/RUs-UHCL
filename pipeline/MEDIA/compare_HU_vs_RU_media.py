import pandas as pd, numpy as np, json, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[2]  # ✅ one level up to /capstone3
RESULTS_DIR = ROOT / "results" / "media"

RUS_FILE = ROOT / "results" / "media" / "media_RUs.csv"   # ✅ changed file name
HUMAN_FILE = ROOT / "results" / "media" / "media_human_resp.json"
OUT_JSON = RESULTS_DIR / "media_RUs_percentage.json"       # ✅ renamed output

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load data
# -------------------------
rus_df = pd.read_csv(RUS_FILE)
with open(HUMAN_FILE, "r") as f:
    human_data = json.load(f)

print(f"✅ Loaded {len(rus_df)} RU responses and {len(human_data)} human survey questions")

# -------------------------
# Compute RU percentage per question
# -------------------------
rus_results = []
for q in human_data:
    qid = q["id"]
    human_dist = q["human_response_percent"]
    options = list(human_dist.keys())

    subset = rus_df[rus_df["question_id"] == qid]
    total = len(subset)
    counts = {opt: 0 for opt in options}

    for resp in subset["response"]:
        if resp in counts:
            counts[resp] += 1
        else:
            counts.setdefault(resp, 0)

    percentages = {opt: (v / total) * 100 if total else 0 for opt, v in counts.items()}
    rus_results.append({"question_id": qid, "RU_distribution": percentages})  # ✅ renamed key

# -------------------------
# Save RU percentage JSON
# -------------------------
with open(OUT_JSON, "w") as f:
    json.dump(rus_results, f, indent=2)
print(f"📄 Saved RU percentage distributions → {OUT_JSON}")

# -------------------------
# Compare with human data (only KL & JS)
# -------------------------
metrics = []
for q in human_data:
    qid = q["id"]
    human_dist = q["human_response_percent"]
    ru_dist = next((x["RU_distribution"] for x in rus_results if x["question_id"] == qid), None)
    if not ru_dist:
        continue

    options = list(human_dist.keys())
    h_vals = np.array([human_dist[o] for o in options])
    r_vals = np.array([ru_dist.get(o, 0) for o in options])

    h_probs = h_vals / h_vals.sum()
    r_probs = r_vals / r_vals.sum()

    kl = float(entropy(r_probs, h_probs))
    js = float(jensenshannon(r_probs, h_probs))

    metrics.append({
        "question_id": qid,
        "kl_divergence": float(kl),
        "js_divergence": float(js)
    })

# -------------------------
# Save and print metrics
# -------------------------
df = pd.DataFrame(metrics)

print("\n==== Question-wise Metrics (KL & JS only) ====")
for _, row in df.iterrows():
    print(f"Q{int(row['question_id'])}: KL = {row['kl_divergence']:.6f}, JS = {row['js_divergence']:.6f}")

print("\n==== Overall Metrics ====")
print(f"Average KL Divergence : {df['kl_divergence'].mean():.4f}")
print(f"Average JS Divergence : {df['js_divergence'].mean():.4f}")
print("=========================")

# -------------------------
# 📊 Visualization Section
# -------------------------
plt.figure(figsize=(8, 5))
x = np.arange(len(df))
width = 0.35

bars1 = plt.bar(x - width/2, df["kl_divergence"], width, label="KL Divergence", color="#ff7f0e")
bars2 = plt.bar(x + width/2, df["js_divergence"], width, label="JS Divergence", color="#2ca02c")

plt.xticks(x, [f"Q{int(i)}" for i in df["question_id"]])
plt.xlabel("Question ID")
plt.ylabel("Metric Value")
plt.title("Media Human vs RUs (Cloud) — KL & JS Divergence per Question")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

def label_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=9
        )

label_bars(bars1)
label_bars(bars2)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "media_RUs_metrics_bar.png", dpi=300)
plt.close()
print(f"🖼️ Saved detailed bar chart → {RESULTS_DIR / 'media_RUs_metrics_bar.png'}")

# === Average Metrics Graph (Fixed Label Overlap) ===
avg_kl = df["kl_divergence"].mean()
avg_js = df["js_divergence"].mean()

plt.figure(figsize=(6, 5))
metrics_names = ["KL Divergence", "JS Divergence"]
values = [avg_kl, avg_js]
colors = ["#ff7f0e", "#2ca02c"]

bars = plt.bar(metrics_names, values, color=colors, width=0.5)
plt.ylabel("Average Metric Value", fontsize=11)
plt.title("Average KL & JS Divergence — Media Human vs RUs (Cloud)", fontsize=13, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Add text labels with auto offset
for bar in bars:
    height = bar.get_height()
    offset = height * 0.05  # 5% of bar height as spacing
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + offset,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black"
    )

# Expand Y limit dynamically for safe headroom
max_val = max(values)
plt.ylim(0, max_val * 1.25)  # Adds 25% margin on top

plt.tight_layout(pad=2.0)
plt.savefig(RESULTS_DIR / "media_RUs_metrics_average.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"🖼️ Saved average metrics chart → {RESULTS_DIR / 'media_RUs_metrics_average.png'}")
