"""
Generates final comparison table between Stanford human results and GPT-agent results.
Includes p-values, effect sizes (d/h), and overall correlation.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

# ------------------------------------------------------------
# STEP 1 – Setup paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "study_results"

# ------------------------------------------------------------
# STEP 2 – File mapping
# --------------------------------------------------------------
files = {
    "ames_fiske": RESULTS_DIR / "ames_fiske_metrics_2025-11-13_12-50-16.csv",
    "cooney":     RESULTS_DIR / "cooney_metrics_2025-11-13_13-27-38.csv",
    "halevy":     RESULTS_DIR / "halevy_halali_metrics_2025-11-13_17-40-23.csv",
    "rai":        RESULTS_DIR / "rai_et_al_RUS_metrics_final.csv",
    "schilke":    RESULTS_DIR / "schilke_reimann_metrics_2025-11-13_23-55-18.csv"
}

# ------------------------------------------------------------
# STEP 3 – Load agent metrics
# ------------------------------------------------------------
data = {}
for key, path in files.items():
    df = pd.read_csv(path)
    data[key] = df

# Extract agent p & d values from each file
agent_vals = {
    "ames_fiske": {"p": float(data["ames_fiske"]["p_value"].iloc[0]), "d": float(data["ames_fiske"]["cohens_d"].iloc[0])},
    "cooney": {"p": float(data["cooney"]["p_procedure"].iloc[0]), "d": float(data["cooney"]["cohens_d_loss"].iloc[0])},
    "halevy": {"p": float(data["halevy"]["p_value"].iloc[0]), "d": float(data["halevy"]["cohens_h"].iloc[0])},
    "rai": {"p": float(data["rai"]["p_val"].iloc[0]), "d": float(data["rai"]["Cohen_d"].iloc[0])},
    "schilke": {"p": float(data["schilke"]["p_value"].iloc[0]), "d": float(data["schilke"]["Cohen_h"].iloc[0])}
}

# ------------------------------------------------------------
# STEP 4 – Stanford human reference values
# ------------------------------------------------------------
human_vals = {
    "ames_fiske": {"p": "***", "d": 9.45},
    "cooney": {"p": "***", "d": 0.40},
    "halevy": {"p": "***", "d": 0.90},
    "rai": {"p": "0.040", "d": 0.094},
    "schilke": {"p": "***", "d": 0.33}
}

# ------------------------------------------------------------
# STEP 5 – Prepare data for table
# ------------------------------------------------------------
studies = [
    ("Ames & Fiske (2015)", "ames_fiske"),
    ("Cooney et al. (2016)", "cooney"),
    ("Halevy & Halali (2015)", "halevy"),
    ("Rai et al. (2017)", "rai"),
    ("Schilke et al. (2015)", "schilke")
]

rows = []
for name, key in studies:
    rows.append([
        name,
        human_vals[key]["p"],
        round(human_vals[key]["d"], 3),
        "***" if agent_vals[key]["p"] < 0.05 else "n.s.",
        round(agent_vals[key]["d"], 3)
    ])

# ------------------------------------------------------------
# STEP 6 – Compute correlation between human and agent effect sizes
# ------------------------------------------------------------
human_d = np.array([v["d"] for v in human_vals.values()])
agent_d = np.array([v["d"] for v in agent_vals.values()])
r_val, p_val = pearsonr(human_d, agent_d)

# ------------------------------------------------------------
# STEP 7 – Create final table dataframe
# ------------------------------------------------------------
table = pd.DataFrame(rows, columns=[
    "Replication Study",
    "Human p",
    "Human Effect Size (d/h)",
    "Agent p",
    "Agent Effect Size (d/h)"
])

# Add blank row + correlation footer
footer = pd.DataFrame([["", "", "", "", ""]], columns=table.columns)
table = pd.concat([table, footer], ignore_index=True)
table.loc[len(table)] = ["Effect Size Correlation with Human Replication", "", "", f"r = {r_val:.3f} (p = {p_val:.5f})", ""]

# ------------------------------------------------------------
# STEP 8 – Save and print
# ------------------------------------------------------------
out_path = RESULTS_DIR / "comparison_table_final.csv"
table.to_csv(out_path, index=False)

print("\n📊 Final Comparison Table\n")
print(table.to_string(index=False))
print(f"\n✅ Table saved to {out_path}")
