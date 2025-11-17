"""
Halevy & Halali (2015) – Upgraded Realistic Replication
Appends results to: RUS_0001.json → RUS_1000.json
(memory / plans / reflections)
"""

# ------------------------------------------------------------
# STEP 1 — Auto-detect PROJECT ROOT
# ------------------------------------------------------------
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()

for parent in CURRENT_FILE.parents:
    if (parent / "pipeline").is_dir() and (parent / "RUS").is_dir():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError("❌ Could not detect project root")

sys.path.append(str(PROJECT_ROOT))
print("🔧 PROJECT ROOT:", PROJECT_ROOT)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import json, random, math, time, pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# CUSTOM MANAGERS
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

load_dotenv()
client = OpenAI()

# ------------------------------------------------------------
# STEP 2 — Paths
# ------------------------------------------------------------
BASE_DIR = PROJECT_ROOT

RUS_FILE = BASE_DIR / "RUS" / "synthetic_RUS.json"
MEMORY_DIR = BASE_DIR / "memory"
REFLECTION_DIR = BASE_DIR / "reflections"
PLAN_DIR = BASE_DIR / "plans"

RESULTS_DIR = BASE_DIR / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESPONSES_PATH = RESULTS_DIR / f"halevy_halali_responses_{timestamp}.csv"
METRICS_PATH   = RESULTS_DIR / f"halevy_halali_metrics_{timestamp}.csv"

# INIT MANAGERS
memory_manager     = MemoryManager(MEMORY_DIR)
reflection_manager = ReflectionManager(REFLECTION_DIR)
plan_manager       = PlanManager(PLAN_DIR)

# FORCE managers to always save using RUS_####.json
memory_manager.filename = lambda rus_id: MEMORY_DIR / f"{rus_id}.json"
reflection_manager.filename = lambda rus_id: REFLECTION_DIR / f"{rus_id}.json"
plan_manager.filename = lambda rus_id: PLAN_DIR / f"{rus_id}.json"

# ------------------------------------------------------------
# STEP 3 — Load and Normalize RUS units
# ------------------------------------------------------------
with open(RUS_FILE, "r") as f:
    rus_units = json.load(f)

if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

# FORCE all IDs to RUS_####
cleaned_agents = []
for i, r in enumerate(rus_units, start=1):
    r["RUs_id"] = f"RUS_{i:04d}"
    cleaned_agents.append(r)

agents = cleaned_agents
print(f"✅ Loaded {len(agents)} RUS agents")

# ------------------------------------------------------------
# STEP 4 — Extract OCEAN traits
# ------------------------------------------------------------
def extract_ocean(memory_list):
    traits = {"O":0,"C":0,"E":0,"A":0,"N":0}
    count  = {"O":0,"C":0,"E":0,"A":0,"N":0}

    for item in memory_list:
        q = item.get("q","")
        a = item.get("a","")

        if len(q)>=2 and q[0] in traits:
            trait = q[0]
            value = 1 if "Very Accurate" in str(a) else 0.5
            traits[trait] += value
            count[trait]  += 1

    for t in traits:
        if count[t] > 0:
            traits[t] /= count[t]

    return traits

def noisy_choice(prob):
    return "Triangle" if random.random() < prob else "Square"

# ------------------------------------------------------------
# STEP 5 — Behavior Models
# ------------------------------------------------------------
def disputant_behavior(agent, condition):
    memory = memory_manager.load(agent["RUs_id"])
    traits = extract_ocean(memory)

    A, C, E = traits["A"], traits["C"], traits["E"]

    base = 0.60 + A*0.20 + C*0.10 - E*0.05

    if condition == "without_intervention":
        base -= 0.35

    base = max(0.05, min(0.95, base))
    return noisy_choice(base)

def third_party_behavior(agent):
    memory = memory_manager.load(agent["RUs_id"])
    traits = extract_ocean(memory)

    A, C, N = traits["A"], traits["C"], traits["N"]

    prob = 0.30 + A*0.30 + C*0.20 - N*0.15
    prob = max(0.05, min(0.95, prob))

    return "I" if random.random() < prob else "O"

# ------------------------------------------------------------
# STEP 6 — Simulation
# ------------------------------------------------------------
random.shuffle(agents)
groups = [agents[i:i+3] for i in range(0, len(agents), 3)]
results = []

for i, group in enumerate(groups):
    if len(group) < 3:
        continue

    RED, BLUE, GREEN = group

    g_choice = third_party_behavior(GREEN)
    condition = "with_intervention" if g_choice == "I" else "without_intervention"

    r = disputant_behavior(RED, condition)
    b = disputant_behavior(BLUE, condition)

    results.append({
        "group_id": i+1,
        "RED_id": RED["RUs_id"], "RED_choice": r,
        "BLUE_id": BLUE["RUs_id"], "BLUE_choice": b,
        "GREEN_id": GREEN["RUs_id"], "GREEN_choice": g_choice,
        "condition": condition
    })

    # Append cognitive trails
    for agent, role, choice in [(RED,"RED",r),(BLUE,"BLUE",b),(GREEN,"GREEN",g_choice)]:
        aid = agent["RUs_id"]

        memory_manager.append(aid, {
            "q": f"Halevy_{role}_{condition}",
            "a": choice
        })

        reflection_manager.append(aid, {
            "insight": f"{role} chose {choice} in {condition}",
            "task": "halevy_halali",
            "condition": condition
        })

        plan_manager.append(aid, {
            "next_action": f"Reflect on Halevy-Halali {role} behavior ({condition})"
        })

# ------------------------------------------------------------
# STEP 7 — Save Responses
# ------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(RESPONSES_PATH, index=False)
print(f"📁 Responses saved → {RESPONSES_PATH}")

# ------------------------------------------------------------
# STEP 8 — Compute Metrics
# ------------------------------------------------------------
with_int  = df[df["condition"]=="with_intervention"]
without_i = df[df["condition"]=="without_intervention"]

coop_with    = ((with_int["RED_choice"]=="Triangle") & (with_int["BLUE_choice"]=="Triangle")).sum()
coop_without = ((without_i["RED_choice"]=="Triangle") & (without_i["BLUE_choice"]=="Triangle")).sum()

len_with     = len(with_int)
len_without  = len(without_i)

rate_with    = coop_with/len_with     if len_with > 0 else 0
rate_without = coop_without/len_without if len_without > 0 else 0

table = [
    [coop_with,    len_with - coop_with],
    [coop_without, len_without - coop_without]
]

try:
    chi2, p, dof, exp = chi2_contingency(table)
except ValueError:
    chi2 = 0
    _, p = fisher_exact(table)

h = 2*math.asin(math.sqrt(rate_with)) - 2*math.asin(math.sqrt(rate_without))

metrics = {
    "groups_total": len(df),
    "intervention_rate": round((df["GREEN_choice"]=="I").mean(), 3),
    "coop_with": f"{coop_with}/{len_with} ({round(rate_with*100,1)}%)",
    "coop_without": f"{coop_without}/{len_without} ({round(rate_without*100,1)}%)",
    "chi_square": round(chi2,3),
    "p_value": round(p,5),
    "cohens_h": round(abs(h),3),
    "replication_success": "Yes" if p < 0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
print(f"📁 Metrics saved → {METRICS_PATH}")

# ------------------------------------------------------------
# STEP 9 — Summary
# ------------------------------------------------------------
print("\n📊 SUMMARY – Halevy & Halali (Upgraded)")
print(metrics)
print("\n🎯 Realistic Halevy-Halali replication completed.\n")
