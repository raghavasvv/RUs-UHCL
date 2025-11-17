"""
Ames & Fiske (2015) – OpenAI Edition (Append to existing Agent memory)
Each RU updates existing Agent_XXXX.json files (OCEAN → moral judgment).
No new RU_xxxx files are created.
"""

# ------------------------------------------------------------
# STEP 1 — Auto-detect PROJECT ROOT (cross-platform)
# ------------------------------------------------------------
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()

for parent in CURRENT_FILE.parents:
    if (parent / "pipeline").is_dir() and (parent / "RUS").is_dir():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError("❌ Could not detect project root (missing pipeline/ or RUS/)")

sys.path.append(str(PROJECT_ROOT))
print("🔧 PROJECT ROOT:", PROJECT_ROOT)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import json, math, pandas as pd, time, re
from scipy import stats
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Import custom managers AFTER adding PROJECT_ROOT
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# ------------------------------------------------------------
# STEP 2 — Setup Directories
# ------------------------------------------------------------
BASE_DIR = PROJECT_ROOT

RUS_FILE = BASE_DIR / "RUS" / "synthetic_RUS.json"
MEMORY_DIR = BASE_DIR / "memory"
REFLECTION_DIR = BASE_DIR / "reflections"
PLAN_DIR = BASE_DIR / "plans"

RESULTS_DIR = BASE_DIR / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESPONSES_PATH = RESULTS_DIR / f"ames_fiske_responses_{timestamp}.csv"
METRICS_PATH   = RESULTS_DIR / f"ames_fiske_metrics_{timestamp}.csv"

load_dotenv()   # load .env at project root
client = OpenAI()

# Managers
memory_manager     = MemoryManager(MEMORY_DIR)
reflection_manager = ReflectionManager(REFLECTION_DIR)
plan_manager       = PlanManager(PLAN_DIR)

# ------------------------------------------------------------
# STEP 3 — ID Mapping Helper (RU → Agent)
# ------------------------------------------------------------
def map_to_agent(ru_id: str) -> str:
    """RU_001 → Agent_0001 (always append)."""
    num = ru_id.split("_")[1]
    return f"Agent_{int(num):04d}"

# ------------------------------------------------------------
# STEP 4 — Load Reflector Units (RUs)
# ------------------------------------------------------------
with open(RUS_FILE, "r", encoding="utf-8") as f:
    rus_units = json.load(f)

# If wrapped in "RUs": [...]
if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

# Ensure index-based ru ids exist
for i, r in enumerate(rus_units, start=1):
    r.setdefault("RUs_id", f"RU_{i:04d}")

agents = rus_units
print(f"✅ Loaded {len(agents)} Reflector Units")

# ------------------------------------------------------------
# STEP 5 — Vignettes
# ------------------------------------------------------------
INTENTIONAL_VIGNETTE = """
Jake is a nursing-home worker who DELIBERATELY gave residents the wrong medicines
to see what would happen. Several residents became sick and needed medical care.
"""

UNINTENTIONAL_VIGNETTE = """
Jake is a nursing-home worker who ACCIDENTALLY mixed up the medicines.
He did not realize the mistake until residents became sick and needed medical care.
"""

# ------------------------------------------------------------
# STEP 6 — OpenAI Query
# ------------------------------------------------------------
def ask_openai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Respond ONLY with a number 1–7."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ API error:", e)
        return ""

# ------------------------------------------------------------
# STEP 7 — Ask Agent for Rating
# ------------------------------------------------------------
def ask_agent_rating(agent, vignette, condition):
    ru_id = agent["RUs_id"]
    agent_id = map_to_agent(ru_id)

    memory     = memory_manager.load(agent_id)
    reflection = reflection_manager.load(agent_id)
    plan       = plan_manager.load(agent_id)

    extras = []
    if memory:     extras.append(f"Memory: {memory}")
    if reflection: extras.append(f"Reflection: {reflection}")
    if plan:       extras.append(f"Plan: {plan}")
    extras_text = "\n".join(extras)

    prompt = f"""
You are {agent.get('persona', 'a human respondent')}.
Your prior OCEAN memories and reflections should inform your moral judgment.

Background Knowledge:
{extras_text}

Rate Jake’s blame from 1 (not blameworthy) to 7 (extremely blameworthy):

Scenario ({condition}):
{vignette}

Answer ONLY with a number 1–7.
"""

    text = ask_openai(prompt)
    match = re.search(r"\b([1-7])\b", text)

    if not match:
        print(f"⚠️ No valid rating from {agent_id}: {text!r}")
        return None

    rating = int(match.group(1))

    # -------------------------
    # Append Cognitive Data
    # -------------------------
    memory_manager.append(agent_id, {
        "q": f"Ames_Fiske_{condition}",
        "a": rating
    })

    reflection_manager.append(agent_id, {
        "insight": f"Rated Jake {rating} for {condition} harm",
        "task": "ames_fiske",
        "condition": condition
    })

    plan_manager.append(agent_id, {
        "next_action": f"Reflect on {condition} moral reasoning"
    })

    return rating

# ------------------------------------------------------------
# STEP 8 — Run Study
# ------------------------------------------------------------
total_agents = len(agents)
half = total_agents // 2

intentional_group = agents[:half]
unintentional_group = agents[half:]

results = []
print(f"\n🧠 Running Ames & Fiske on {total_agents} Agents...\n")

# Intentional
for idx, agent in enumerate(intentional_group, 1):
    rating = ask_agent_rating(agent, INTENTIONAL_VIGNETTE, "intentional")
    if rating is not None:
        results.append({"agent_id": map_to_agent(agent["RUs_id"]), "condition": "intentional", "rating": rating})

# Unintentional
for idx, agent in enumerate(unintentional_group, 1):
    rating = ask_agent_rating(agent, UNINTENTIONAL_VIGNETTE, "unintentional")
    if rating is not None:
        results.append({"agent_id": map_to_agent(agent["RUs_id"]), "condition": "unintentional", "rating": rating})

# ------------------------------------------------------------
# STEP 9 — Save Responses
# ------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(RESPONSES_PATH, index=False)

if df.empty:
    print("❌ No ratings collected. Exiting.")
    sys.exit()

# ------------------------------------------------------------
# STEP 10 — Compute Stats
# ------------------------------------------------------------
intent = df[df["condition"] == "intentional"]["rating"].to_numpy()
unintent = df[df["condition"] == "unintentional"]["rating"].to_numpy()

t_stat, p_val = stats.ttest_ind(intent, unintent, equal_var=True)

m1, m2 = intent.mean(), unintent.mean()
sd1, sd2 = intent.std(ddof=1), unintent.std(ddof=1)
n1,  n2 = len(intent), len(unintent)

sd_pooled = math.sqrt(((sd1 ** 2) + (sd2 ** 2)) / 2)
cohen_d   = (m1 - m2) / sd_pooled if sd_pooled != 0 else 0

metrics = {
    "intentional_mean": round(m1, 3),
    "intentional_sd": round(sd1, 3),
    "intentional_n": n1,
    "unintentional_mean": round(m2, 3),
    "unintentional_sd": round(sd2, 3),
    "unintentional_n": n2,
    "t_value": round(t_stat, 3),
    "p_value": round(p_val, 5),
    "cohens_d": round(abs(cohen_d), 3),
    "replication_success": "Yes" if p_val < 0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

# ------------------------------------------------------------
# STEP 11 — Print Summary
# ------------------------------------------------------------
print("\n📊 SUMMARY — Ames & Fiske (OpenAI Edition)")
print("------------------------------------------------")
print(f"Intentional Mean:   {m1:.2f} (SD={sd1:.2f}, n={n1})")
print(f"Unintentional Mean: {m2:.2f} (SD={sd2:.2f}, n={n2})")
print(f"t={t_stat:.3f}, p={p_val:.5f}, Cohen’s d={abs(cohen_d):.3f}")

print("\nReplication Result:",
      "✅ SUCCESS" if p_val < 0.05 else "❌ FAILED")

print(f"\n🧾 Metrics saved →   {METRICS_PATH}")
print(f"🗂️ Responses saved → {RESPONSES_PATH}\n")
