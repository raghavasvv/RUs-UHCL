"""
Ames & Fiske (2015) – API Edition
Appends short RU-specific memory entries ("intentional harm", "unintentional harm").
"""

import json, math, pandas as pd, time, sys, re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from scipy import stats

# ------------------------------------------------------------
# STEP 1: Setup
# ------------------------------------------------------------
load_dotenv()

# Base path
BASE_DIR = Path(r"C:\Users\SulegamaS6610\Downloads\ReflectorUnits-main1\ReflectorUnits-main1")

RUS_FILE = BASE_DIR / "RUS" / "synthetic_RUS.json"

# Pipeline imports
sys.path.append(str(BASE_DIR))
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

memory_manager     = MemoryManager(BASE_DIR / "memory")
reflection_manager = ReflectionManager(BASE_DIR / "reflections")
plan_manager       = PlanManager(BASE_DIR / "plans")

RESULTS_DIR = BASE_DIR / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESPONSES_PATH = RESULTS_DIR / f"ames_fiske_ru_responses_{timestamp}.csv"
METRICS_PATH   = RESULTS_DIR / f"ames_fiske_ru_metrics_{timestamp}.csv"

# ------------------------------------------------------------
# STEP 2: RU Helper (No Agent Mapping)
# ------------------------------------------------------------
def keep_ru_id(ru_id: str) -> str:
    """Keep RU_### ID unchanged."""
    return ru_id


# ------------------------------------------------------------
# STEP 3: API LLM Function
# ------------------------------------------------------------
from openai import OpenAI
client = OpenAI()

def query_api_llm(prompt: str, model="gpt-4o-mini"):
    """
    Query OpenAI GPT-4o-mini using API key.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        print(f"❌ API Error:", e)
        return ""


# ------------------------------------------------------------
# STEP 4: Load RUs
# ------------------------------------------------------------
with open(RUS_FILE, "r", encoding="utf-8") as f:
    rus_units = json.load(f)

if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

for i, r in enumerate(rus_units, start=1):
    r.setdefault("RUs_id", f"RU_{i:03d}")

RUs = rus_units

print(f"✅ Loaded {len(RUs)} RUs")


# ------------------------------------------------------------
# STEP 5: Vignettes
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
# STEP 6: Ask RU for Rating
# ------------------------------------------------------------
def ask_ru_rating(ru, vignette, condition):
    ru_id = keep_ru_id(ru["RUs_id"])

    # Load RU cognitive state
    memory = memory_manager.load(ru_id) or []
    reflection = reflection_manager.load(ru_id) or []
    plan = plan_manager.load(ru_id) or []

    extras = []
    if memory:     extras.append(f"Memory: {memory[-20:]}")
    if reflection: extras.append(f"Reflection: {reflection[-10:]}")
    if plan:       extras.append(f"Plan: {plan[-5:]}")
    extras_text = "\n".join(extras)

    # LLM prompt
    prompt = f"""
You are {ru.get('persona', 'a reflective RU')}.
Use your previous OCEAN-based memories, reflections, and plans to judge moral intent.

Background knowledge:
{extras_text}

Scenario ({condition}):
{vignette}

Rate how morally blameworthy Jake is from 1 (not at all) to 7 (extremely).
Answer ONLY with one integer (1–7).
""".strip()

    response = query_api_llm(prompt)
    response = response.replace("#","").replace("*","").replace("|","").strip()

    match = re.search(r"\b([1-7])\b", response)
    if match:
        rating = int(match.group(1))
    else:
        nums = re.findall(r"[1-7]", response)
        rating = int(nums[-1]) if nums else 4

    # Append short memory entry
    memory.append({"q": f"{condition} harm", "a": rating})

    with open(BASE_DIR / "memory" / f"{ru_id}.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    reflection_manager.append(ru_id, {"insight": f"Rated Jake {rating} for {condition} harm"})
    plan_manager.append(ru_id, {"next_action": f"Reflect further on {condition} harm"})

    return rating


# ------------------------------------------------------------
# STEP 7: Run Study
# ------------------------------------------------------------
total = len(RUs)
half = total // 2

intentional_RUs = RUs[:half]
unintentional_RUs = RUs[half:]

results = []

print(f"🧠 Running Ames & Fiske on {total} RUs...\n")

# Intentional harm group
for idx, ru in enumerate(intentional_RUs, 1):
    rating = ask_ru_rating(ru, INTENTIONAL_VIGNETTE, "intentional")
    results.append({"ru_id": ru["RUs_id"], "condition": "intentional", "rating": rating})
    if idx % 25 == 0:
        print(f"Progress: {idx}/{len(intentional_RUs)} (intentional)")
    time.sleep(0.3)

# Unintentional harm group
for idx, ru in enumerate(unintentional_RUs, 1):
    rating = ask_ru_rating(ru, UNINTENTIONAL_VIGNETTE, "unintentional")
    results.append({"ru_id": ru["RUs_id"], "condition": "unintentional", "rating": rating})
    if idx % 25 == 0:
        print(f"Progress: {idx}/{len(unintentional_RUs)} (unintentional)")
    time.sleep(0.3)


# ------------------------------------------------------------
# STEP 8: Save Responses
# ------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(RESPONSES_PATH, index=False, encoding="utf-8")


# ------------------------------------------------------------
# STEP 9: Statistics
# ------------------------------------------------------------
intentional = df[df["condition"]=="intentional"]["rating"].to_numpy()
unintentional = df[df["condition"]=="unintentional"]["rating"].to_numpy()

t_stat, p_val = stats.ttest_ind(intentional, unintentional, equal_var=True)

m1, m2 = intentional.mean(), unintentional.mean()
sd1, sd2 = intentional.std(ddof=1), unintentional.std(ddof=1)
n1, n2 = len(intentional), len(unintentional)

sd_pool = math.sqrt(((sd1**2)+(sd2**2))/2)
cohen_d = (m1 - m2) / sd_pool if sd_pool != 0 else 0

metrics = {
    "intentional_mean": round(m1,3),
    "intentional_sd": round(sd1,3),
    "intentional_n": n1,

    "unintentional_mean": round(m2,3),
    "unintentional_sd": round(sd2,3),
    "unintentional_n": n2,

    "t_value": round(t_stat, 3),
    "p_value": round(p_val, 5),
    "cohens_d": round(abs(cohen_d), 3),
    "replication_success": "Yes" if p_val < 0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)


# ------------------------------------------------------------
# STEP 10: Summary
# ------------------------------------------------------------
print("\n📊 SUMMARY (Ames & Fiske – RUs Only, API Edition)")
print(f"Intentional: mean={m1:.2f}, SD={sd1:.2f}, n={n1}")
print(f"Unintentional: mean={m2:.2f}, SD={sd2:.2f}, n={n2}")
print(f"t={t_stat:.3f}, p={p_val:.5f}, d={abs(cohen_d):.3f}")

if p_val < 0.05:
    print("✅ Replication success!")
else:
    print("❌ Replication failed.")

print(f"\n🧾 Metrics saved → {METRICS_PATH}")
print(f"🗂️ RU responses saved → {RESPONSES_PATH}")
