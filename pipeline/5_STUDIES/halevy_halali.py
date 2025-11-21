import json, math, time, pandas as pd, random, re
from pathlib import Path
import sys
from scipy.stats import chi2_contingency, fisher_exact
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------
# 1. AUTODETECT PROJECT ROOT
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
for parent in CURRENT_FILE.parents:
    if (parent / "pipeline").is_dir() and (parent / "RUS").is_dir():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError("❌ Could not detect project root")

sys.path.append(str(PROJECT_ROOT))

# Managers
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

MEMORY_DIR     = PROJECT_ROOT / "memory"
REFLECTION_DIR = PROJECT_ROOT / "reflections"
PLAN_DIR       = PROJECT_ROOT / "plans"

memory_manager     = MemoryManager(MEMORY_DIR)
reflection_manager = ReflectionManager(REFLECTION_DIR)
plan_manager       = PlanManager(PLAN_DIR)

load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI()

# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
RUS_PATH    = PROJECT_ROOT / "RUS" / "synthetic_RUS.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
RESPONSES_PATH = RESULTS_DIR / f"halevy_halali_responses_{timestamp}.csv"
METRICS_PATH   = RESULTS_DIR / f"halevy_halali_metrics_{timestamp}.csv"

# ------------------------------------------------------------
# 3. Load RUS Units
# ------------------------------------------------------------
with open(RUS_PATH, "r", encoding="utf-8") as f:
    rus_units = json.load(f)

for i, r in enumerate(rus_units, start=1):
    r["RUs_id"] = f"RU_{i:04d}"

total_rus = len(rus_units)
print(f"Loaded {total_rus} RUS units")


# ------------------------------------------------------------
# 4. EXACT PROMPTS 
# ------------------------------------------------------------
DISPUTE_SCENARIO = """
You are assigned the role of either RED or BLUE.
You are in conflict about sharing valuable resources.

You must choose between:
△ TRIANGLE – cooperate fairly with the other person.
▢ SQUARE – compete and try to win more for yourself.

If both you and the other choose △ (triangle), you each get $3.
If both choose ▢ (square), you each get $1.
If one chooses △ and the other chooses ▢:
• The one choosing △ gets $2.
• The one choosing ▢ gets $4.

Remember: cooperating benefits both, but competing may give you a higher personal reward if the other cooperates.
However, if both compete, both of you lose heavily.
"""

THIRD_PARTY_SCENARIO = """
You are assigned the role of GREEN — a third party observing a conflict between two people, RED and BLUE.

You must choose between:
O – Do not intervene (just observe)
I – Intervene and try to make peace

If you choose O, you will get $2 regardless of what RED and BLUE do.
If you choose I, your payoff depends on their choices:
• both △ → all get $4
• both ▢ → all get $0
• one △ and one ▢ → RED & BLUE get $3 each, you get $2.

Intervening requires effort but might increase cooperation between RED and BLUE.
"""
# ------------------------------------------------------------
# 5. OpenAI Query
# ------------------------------------------------------------
def ask_openai(prompt):
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Respond ONLY with the choice number."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.9
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print("❌ OpenAI error:", e)
        return ""

# ------------------------------------------------------------
# 6. Extract ONLY OCEAN memory
# ------------------------------------------------------------
def get_ocean_memory(ru_file_id):
    mem = memory_manager.load(ru_file_id) or []
    return [
        m for m in mem
        if isinstance(m, dict)
        and "q" in m
        and m["q"][0] in ("O", "C", "E", "A", "N")
    ]

# ------------------------------------------------------------
# 7. Third Party (GREEN)
# ------------------------------------------------------------
def third_party_decision(ru):
    ru_id = ru["RUs_id"]
    ru_file_id = f"RUS_{ru_id.split('_')[1]}"

    ocean = get_ocean_memory(ru_file_id)
    ocean_text = "\n".join(f"{m['q']}: {m['a']}" for m in ocean[-50:])

    prompt = f"""
You are {ru.get('persona','a participant')}.

Use ONLY your OCEAN answers:
{ocean_text}

{THIRD_PARTY_SCENARIO}

Choose:
1 = I (Intervene)
2 = O (Observe)
"""

    ans = ask_openai(prompt)
    choice = "I" if ans.startswith("1") else "O"

    memory_manager.append(ru_file_id, {"q": "Halevy_GREEN", "a": choice})
    reflection_manager.append(ru_file_id, {"insight": f"Chose {choice} as GREEN"})
    plan_manager.append(ru_file_id, {"next_action": "Reflect on intervention"})

    return choice

# ------------------------------------------------------------
# 8. RED / BLUE Decisions
# ------------------------------------------------------------
def disputant_decision(ru, condition, role):
    ru_id = ru["RUs_id"]
    ru_file_id = f"RUS_{ru_id.split('_')[1]}"

    ocean = get_ocean_memory(ru_file_id)
    ocean_text = "\n".join(f"{m['q']}: {m['a']}" for m in ocean[-50:])

    prompt = f"""
You are {ru.get('persona','a participant')}.

Use ONLY your OCEAN answers:
{ocean_text}

{DISPUTE_SCENARIO}

Condition: {condition}

Choose:
1 = △ (Cooperate)
2 = ▢ (Compete)
"""

    ans = ask_openai(prompt)
    choice = "Triangle" if ans.startswith("1") else "Square"

    memory_manager.append(ru_file_id, {"q": f"Halevy_{role}", "a": choice})
    reflection_manager.append(ru_file_id, {"insight": f"{role} → {choice}"})
    plan_manager.append(ru_file_id, {"next_action": "Reflect on cooperation"})

    return choice

groups = [
    rus_units[i:i+3]
    for i in range(0, total_rus, 3)
    if len(rus_units[i:i+3]) == 3
]

# ------------------------------------------------------------
# 9. Run Simulation
# ------------------------------------------------------------
results = []
processed = 0

print("\n🧠 Running Halevy & Halali...\n")

for g in groups:
    RED, BLUE, GREEN = g

    green_choice  = third_party_decision(GREEN)
    condition     = "with_intervention" if green_choice == "I" else "without_intervention"

    red_choice    = disputant_decision(RED,  condition, "RED")
    blue_choice   = disputant_decision(BLUE, condition, "BLUE")

    results.append({
        "RED": RED["RUs_id"],
        "BLUE": BLUE["RUs_id"],
        "GREEN": GREEN["RUs_id"],
        "GREEN_choice": green_choice,
        "condition": condition,
        "RED_choice": red_choice,
        "BLUE_choice": blue_choice
    })

    processed += 3
    print(f"Progress: {processed}/{total_rus}")

    time.sleep(0.15)

df = pd.DataFrame(results)
df.to_csv(RESPONSES_PATH, index=False)
print(f"\nResponses saved → {RESPONSES_PATH}")

# ------------------------------------------------------------
# 10. Metrics
# ------------------------------------------------------------
with_int  = df[df["condition"] == "with_intervention"]
without_i = df[df["condition"] == "without_intervention"]

coop_with    = ((with_int.RED_choice=="Triangle") & (with_int.BLUE_choice=="Triangle")).sum()
coop_without = ((without_i.RED_choice=="Triangle") & (without_i.BLUE_choice=="Triangle")).sum()

len_with     = len(with_int)
len_without  = len(without_i)

rate_with    = coop_with/len_with if len_with else 0
rate_without = coop_without/len_without if len_without else 0

table = [
    [coop_with, len_with-coop_with],
    [coop_without, len_without-coop_without]
]

try:
    chi2, p, _, _ = chi2_contingency(table)
except:
    _, p = fisher_exact(table)
    chi2 = 0

h = 2 * abs(
    math.asin(math.sqrt(rate_with)) -
    math.asin(math.sqrt(rate_without))
)

metrics = {
    "groups_total": len(df),
    "intervention_rate_%": round(df.GREEN_choice.eq("I").mean()*100,1),
    "coop_with_intervention_%": round(rate_with*100,1),
    "coop_without_intervention_%": round(rate_without*100,1),
    "chi_square": round(chi2,3),
    "p_value": round(p,5),
    "cohens_h": round(h,3),
    "replication_success": "Yes" if p < 0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

print(f"Metrics saved → {METRICS_PATH}")
print("\n🎯 Halevy & Halali (2015) – Completed.\n")
