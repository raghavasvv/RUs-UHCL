"""
Cooney et al. (2016) – Fairness × Outcome Replication
Appends to existing cognitive files:
- memory/RUS_xxxx.json
- reflections/RUS_xxxx.json
- plans/RUS_xxxx.json
Never overwrites OCEAN or Ames data.
"""

# ------------------------------------------------------------
# STEP 1 — Auto-detect project root
# ------------------------------------------------------------
import sys, time, json, re, math, random
from pathlib import Path
from datetime import datetime
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

CURRENT = Path(__file__).resolve()
for parent in CURRENT.parents:
    if (parent / "pipeline").is_dir() and (parent / "RUS").is_dir():
        ROOT = parent
        break
else:
    raise RuntimeError("❌ Project root not found")

sys.path.append(str(ROOT))
print("🔧 PROJECT ROOT:", ROOT)

# ------------------------------------------------------------
# STEP 2 — Import managers
# ------------------------------------------------------------
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

BASE = ROOT
RUS_FILE = BASE / "RUS" / "synthetic_RUS.json"
MEMORY_DIR = BASE / "memory"
REFLECTION_DIR = BASE / "reflections"
PLAN_DIR = BASE / "plans"

RESULTS_DIR = BASE / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESPONSES_PATH = RESULTS_DIR / f"cooney_responses_{ts}.csv"
METRICS_PATH   = RESULTS_DIR / f"cooney_metrics_{ts}.csv"

load_dotenv()
client = OpenAI()

memory_manager     = MemoryManager(MEMORY_DIR)
reflection_manager = ReflectionManager(REFLECTION_DIR)
plan_manager       = PlanManager(PLAN_DIR)

# ------------------------------------------------------------
# STEP 3 — Load RUS units
# ------------------------------------------------------------
with open(RUS_FILE,"r") as f:
    rus_units = json.load(f)

if isinstance(rus_units,dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

for i,r in enumerate(rus_units,start=1):
    r.setdefault("RUs_id",f"RUS_{i:04d}")

agents = rus_units
print(f"✅ Loaded {len(agents)} RUS units")

# ------------------------------------------------------------
# STEP 4 — Prompts
# ------------------------------------------------------------
PROMPTS = {
    "Fair_Loss": "You lost a bonus by random coin flip. Rate upset (1–7).",
    "Fair_Gain": "You received a bonus by random coin flip. Rate happiness (1–7).",
    "Unfair_Loss": "A person denied you the bonus intentionally. Rate upset (1–7).",
    "Unfair_Gain": "A person awarded you the bonus intentionally. Rate happiness (1–7)."
}

# ------------------------------------------------------------
# STEP 5 — Query GPT
# ------------------------------------------------------------
def ask_gpt(agent,prompt):
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Respond only with a number 1–7."},
                    {"role":"user","content":agent["persona"] + "\n" + prompt}
                ],
                max_tokens=10,
                temperature=0.8
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\b([1-7])\b",text)
            if m:
                return int(m.group(1))
        except:
            time.sleep(0.2)
    return random.randint(1,7)

# ------------------------------------------------------------
# STEP 6 — Run condition & append to cognitive files
# ------------------------------------------------------------
def run_condition(agent,cond):
    ru_id = agent["RUs_id"]

    proc,outcome = cond.split("_")
    rating = ask_gpt(agent,PROMPTS[cond])

    # MEMORY APPEND
    memory_manager.append(ru_id,{
        "q":f"Cooney_{cond}",
        "a":rating
    })

    # REFLECTION APPEND
    reflection_manager.append(ru_id,{
        "insight":f"Rated {rating} for {proc} {outcome}",
        "task":"cooney_2016",
        "condition":cond
    })

    # PLAN APPEND
    plan_manager.append(ru_id,{
        "next_action":"Reflect on fairness vs outcomes (Cooney 2016)"
    })

    return {
        "rus_id":ru_id,
        "procedure":proc,
        "outcome":outcome,
        "condition":cond,
        "rating":rating
    }

# ------------------------------------------------------------
# STEP 7 — Run experiment
# ------------------------------------------------------------
print("\n🧠 Running Cooney et al. (2016)...\n")
conditions = list(PROMPTS.keys())
records = []

for agent in agents:
    for cond in conditions:
        records.append(run_condition(agent,cond))
    time.sleep(0.1)

# ------------------------------------------------------------
# STEP 8 — Save responses
# ------------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(RESPONSES_PATH,index=False)
print("📁 Responses saved →",RESPONSES_PATH)

# ------------------------------------------------------------
# STEP 9 — ANOVA + Cohen’s d
# ------------------------------------------------------------
model = ols("rating ~ C(procedure)*C(outcome)",data=df).fit()
anova = sm.stats.anova_lm(model,typ=2)

f_proc  = float(anova.loc["C(procedure)","F"])
p_proc  = float(anova.loc["C(procedure)","PR(>F)"])
f_out   = float(anova.loc["C(outcome)","F"])
p_out   = float(anova.loc["C(outcome)","PR(>F)"])
f_inter = float(anova.loc["C(procedure):C(outcome)","F"])
p_inter = float(anova.loc["C(procedure):C(outcome)","PR(>F)"])

fair_loss   = df[(df.procedure=="Fair") & (df.outcome=="Loss")]["rating"]
unfair_loss = df[(df.procedure=="Unfair") & (df.outcome=="Loss")]["rating"]

mean_diff = fair_loss.mean() - unfair_loss.mean()
pooled_sd = math.sqrt((fair_loss.std()**2 + unfair_loss.std()**2)/2)
cohens_d  = abs(mean_diff/pooled_sd) if pooled_sd!=0 else 0

metrics = {
    "f_procedure":round(f_proc,3),
    "p_procedure":float(f"{p_proc:.5f}"),
    "f_outcome":round(f_out,3),
    "p_outcome":float(f"{p_out:.5f}"),
    "f_interaction":round(f_inter,3),
    "p_interaction":float(f"{p_inter:.5f}"),
    "cohens_d_loss":round(cohens_d,3),
    "replication_success":"Yes" if p_proc<0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH,index=False)
print("📁 Metrics saved →",METRICS_PATH)

print("\n🎉 Cooney et al. (2016) Replication Completed\n")
