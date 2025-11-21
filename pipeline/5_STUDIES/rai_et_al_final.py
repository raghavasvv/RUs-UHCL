"""
Rai et al. (2017)
"""

import json, random, time, math, pandas as pd, sys, re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from openai import OpenAI

# ------------------------------------------------------------
# STEP 1 — AUTO-DETECT PROJECT ROOT
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parents[2]
sys.path.append(str(BASE_DIR))

print(f"🔍 Auto-detected BASE_DIR = {BASE_DIR}")

# Load managers
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

load_dotenv(BASE_DIR / ".env")
client = OpenAI()

# ------------------------------------------------------------
# STEP 2 — PATHS
# ------------------------------------------------------------
RUS_FILE = BASE_DIR / "RUS" / "synthetic_RUS.json"
RESULTS_DIR = BASE_DIR / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESPONSES_PATH = RESULTS_DIR / f"rai_responses_{timestamp}.csv"
METRICS_PATH   = RESULTS_DIR / f"rai_metrics_{timestamp}.csv"

memory_manager     = MemoryManager(BASE_DIR / "memory")
reflection_manager = ReflectionManager(BASE_DIR / "reflections")
plan_manager       = PlanManager(BASE_DIR / "plans")


# ------------------------------------------------------------
# STEP 3 — OpenAI GPT-4O-Mini Query
# ------------------------------------------------------------
def ask_openai(prompt):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Respond ONLY with a number 1–7."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=1.1
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return ""


# ------------------------------------------------------------
# STEP 4 — LOAD RUS UNITS
# ------------------------------------------------------------
with open(RUS_FILE, "r", encoding="utf-8") as f:
    rus_units = json.load(f)

if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

for i, r in enumerate(rus_units, start=1):
    r.setdefault("RUs_id", f"RU_{i:04d}")

agents = rus_units
print(f"✅ Loaded {len(agents)} RUS units")

# ------------------------------------------------------------
# STEP 4 — PROMPTS
# ------------------------------------------------------------
INSTRUMENTAL_HUMANIZED = """
John is kind, caring, and emotional.
You could harm him slightly to earn $500 for yourself.
Rate 1–7. Only return one number.
"""

INSTRUMENTAL_DEHUMANIZED = """
You are reading about a man described as cold, mechanical, and self-interested.
Imagine harming him slightly for $500.
Rate 1–7. Only return one number.
"""

MORAL_HUMANIZED = """
John is emotional but committed a moral crime (stole from charity).
Would you harm him to punish him?
Rate 1–7. Only return one number.
"""

MORAL_DEHUMANIZED = """
A cold, emotionless man stole from charity.
Would you harm him to punish him?
Rate 1–7. Only return one number.
"""

PROMPT_MAP = {
    ("Instrumental", "Humanized"): INSTRUMENTAL_HUMANIZED,
    ("Instrumental", "Dehumanized"): INSTRUMENTAL_DEHUMANIZED,
    ("Moral", "Humanized"): MORAL_HUMANIZED,
    ("Moral", "Dehumanized"): MORAL_DEHUMANIZED
}
# ------------------------------------------------------------
# STEP 6 — Mapping
# ------------------------------------------------------------
def map_to_ru(ru_id):
    num = ru_id.split("_")[1]
    return f"RUS_{int(num):04d}"


# ------------------------------------------------------------
# STEP 7 — ASK RATING
# ------------------------------------------------------------
def ask_ru_rating(agent, vtype, cond):
    ru_id = agent["RUs_id"]
    ru_file_id = map_to_ru(ru_id)

    # Load cognitive state
    memory = memory_manager.load(ru_file_id) or []
    reflection = reflection_manager.load(ru_file_id) or []
    plan = plan_manager.load(ru_file_id) or []

    # ONLY OCEAN ITEMS
    ocean = [
        m for m in memory
        if isinstance(m, dict) and "q" in m and m["q"][0] in ("O", "C", "E", "A", "N")
    ]

    ocean_text = "\n".join([f"{m['q']}: {m['a']}" for m in ocean[-50:]])
    vignette = PROMPT_MAP[(vtype, cond)]

    prompt = f"""
You are {agent.get('persona','a reflective respondent')}.
Use ONLY your personality (O,C,E,A,N) answers:

{ocean_text}

Scenario ({vtype} – {cond}):
{vignette}

Answer ONLY with a single number from 1–7.
""".strip()

    raw = ask_openai(prompt)
    cleaned = re.findall(r"[1-7]", raw)
    rating = int(cleaned[-1]) if cleaned else 4

    # Append to memory
    memory_manager.append(ru_file_id, {"q": f"{vtype}_{cond}", "a": rating})
    reflection_manager.append(ru_file_id, {"insight": f"Rated {vtype}-{cond}: {rating}"})
    plan_manager.append(ru_file_id, {"next_action": f"Reflect on {vtype}-{cond}"})

    return rating


# ------------------------------------------------------------
# STEP 8 — GROUPING + RUN
# ------------------------------------------------------------
random.shuffle(agents)
groups = [
    ("Instrumental", "Humanized"),
    ("Instrumental", "Dehumanized"),
    ("Moral", "Humanized"),
    ("Moral", "Dehumanized")
]

size = len(agents) // 4
records = []

print(f"🧠 Running Rai et al. on {len(agents)} RUS units...\n")

for i, (vtype, cond) in enumerate(groups):
    chunk = agents[i*size:(i+1)*size]

    for idx, agent in enumerate(chunk, 1):
        rating = ask_ru_rating(agent, vtype, cond)

        records.append({
            "ru_id": map_to_ru(agent["RUs_id"]),
            "violence_type": vtype,
            "condition": cond,
            "rating": rating
        })

        if idx % 25 == 0:
            print(f"Progress {vtype}-{cond}: {idx}/{size}")

        time.sleep(0.20)

df = pd.DataFrame(records)
df.to_csv(RESPONSES_PATH, index=False)
print(f"✅ Responses saved → {RESPONSES_PATH}")


# ------------------------------------------------------------
# STEP 9 — STATS (ANOVA + EFFECT SIZES)
# ------------------------------------------------------------
model = ols('rating ~ C(violence_type) * C(condition)', data=df).fit()
anova = sm.stats.anova_lm(model, typ=2)

interaction = anova.loc["C(violence_type):C(condition)"]
p_val = float(interaction["PR(>F)"])
eta = float(interaction["sum_sq"] / anova["sum_sq"].sum())

inst_h = df.query("violence_type=='Instrumental' & condition=='Humanized'")["rating"]
inst_d = df.query("violence_type=='Instrumental' & condition=='Dehumanized'")["rating"]

mean_diff = inst_d.mean() - inst_h.mean()
pooled_sd = math.sqrt((inst_d.var() + inst_h.var()) / 2)
d = round(mean_diff / pooled_sd, 3)

human = [3.2, 5.5, 3.3, 3.4]
rus_means = [
    inst_h.mean(),
    inst_d.mean(),
    df.query("violence_type=='Moral' & condition=='Humanized'")["rating"].mean(),
    df.query("violence_type=='Moral' & condition=='Dehumanized'")["rating"].mean()
]
r, _ = pearsonr(human, rus_means)


# ------------------------------------------------------------
# STEP 10 — SAVE METRICS
# ------------------------------------------------------------
pd.DataFrame([{
    "Inst_Hum_Mean": round(inst_h.mean(),2),
    "Inst_Deh_Mean": round(inst_d.mean(),2),
    "Moral_Hum_Mean": round(rus_means[2],2),
    "Moral_Deh_Mean": round(rus_means[3],2),
    "Cohen_d": d,
    "Eta_sq": round(eta,4),
    "p_val": round(p_val,5),
    "Pearson_r": round(r,3),
    "Replication": "Yes" if eta>0.005 and p_val<0.05 and mean_diff>0 else "No"
}]).to_csv(METRICS_PATH, index=False)

print(f"✅ Metrics saved → {METRICS_PATH}")


# ------------------------------------------------------------
# STEP 11 — SUMMARY
# ------------------------------------------------------------
print("\n📊 SUMMARY (Rai et al. 2017 – RUS Replication)")
print(f"Instrumental: {inst_h.mean():.2f} → {inst_d.mean():.2f}")
print(f"Moral:        {rus_means[2]:.2f} → {rus_means[3]:.2f}")
print(f"d={d}, η²={round(eta,4)}, p={round(p_val,5)}, r={round(r,3)}")

if eta>0.005 and p_val<0.05 and mean_diff>0:
    print("✅ Significant → Replication Successful")
else:
    print("❌ Not Significant → Replication Failed")

print("\n🎯 Rai et al. (2017) — Completed.\n")
