"""
Run_RUS_cloud.py – GPT-4o-mini Runner (CLEAN VERSION)
------------------------------------------------------------
• Choose number of RUs
• Choose number of questions
• CSV includes question_id, options, response_num, etc.
------------------------------------------------------------
"""

# =====================================================
# 1. Imports
# =====================================================
import json, os, time, statistics, csv, sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import contextlib

load_dotenv()

# =====================================================
# 2. Paths
# =====================================================
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

print("🔧 Project root:", ROOT)

from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager

# =====================================================
# 3. Config (YOU CONTROL THESE)
# =====================================================
NUM_RUs_TO_RUN = 5             # <---- Choose number of RUs
NUM_QUESTIONS_TO_USE = 5     # <---- Choose number of questions

RUS_FILE       = ROOT / "RUS" / "generated_RUS_20251120_105651.json"
QUESTION_FILE  = ROOT / "questions" / "Psychometrics.json"

RESULTS_DIR = ROOT / "results" / "OCEAN_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE_JSONL = RESULTS_DIR / f"ocean_responses_RUS{NUM_RUs_TO_RUN}.jsonl"
RESULTS_FILE_CSV   = RESULTS_DIR / f"ocean_responses_RUS{NUM_RUs_TO_RUN}.csv"
METRICS_FILE       = RESULTS_DIR / f"ocean_metrics_RUS{NUM_RUs_TO_RUN}.json"

# =====================================================
# 4. Managers
# =====================================================
memory_manager     = MemoryManager(ROOT / "memory")
reflection_manager = ReflectionManager(ROOT / "reflections")

# =====================================================
# 5. Load FIRST N RUs
# =====================================================
with open(RUS_FILE, "r") as f:
    rus_units = json.load(f)

if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

total_available = len(rus_units)
print(f"Loaded {total_available} total RUs")

for i, r in enumerate(rus_units, start=1):
    r.setdefault("RUs_id", f"RU_{i:04d}")

rus_units = rus_units[:NUM_RUs_TO_RUN]

print(f"Selected {len(rus_units)} RUs\n")

# =====================================================
# 6. Load questions
# =====================================================
with open(QUESTION_FILE, "r") as f:
    data = json.load(f)

questions = data["questions"] if "questions" in data else data
questions = questions[:NUM_QUESTIONS_TO_USE]   # <---- Limit questions

print(f"Loaded {len(questions)} questions\n")

# =====================================================
# 7. Likert normalization
# =====================================================
SCALE_MAP = {
    "very inaccurate": 1, "moderately inaccurate": 2,
    "neither accurate nor inaccurate": 3,
    "moderately accurate": 4, "very accurate": 5,
    "strongly disagree": 1, "disagree": 2,
    "neutral": 3, "agree": 4, "strongly agree": 5
}

def normalize_response(text):
    if not text: return None
    t = text.lower()
    for k, v in SCALE_MAP.items():
        if k in t:
            return v
    return None

# =====================================================
# 8. Build Prompt
# =====================================================
def build_prompt(ru, question):

    demo_parts = []
    for key in ["age", "gender", "race", "education", "location"]:
        if key in ru:
            demo_parts.append(f"{key}: {ru[key]}")
    demo = ", ".join(demo_parts) if demo_parts else "none"

    persona = f" Persona: {ru.get('persona')}" if ru.get("persona") else ""

    memory = memory_manager.load(ru["RUs_id"])
    memory_text = ""
    if memory:
        memory_text = "\nMemory:\n" + "\n".join(f"- {m}" for m in memory)

    reflection = reflection_manager.load(ru["RUs_id"])
    reflection_text = ""
    if reflection:
        reflection_text = "\nReflection:\n" + "\n".join(f"- {r['insight']}" for r in reflection)

    opts = ", ".join(question["options"])

    return f"""
You are a Reflector Unit (RU) simulating a human respondent.

Background:
Demographics: {demo}.{persona}
{memory_text}
{reflection_text}

Question: "{question['question']}"
Options: {opts}

Reply with ONLY one option exactly as written.
""".strip()

# =====================================================
# 9. GPT-4o-mini Call
# =====================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_model(ru, question):
    prompt = build_prompt(ru, question)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Reflector Unit (RU)."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=25,
            temperature=0.2
        )

        reply = resp.choices[0].message.content.strip().lower()

        input_tokens  = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens

        valid = [o.lower() for o in question["options"]]
        for opt in valid:
            if opt in reply:
                return question["options"][valid.index(opt)], input_tokens, output_tokens

        return "Invalid", input_tokens, output_tokens

    except Exception:
        return "ERROR_AuthenticationError", 0, 0

# =====================================================
# 10. RUNNER
# =====================================================
def run_rus():

    latencies = []
    total_calls = 0

    total_input_tokens  = 0
    total_output_tokens = 0

    start = time.perf_counter()

    with open(RESULTS_FILE_JSONL, "w") as fj, open(RESULTS_FILE_CSV, "w", newline="") as fc:

        writer = csv.DictWriter(fc, fieldnames=[
            "RUs_id",
            "question_id",
            "question",
            "options",
            "response",
            "response_num"
        ])
        writer.writeheader()

        for idx, ru in enumerate(rus_units, start=1):

            for q in questions:

                total_calls += 1
                t0 = time.perf_counter()

                resp, in_tok, out_tok = call_model(ru, q)

                latencies.append(time.perf_counter() - t0)

                total_input_tokens  += in_tok
                total_output_tokens += out_tok

                # ---- WRITE OUTPUT ----
                rec = {
                    "RUs_id": ru["RUs_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "options": "|".join(q["options"]),
                    "response": resp,
                    "response_num": normalize_response(resp)
                }
                fj.write(json.dumps(rec) + "\n")
                writer.writerow(rec)

                # ---- SILENT MEMORY + REFLECTION ----
                with contextlib.redirect_stdout(None):
                    if not resp.startswith("ERROR_"):
                        memory_manager.append(ru["RUs_id"], {"q": q["id"], "a": resp})
                        insight = f"You answered '{q['question']}' with '{resp}'."
                        reflection_manager.append(ru["RUs_id"], {"insight": insight})

            if idx % 100 == 0:
                print(f"✔ Completed {idx} RUs...")

    # metrics
    total_time = time.perf_counter() - start
    total_tokens = total_input_tokens + total_output_tokens

    metrics = {
        "RUs_run": NUM_RUs_TO_RUN,
        "questions_used": NUM_QUESTIONS_TO_USE,
        "total_responses": total_calls,
        "total_seconds": round(total_time, 2),
        "avg_latency": round(statistics.mean(latencies), 3),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens_used": total_tokens
    }

    with open(METRICS_FILE, "w") as mf:
        json.dump(metrics, mf, indent=2)

    print("\n===== RUN COMPLETE =====")
    print(metrics)
    print("========================")

# =====================================================
# 11. Entrypoint
# =====================================================
if __name__ == "__main__":
    print(f"🚀 Running {NUM_RUs_TO_RUN} RUs with {NUM_QUESTIONS_TO_USE} questions...\n")
    run_rus()
