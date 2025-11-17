"""
run_RUS_LLM.py
--------------------------------------------------------
Runs Reflector Units (RUs) locally using an Ollama model
(llama3 by default). This replaces the OpenAI API version.

This script:
    • Loads existing Reflector Units (synthetic_RUS.json)
    • Loads Psychometrics questions
    • Builds contextual prompts using Memory / Reflection / Plan
    • Sends prompts to Ollama (local llama3)
    • Extracts a valid Likert-style response
    • Stores responses in JSONL + CSV
    • Appends Memory, Reflection, Plan persistently

--------------------------------------------------------
"""

# =====================================================
# 1. Imports and dynamic path setup
# =====================================================
import json
import os
import sys
import time
import csv
import random
import statistics
import requests
from pathlib import Path
from datetime import datetime, UTC

# Dynamically detect project root no matter how folder is named
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]        # LOCAL_LLM → pipeline → PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# =====================================================
# 2. File paths
# =====================================================
RUS_FILE = PROJECT_ROOT / "RUS" / "synthetic_RUS.json"
QUESTION_FILE = PROJECT_ROOT / "questions" / "Psychometrics.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "local_llm_results"

NUM_RUs_TO_RUN = 1000  # professor can change this anytime

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSONL = RESULTS_DIR / f"local_responses_RUS{NUM_RUs_TO_RUN}.jsonl"
RESULTS_CSV   = RESULTS_DIR / f"local_responses_RUS{NUM_RUs_TO_RUN}.csv"
METRICS_FILE  = RESULTS_DIR / f"local_metrics_RUS{NUM_RUs_TO_RUN}.json"

# =====================================================
# 3. Cognitive managers (persistent)
# =====================================================
memory_manager = MemoryManager(PROJECT_ROOT / "memory")
reflection_manager = ReflectionManager(PROJECT_ROOT / "reflections")
plan_manager = PlanManager(PROJECT_ROOT / "plans")

# =====================================================
# 4. Load RUs
# =====================================================
if not RUS_FILE.exists():
    raise FileNotFoundError(f"RUS file not found: {RUS_FILE}")

with open(RUS_FILE, "r") as f:
    rus_units = json.load(f)

if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

for i, ru in enumerate(rus_units, start=1):
    ru.setdefault("RUs_id", f"RU_{i:03d}")

rus_units = random.sample(rus_units, min(NUM_RUs_TO_RUN, len(rus_units)))

print(f"Loaded {len(rus_units)} RUs")

# Ensure RU cognitive files exist
for ru in rus_units:
    (PROJECT_ROOT / "memory"      / f"{ru['RUs_id']}.json").touch(exist_ok=True)
    (PROJECT_ROOT / "reflections" / f"{ru['RUs_id']}.json").touch(exist_ok=True)
    (PROJECT_ROOT / "plans"       / f"{ru['RUs_id']}.json").touch(exist_ok=True)

# =====================================================
# 5. Load questions
# =====================================================
if not QUESTION_FILE.exists():
    raise FileNotFoundError(f"Question file not found: {QUESTION_FILE}")

with open(QUESTION_FILE, "r") as f:
    qdata = json.load(f)

questions = qdata["questions"] if isinstance(qdata, dict) and "questions" in qdata else qdata
questions = questions[:50]

print(f"Loaded {len(questions)} questions")

# =====================================================
# 6. Likert mapping
# =====================================================
SCALE_MAP = {
    "very inaccurate": 1,
    "moderately inaccurate": 2,
    "neither accurate nor inaccurate": 3,
    "moderately accurate": 4,
    "very accurate": 5,
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "agree": 4,
    "strongly agree": 5,
}

def normalize_response(text):
    if not text:
        return None
    lower = text.lower()
    for k, v in SCALE_MAP.items():
        if k in lower:
            return v
    return None

# =====================================================
# 7. Prompt Builder
# =====================================================
def build_prompt(ru, q):
    demo = ", ".join(f"{k}: {v}" for k, v in ru.get("demographics", {}).items()) if ru.get("demographics") else ""
    persona = ru.get("persona", "")

    memory = memory_manager.load(ru["RUs_id"])
    reflection = reflection_manager.load(ru["RUs_id"])
    plan = plan_manager.load(ru["RUs_id"])

    ctx = []
    if memory: ctx.append(f"Memory: {memory}")
    if reflection: ctx.append(f"Reflection: {reflection}")
    if plan: ctx.append(f"Plan: {plan}")

    return f"""
You are a Reflector Unit (RU) simulating a human participant.

Demographics: {demo}
Persona: {persona}
{os.linesep.join(ctx)}

Question [{q['id']}]: {q['question']}

Options: {', '.join(q['options'])}

RULES:
1. Choose exactly ONE option from the list.
2. Copy it EXACTLY as written.
3. Do NOT include explanations or extra text.

Output only the option.
""".strip()

# =====================================================
# 8. LLM extraction helper
# =====================================================
def extract_option(text, options):
    # First attempt: exact match
    for opt in options:
        if opt.lower() in text.lower():
            return opt

    # Fallback: match first keyword
    words = text.lower().split()
    for opt in options:
        key = opt.lower().split()[0]
        if key in words:
            return opt

    return ""

# =====================================================
# 9. Ollama local LLM request
# =====================================================
def local_llm_response(ru, q, model="llama3"):
    try:
        prompt = build_prompt(ru, q)

        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            stream=True,
            timeout=200,
        )

        text = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode())
                text += data.get("response", "")
            except:
                continue

        return extract_option(text, q["options"])

    except Exception as e:
        return f"ERROR_{type(e).__name__}"

# =====================================================
# 10. Run RU pipeline
# =====================================================
def run_rus():
    print("Running Reflector Units using local llama3...")

    total_calls = 0
    latencies = []
    start = time.perf_counter()

    with open(RESULTS_JSONL, "w") as fj, open(RESULTS_CSV, "w", newline="") as fc:
        writer = csv.DictWriter(fc, fieldnames=[
            "timestamp","RUs_id","question_id","question","response","response_num"
        ])
        writer.writeheader()

        for ru in rus_units:
            for q in questions:

                t0 = time.perf_counter()
                resp = local_llm_response(ru, q)
                t1 = time.perf_counter()

                total_calls += 1
                latencies.append(t1 - t0)

                score = normalize_response(resp)

                rec = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "RUs_id": ru["RUs_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "response": resp,
                    "response_num": score
                }

                fj.write(json.dumps(rec) + "\n")
                writer.writerow(rec)

                memory_manager.append(ru["RUs_id"], {"q": q["id"], "a": resp})
                reflection_manager.append(ru["RUs_id"], {"insight": f"Answered {q['id']} as {resp}"})
                plan_manager.append(ru["RUs_id"], {"next_action": f"Reflect on {q['id']}"})

    total_time = max(time.perf_counter() - start, 1e-9)

    metrics = {
        "RUs": len(rus_units),
        "questions": len(questions),
        "responses": total_calls,
        "avg_latency": statistics.mean(latencies),
        "responses_per_second": total_calls / total_time,
    }

    with open(METRICS_FILE, "w") as mf:
        json.dump(metrics, mf, indent=2)

    print("Run complete.")

# =====================================================
# 11. Entrypoint
# =====================================================
if __name__ == "__main__":
    run_rus()

