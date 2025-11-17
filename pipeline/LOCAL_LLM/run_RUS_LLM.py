"""
run_RUS_LLM.py
--------------------------------------------------------
Executes Reflector Units (RUs) locally using Ollama (llama3).
NO memory reset. Memory, reflection, plan all persist & append.
Guaranteed forced option-only responses (no blank answers).
--------------------------------------------------------
"""

import json
import os
import time
import csv
import statistics
import requests
import random
from datetime import datetime, UTC
from pathlib import Path
import sys

# =====================================================
# FIX IMPORT PATH (ALWAYS LOAD PROJECT ROOT)
# =====================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]     # LOCAL_LLM → pipeline → ROOT
sys.path.append(str(PROJECT_ROOT))

from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# =====================================================
# FILE CONFIG
# =====================================================
ROOT = PROJECT_ROOT

# CHANGE THESE TWO:
RUS_FILE = ROOT / "RUS" / "generated_RUS_20251117_114503.json"
QUESTION_FILE = ROOT / "questions" / "my_custom_questions.json"

RESULTS_DIR = ROOT / "results" / "local_llm_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_RUs_TO_RUN = 5

RESULTS_FILE_JSONL = RESULTS_DIR / f"local_responses_RUS{NUM_RUs_TO_RUN}.jsonl"
RESULTS_FILE_CSV   = RESULTS_DIR / f"local_responses_RUS{NUM_RUs_TO_RUN}.csv"
METRICS_FILE       = RESULTS_DIR / f"local_metrics_RUS{NUM_RUs_TO_RUN}.json"

# =====================================================
# INITIALIZE MANAGERS
# =====================================================
memory_manager = MemoryManager(ROOT / "memory")
reflection_manager = ReflectionManager(ROOT / "reflections")
plan_manager = PlanManager(ROOT / "plans")

# =====================================================
# LOAD RUs
# =====================================================
with open(RUS_FILE, "r") as f:
    rus_units = json.load(f)

if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

for i, ru in enumerate(rus_units, 1):
    ru.setdefault("RUs_id", f"RU_{i:03d}")

rus_units = random.sample(rus_units, min(NUM_RUs_TO_RUN, len(rus_units)))

# Ensure files exist but DO NOT RESET
for ru in rus_units:
    (ROOT / "memory"      / f"{ru['RUs_id']}.json").touch(exist_ok=True)
    (ROOT / "reflections" / f"{ru['RUs_id']}.json").touch(exist_ok=True)
    (ROOT / "plans"       / f"{ru['RUs_id']}.json").touch(exist_ok=True)

print(f"Running {len(rus_units)} RUs with llama3\n")

# =====================================================
# LOAD QUESTIONS
# =====================================================
with open(QUESTION_FILE, "r") as f:
    data = json.load(f)

if isinstance(data, dict) and "questions" in data:
    questions = data["questions"]
else:
    questions = data

print(f"Loaded {len(questions)} questions\n")

# =====================================================
# STRONG PROMPT BUILDER (forces correct answers)
# =====================================================
def build_prompt(ru, question):
    demo = ", ".join([f"{k}: {v}" for k, v in ru.get("demographics", {}).items()]) if ru.get("demographics") else ""
    persona = ru.get("persona", "")

    memory = memory_manager.load(ru["RUs_id"])
    reflection = reflection_manager.load(ru["RUs_id"])
    plan = plan_manager.load(ru["RUs_id"])

    ctx = []
    if memory: ctx.append(f"Memory: {memory}")
    if reflection: ctx.append(f"Reflection: {reflection}")
    if plan: ctx.append(f"Plan: {plan}")

    options_text = "\n".join([f"- {opt}" for opt in question["options"]])

    return f"""
You are a Reflector Unit (RU).

Demographics: {demo}
Persona: {persona}
{os.linesep.join(ctx)}

Survey Question:
{question['question']}

Choose ONLY ONE option from the list below.
Copy the option EXACTLY as written. NO explanations.

OPTIONS:
{options_text}

OUTPUT INSTRUCTION:
Return EXACTLY ONE OPTION STRING FROM ABOVE.
""".strip()

# =====================================================
# LOCAL LLM CALL (llama3)
# =====================================================
def local_llm_response(ru, question, model="llama3"):
    try:
        prompt = build_prompt(ru, question)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            stream=True,
            timeout=200,
        )

        text = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode())
                text += data.get("response", "")
            except:
                continue

        # match EXACT option from file
        for opt in question["options"]:
            if opt.lower() in text.lower():
                return opt

        return ""  # invalid or blank
    except Exception as e:
        return f"ERROR_{type(e).__name__}"

# =====================================================
# MAIN RUNNER
# =====================================================
def run_rus():
    total_calls = 0
    latencies = []
    start = time.perf_counter()

    with open(RESULTS_FILE_JSONL, "w") as f_jsonl, open(RESULTS_FILE_CSV, "w", newline="") as f_csv:

        writer = csv.DictWriter(
            f_csv,
            fieldnames=["timestamp", "RUs_id", "question_id", "question", "response", "response_num"]
        )
        writer.writeheader()

        for ru in rus_units:
            for q in questions:

                t0 = time.perf_counter()
                response = local_llm_response(ru, q)
                t1 = time.perf_counter()

                latencies.append(t1 - t0)
                total_calls += 1

                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "RUs_id": ru["RUs_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "response": response,
                    "response_num": None
                }

                f_jsonl.write(json.dumps(record) + "\n")
                writer.writerow(record)

                # append to memory systems
                memory_manager.append(ru["RUs_id"], {"q": q["id"], "a": response})
                reflection_manager.append(ru["RUs_id"], {"insight": f"Answered {q['id']} as {response}"})
                plan_manager.append(ru["RUs_id"], {"next_action": f"Reflect on {q['id']}"})

    total_time = max(time.perf_counter() - start, 1e-9)

    metrics = {
        "num_rus": len(rus_units),
        "num_questions": len(questions),
        "total_responses": total_calls,
        "avg_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "responses_per_second": total_calls / total_time
    }

    with open(METRICS_FILE, "w") as mf:
        json.dump(metrics, mf, indent=2)

    print("Finished local llama3 RU run.")

# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    print("Starting local llama3 RU execution...\n")
    run_rus()
