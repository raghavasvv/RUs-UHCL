"""
run_rus_local.py
--------------------------------------------------------
This script executes Reflector Units (RUs) using a *local* LLM
(e.g., Ollama running phi3 model) instead of the OpenAI cloud.

Each RU simulates a human-like personality with:
    - Demographics
    - Memory (short-term context of previous answers)
    - Reflection (self-insight per answer)
    - Plan (future self-guided actions)

--------------------------------------------------------
⚙️ Configuration:
Change this variable to control how many RUs are randomly simulated:
    NUM_RUs_TO_RUN = 250   # 👈 (e.g., 100, 250, 500, 1000)
--------------------------------------------------------
All outputs are automatically prefixed with "local_" for clarity.
--------------------------------------------------------
"""

import json
import os
import time
import statistics
import csv
import requests
from datetime import datetime, UTC
from pathlib import Path
import sys
import random
from dotenv import load_dotenv

# =====================================================
# 1. Import project modules (managers)
# =====================================================
# These custom managers store cognitive state of each RU:
#   • MemoryManager – stores short-term history of Q/A
#   • ReflectionManager – logs introspective statements
#   • PlanManager – tracks next intended actions

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# =====================================================
# 2. File Path Configuration
# =====================================================
ROOT = Path(__file__).resolve().parent.parent
RUS_FILE = ROOT / "RUS" / "synthetic_RUS.json"             # Input file (1000 RUs)
QUESTION_FILE = ROOT / "questions" / "Psychometrics.json"  # OCEAN-style survey questions
RESULTS_DIR = ROOT / "results" / "local_llm_results"

# 🧩 Choose how many RUs to run
NUM_RUs_TO_RUN = 250  # 👈 Change to 100, 250, 500, 1000

# 🧾 Output file names include "local_" prefix and RU count
RESULTS_FILE_JSONL = RESULTS_DIR / f"local_responses_RUS{NUM_RUs_TO_RUN}.jsonl"
RESULTS_FILE_CSV   = RESULTS_DIR / f"local_responses_RUS{NUM_RUs_TO_RUN}.csv"
METRICS_FILE       = RESULTS_DIR / f"local_metrics_RUS{NUM_RUs_TO_RUN}.json"

# =====================================================
# 3. Initialize Managers
# =====================================================
memory_manager = MemoryManager(ROOT / "memory")
reflection_manager = ReflectionManager(ROOT / "reflections")
plan_manager = PlanManager(ROOT / "plans")

# =====================================================
# 4. Load RUs (Reflector Units)
# =====================================================
with open(RUS_FILE, "r") as f:
    rus_units = json.load(f)

# Handle both dict["RUs"] and direct list structure
if isinstance(rus_units, dict) and "RUs" in rus_units:
    rus_units = rus_units["RUs"]

# Assign IDs if missing
for i, r in enumerate(rus_units, start=1):
    r.setdefault("RUs_id", f"RU_{i:03d}")

print(f"✅ Loaded {len(rus_units)} total RUs")

# Random subset selection
rus_units = random.sample(rus_units, min(NUM_RUs_TO_RUN, len(rus_units)))
print(f"🎯 Randomly selected {len(rus_units)} RUs for local LLM run\n")

# =====================================================
# 5. Load Psychometric (OCEAN) Questions
# =====================================================
print("📥 Loading Psychometric questions...")
with open(QUESTION_FILE, "r") as f:
    question_data = json.load(f)

if isinstance(question_data, dict) and "questions" in question_data:
    questions = question_data["questions"]
else:
    questions = question_data

questions = questions[:50]  # Optional subset limit
print(f"✅ Loaded {len(questions)} questions\n")

# =====================================================
# 6. Scaling Map (Text → Numeric)
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
    "strongly agree": 5
}

def normalize_response(text: str) -> int | None:
    """Convert Likert-style text (e.g., 'Very Accurate') → numeric (1–5)."""
    if not text:
        return None
    lower = text.strip().lower()
    for key, val in SCALE_MAP.items():
        if key in lower:
            return val
    return None

# =====================================================
# 7. Prompt Builder
# =====================================================
def build_prompt(ru: dict, question: dict) -> str:
    """Builds contextual prompt for each RU including memory, reflection, and plan."""
    demo = ", ".join([f"{k}: {v}" for k, v in ru.get("demographics", {}).items()]) if ru.get("demographics") else ""
    persona = f" Persona: {ru.get('persona')}" if ru.get("persona") else ""

    memory = memory_manager.load(ru["RUs_id"])
    reflection = reflection_manager.load(ru["RUs_id"])
    plan = plan_manager.load(ru["RUs_id"])

    extras = []
    if memory: extras.append(f"Memory: {memory}")
    if reflection: extras.append(f"Reflection: {reflection}")
    if plan: extras.append(f"Plan: {plan}")
    extras_text = "\n".join(extras)

    return f"""
You are a Reflector Unit (RU) simulating a human survey respondent.

Background:
Demographics: {demo}.{persona}
{extras_text}

Question [{question['id']}]: {question['question']}
Options: {', '.join(question['options'])}

Think silently using Memory, Reflection, and Plan but DO NOT output them.
Answer with exactly one of the options only.
""".strip()

# =====================================================
# 8. Local LLM Call (Ollama API)
# =====================================================
def local_llm_response(ru: dict, question: dict, model: str = "phi3"):
    """Queries local Ollama model and returns one Likert-style response."""
    try:
        prompt = build_prompt(ru, question)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            stream=True,
            timeout=300
        )
        text = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                text += data.get("response", "")
            except json.JSONDecodeError:
                continue

        # Filter only valid Likert responses
        for opt in [
            "Very Inaccurate", "Moderately Inaccurate",
            "Neither Accurate Nor Inaccurate",
            "Moderately Accurate", "Very Accurate"
        ]:
            if opt.lower() in text.lower():
                return opt
        return ""
    except Exception as e:
        return f"ERROR_{type(e).__name__}"

# =====================================================
# 9. Main Runner
# =====================================================
def run_rus():
    """Executes the RU simulation using local LLM (phi3)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old cognitive data
    for folder in [ROOT / "memory", ROOT / "reflections", ROOT / "plans"]:
        for f in folder.glob("*.json"):
            f.unlink()

    total_calls, latencies = 0, []
    run_start = time.perf_counter()

    with open(RESULTS_FILE_JSONL, "w") as f_jsonl, open(RESULTS_FILE_CSV, "w", newline="") as f_csv:
        csv_writer = csv.DictWriter(
            f_csv,
            fieldnames=["timestamp", "RUs_id", "question_id", "question", "response", "response_num"]
        )
        csv_writer.writeheader()

        for ri, ru in enumerate(rus_units, 1):
            memory_manager.reset(ru["RUs_id"])
            reflection_manager.reset(ru["RUs_id"])
            plan_manager.reset(ru["RUs_id"])

            for qi, q in enumerate(questions, 1):
                total_calls += 1
                t0 = time.perf_counter()
                response = local_llm_response(ru, q)
                t1 = time.perf_counter()

                latency = t1 - t0
                latencies.append(latency)
                score = normalize_response(response)

                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "RUs_id": ru["RUs_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "response": response,
                    "response_num": score
                }

                f_jsonl.write(json.dumps(record) + "\n")
                f_jsonl.flush()
                csv_writer.writerow(record)
                f_csv.flush()

                memory_manager.append(ru["RUs_id"], {"q": q["id"], "a": response})
                reflection_manager.append(ru["RUs_id"], {"insight": f"Answered {q['id']} as {response}"})
                plan_manager.append(ru["RUs_id"], {"next_action": f"Reflect consistency for {q['id']}"})

            if ri % 10 == 0:
                print(f"Progress: {ri}/{len(rus_units)} RUs completed...", flush=True)

    run_end = time.perf_counter()
    total_seconds = max(1e-9, run_end - run_start)
    responses_per_sec = total_calls / total_seconds

    metrics = {
        "run": {
            "total_RUs": len(rus_units),
            "total_questions": len(questions),
            "expected_total": len(rus_units) * len(questions),
            "actual_total": total_calls,
        },
        "timing": {
            "total_seconds": total_seconds,
            "avg_latency": statistics.mean(latencies) if latencies else None,
            "median_latency": statistics.median(latencies) if latencies else None,
            "p95_latency": sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else None,
            "responses_per_second": responses_per_sec
        },
        "timestamp": datetime.now(UTC).isoformat()
    }

    with open(METRICS_FILE, "w") as mf:
        json.dump(metrics, mf, indent=2)

    # ---- Print Summary ----
    print("\n===== Local Run Summary =====")
    print(f"📂 Local Responses JSONL : {RESULTS_FILE_JSONL}")
    print(f"📂 Local Responses CSV   : {RESULTS_FILE_CSV}")
    print(f"📊 Local Metrics JSON    : {METRICS_FILE}")
    print(f"Total RUs          : {len(rus_units)}")
    print(f"Total Questions    : {len(questions)}")
    print(f"Expected total     : {len(rus_units) * len(questions)}")
    print(f"Actual total rows  : {total_calls}")
    print(f"Throughput (resp/s): {responses_per_sec:.2f}")
    print(f"Avg latency (s)    : {statistics.mean(latencies):.4f}")
    print(f"Median latency (s) : {statistics.median(latencies):.4f}")
    print(f"p95 latency (s)    : {metrics['timing']['p95_latency']:.4f}")
    print("==============================")
    print(f"✅ Finished run_rus successfully (Local LLM, {len(rus_units)}×{len(questions)})")

# =====================================================
# 10. Entrypoint
# =====================================================
if __name__ == "__main__":
    print(f"🚀 Starting Local LLM run (phi3) for {NUM_RUs_TO_RUN} randomly selected RUs...\n", flush=True)
    run_rus()
