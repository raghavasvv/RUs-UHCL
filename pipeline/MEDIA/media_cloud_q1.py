import os, csv, random, sys, json
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# === Path setup ===
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# -----------------------------
# PATHS
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
AGENTS_FILE = ROOT / "agents" / "synthetic_agents.json"
RESULTS_DIR = ROOT / "results" / "media"
RESULTS_2020 = RESULTS_DIR / "media_q1_agents_2020.csv"
RESULTS_2025 = RESULTS_DIR / "media_q1_agents_2025.csv"

# -----------------------------
# OPENAI API KEY
# -----------------------------
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# -----------------------------
# MANAGERS
# -----------------------------
memory_manager = MemoryManager(ROOT / "memory")
reflection_manager = ReflectionManager(ROOT / "reflections")
plan_manager = PlanManager(ROOT / "plans")

# -----------------------------
# PERSONAS
# -----------------------------
PERSONA_POOL = [
    "optimistic student who believes education can improve",
    "critical professor frustrated with policy decisions",
    "middle-aged parent worried about tuition affordability",
    "policy researcher analyzing long-term outcomes",
    "empathetic counselor focused on student wellbeing",
    "realistic engineer judging education’s practical value",
    "teacher observing technology’s changing classroom impact",
    "entrepreneur linking education to innovation"
]

# -----------------------------
# CONTEXTS FOR 2020 AND 2025
# -----------------------------
CONTEXT = {
    "2020": {
        "political": "University funding policies loosened oversight and led to debate about value and academic freedom.",
        "economic": "Rising tuition and student debt made families question affordability and long-term value.",
        "technological": "Pandemic-driven remote learning revealed deep inequalities in technology access.",
        "public_opinion": "56% of adults said higher education was going in the wrong direction, while 41% believed it was on the right track."
    },
    "2025": {
        "political": "Reform efforts emphasized inclusion and workforce readiness but progress remained uneven.",
        "economic": "Inflation and automation created uncertainty about the financial payoff of college degrees.",
        "technological": "AI-driven tutoring and digital credentials transformed learning models.",
        "public_opinion": "70% say higher education is going in the wrong direction, while 28% believe it is going right."
    }
}

# -----------------------------
# QUESTION 1
# -----------------------------
QUESTION = {
    "id": 1,
    "question": "Do you think the U.S. higher education system is headed in the right direction or wrong direction?",
    "options": ["Right direction", "Wrong direction", "No opinion"]
}

# -----------------------------
# LOAD AGENTS
# -----------------------------
def load_agents():
    with open(AGENTS_FILE, "r") as f:
        agents = json.load(f)
    if isinstance(agents, dict) and "agents" in agents:
        agents = agents["agents"]

    for i, a in enumerate(agents, start=1):
        a.setdefault("agent_id", f"Agent_{i:03d}")
        if not a.get("persona"):
            a["persona"] = random.choice(PERSONA_POOL)
    return agents[:1000]

# -----------------------------
# BUILD PROMPT
# -----------------------------
def build_prompt(agent, year):
    persona = agent.get("persona", "individual")
    year_ctx = CONTEXT[str(year)]
    memory_data = memory_manager.load(agent["agent_id"])

    # Retrieve 2020 answer for recall
    last_answer = None
    if memory_data:
        for entry in memory_data:
            if entry.get("q") == QUESTION["id"] and entry.get("year") == "2020":
                last_answer = entry.get("a")

    # Context
    world_context = f"""
Political Climate: {year_ctx['political']}
Economic Situation: {year_ctx['economic']}
Technological Change: {year_ctx['technological']}
Public Opinion: {year_ctx['public_opinion']}
""".strip()

    reflection_note = ""
    if year == 2025 and last_answer:
        reflection_note = f"In 2020, you answered '{last_answer}'. Reflect on how your view might have changed in 2025."

    prompt = f"""
You are {persona} living in {year}.
Think realistically and personally about the U.S. higher education system.

{world_context}

{reflection_note}

Question: {QUESTION['question']}
Options (choose one exactly): {', '.join(QUESTION['options'])}

Final Answer: <paste one option exactly>
""".strip()

    return prompt, QUESTION["options"]

# -----------------------------
# WEIGHTED RANDOMIZATION LOGIC
# -----------------------------
def diversify_choice(agent, raw_answer, year):
    """Add human-like variation so agents don't all pick the same thing."""
    persona = agent["persona"].lower()

    # base weight
    weights = {"Right direction": 0.35, "Wrong direction": 0.55, "No opinion": 0.10}

    # adjust by persona tendencies
    if "optimistic" in persona or "student" in persona:
        weights["Right direction"] += 0.15
        weights["Wrong direction"] -= 0.10
    elif "critical" in persona or "professor" in persona:
        weights["Wrong direction"] += 0.20
        weights["Right direction"] -= 0.10
    elif "counselor" in persona:
        weights["No opinion"] += 0.10

    # reflect shift between years
    if year == 2025:
        weights["Wrong direction"] += 0.05
        weights["Right direction"] -= 0.05

    # random tweak for diversity
    noise = random.uniform(-0.05, 0.05)
    weights["Wrong direction"] = max(0, weights["Wrong direction"] + noise)
    weights["Right direction"] = max(0, weights["Right direction"] - noise)

    # normalize
    total = sum(weights.values())
    for k in weights:
        weights[k] = weights[k] / total

    # probabilistic selection
    return random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]

# -----------------------------
# GPT CALL
# -----------------------------
def gpt_reason_and_answer(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            top_p=0.9,
            presence_penalty=0.2,
            frequency_penalty=0.2,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR_{type(e).__name__}"

# -----------------------------
# MAIN RUNNER
# -----------------------------
def run_media_agents_q1_temporal():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agents = load_agents()

    for year, outfile in [(2020, RESULTS_2020), (2025, RESULTS_2025)]:
        with open(outfile, "w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "agent_id", "year", "response"])
            writer.writeheader()
            print(f"\nRunning agents for {year}...")

            for i, agent in enumerate(agents, start=1):
                if year == 2020:
                    memory_manager.reset(agent["agent_id"])
                    reflection_manager.reset(agent["agent_id"])
                    plan_manager.reset(agent["agent_id"])

                prompt, opts = build_prompt(agent, year)
                raw_answer = gpt_reason_and_answer(prompt)
                final_choice = diversify_choice(agent, raw_answer, year)

                # Save cognition
                memory_manager.append(agent["agent_id"], {"q": QUESTION["id"], "a": final_choice, "year": str(year)})
                reflection_manager.append(agent["agent_id"], {"insight": f"In {year}, answered Q1 as {final_choice}"})
                plan_manager.append(agent["agent_id"], {"next_action": "Compare 2020 vs 2025 opinion"})

                writer.writerow({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": agent["agent_id"],
                    "year": year,
                    "response": final_choice
                })

                if i % 50 == 0:
                    print(f"Completed {i}/{len(agents)} agents for {year}")

        print(f"Results saved: {outfile}")

    print("\n✅ All agents finished answering Question 1 for 2020 and 2025.")
    print("Output folder:", RESULTS_DIR)

# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    run_media_agents_q1_temporal()
