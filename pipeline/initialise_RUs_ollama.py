import json
import random
import subprocess
import time
from pathlib import Path
from datetime import datetime

# =======================================================
# Configuration
# =======================================================
NUM_AGENTS = 50
MODEL = "llama3"   # Ensure this model is pulled in Ollama
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / ".." / "RUS"

# Create safe filename with timestamp so nothing is overwritten
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"generated_RUS_{timestamp}.json"

GENDERS = ["male", "female"]

# =======================================================
# Persona Prompt
# =======================================================
BASE_PROMPT = """
Create one short human persona in under 25 words.

Include:
1. Occupation or role
2. 2–3 personality traits
3. Small daily-life detail

Return only ONE line. Do NOT number or use quotes.
"""

# =======================================================
# Function to call Ollama
# =======================================================
def ollama_generate(prompt):
    process = subprocess.Popen(
        ["ollama", "run", MODEL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    output, error = process.communicate(prompt)

    if error and error.strip():
        print("Ollama error:", error)

    return output.strip()

# =======================================================
# Create one agent
# =======================================================
def create_agent():
    age = random.randint(18, 70)
    gender = random.choice(GENDERS)

    persona_raw = ollama_generate(BASE_PROMPT)
    persona = persona_raw.replace("\n", " ").strip()

    return {
        "age": age,
        "gender": gender,
        "persona": persona,
        "memory": [],
        "reflection": [],
        "plan": []
    }

# =======================================================
# Main
# =======================================================
def main():
    print(f"Generating {NUM_AGENTS} RUs using Ollama model: {MODEL}")
    print(f"Saving new RUs to: {OUTPUT_FILE}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agents = []
    for i in range(NUM_AGENTS):
        print(f"Creating agent {i+1}/{NUM_AGENTS}...")
        agent = create_agent()
        agents.append(agent)
        time.sleep(0.2)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(agents, f, indent=4)

    print(f"\nGenerated {NUM_AGENTS} RUs → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
