import json
import random
import time
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# =======================================================
# Load .env from project root
# =======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure it is in your project root .env file.")

client = OpenAI(api_key=api_key)


# =======================================================
# Configuration
# =======================================================
NUM_RUs = 50  # change as needed
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

OUTPUT_DIR = PROJECT_ROOT / "RUS"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"generated_RUS_{timestamp}.json"

GENDERS = ["male", "female"]


# =======================================================
# Persona Prompt
# =======================================================
BASE_PROMPT = """
Create a short human persona in under 25 words.

Include:
1. Occupation or role
2. 2–3 personality traits
3. A daily-life or social behavior detail

Return only ONE line. No numbering, no lists, no quotes.
"""


# =======================================================
# Function to create one RU
# =======================================================
def create_ru():
    age = random.randint(18, 70)
    gender = random.choice(GENDERS)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": BASE_PROMPT}],
            temperature=1.2,
            max_tokens=70
        )
        persona = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error while generating RU persona: {e}")
        persona = "Generic person with neutral traits."

    return {
        "age": age,
        "gender": gender,
        "persona": persona,
        "memory": [],
        "reflection": [],
        "plan": []
    }


# =======================================================
# Main Generator
# =======================================================
def main():
    print(f"Generating {NUM_RUs} Reflector Units (RUs) using model: {MODEL}")
    print(f"Saving output to: {OUTPUT_FILE}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rus = []
    for i in range(NUM_RUs):
        print(f"Creating RU {i + 1}/{NUM_RUs} ...")
        ru = create_ru()
        rus.append(ru)
        time.sleep(0.4)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(rus, f, indent=4)

    print(f"\nCompleted. Generated {NUM_RUs} RUs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
