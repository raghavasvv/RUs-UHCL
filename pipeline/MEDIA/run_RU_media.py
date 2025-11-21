import json, os, csv, random, time
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# ======================================================
# Path setup
# ======================================================
ROOT = Path(__file__).resolve().parents[2]  
RUS_FILE = ROOT / "RUS" / "synthetic_RUS.json"
MEDIA_Q_FILE = ROOT / "questions" / "media_questions.json"
RESULTS_DIR = ROOT / "results" / "media"
RESULTS_CSV = RESULTS_DIR / "media_RUs.csv"  

# ======================================================
# Run configuration
# ======================================================
NUM_RUs_TO_RUN = 100  # change this number (e.g., 100, 250, 500, 1000)

# ======================================================
# Load API key (cross-platform safe)
# ======================================================
env_path = find_dotenv(filename=".env", raise_error_if_not_found=False)
if env_path:
    load_dotenv(env_path)
else:
    print("⚠️  Warning: .env file not found. Make sure it exists at the project root.")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

# optional masked print for debugging
print(f"🔑 Loaded OpenAI key: {API_KEY[:7]}...{API_KEY[-4:]}")

client = OpenAI(api_key=API_KEY)

# ======================================================
# Helper functions
# ======================================================
def load_rus():
    """Load Reflector Units (RUs) data from synthetic_RUS.json"""
    print(f"📂 Looking for RU file at: {RUS_FILE}")
    if not RUS_FILE.exists():
        raise FileNotFoundError(f"❌ RU file not found at {RUS_FILE}")

    with open(RUS_FILE, "r") as f:
        rus_units = json.load(f)

    if isinstance(rus_units, dict) and "RUs" in rus_units:
        rus_units = rus_units["RUs"]

    for i, r in enumerate(rus_units, start=1):
        r.setdefault("RUs_id", f"RU_{i:03d}")

    print(f"Successfully loaded {len(rus_units)} total RUs from file.")
    return rus_units[:NUM_RUs_TO_RUN]


def load_json(path):
    """General-purpose JSON loader with existence check"""
    print(f"📘 Loading JSON file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"❌ JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

# ======================================================
# Prompt builder
# ======================================================
def build_prompt(ru, question):
    demo = ", ".join([f"{k}: {v}" for k, v in ru.get("demographics", {}).items()]) if ru.get("demographics") else ""
    persona = f" Persona: {ru.get('persona')}" if ru.get("persona") else ""

    options = question["options"][:]
    random.shuffle(options)  # prevent order bias

    return f"""
You are a Reflector Unit (RU) representing a human survey respondent.
{persona} Demographics: {demo}

Answer the following survey question honestly, based on your personal opinion and experience.
There is no right or wrong answer.

Question [{question['id']}]: {question['question']}
Options (copy one EXACTLY as written): {', '.join(options)}

Give only one line in this format:
Final Answer: <paste one option exactly>
""".strip(), options

# ======================================================
# GPT Response
# ======================================================
def gpt_reason_and_answer(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.1,
            top_p=0.9,
            presence_penalty=0.2,
            frequency_penalty=0.2,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR_{type(e).__name__}"

# ======================================================
# Main Function
# ======================================================
def run_media_RUs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rus_units = load_rus()
    media_qs = load_json(MEDIA_Q_FILE)
    print(f"✅ Loaded {len(rus_units)} RUs and {len(media_qs)} media questions")

    # --- Append mode (Option 1) ---
    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "RUs_id", "question_id", "response"])
        if not file_exists:
            writer.writeheader()  # write header only once

        for i, ru in enumerate(rus_units, start=1):
            for q in media_qs:
                qid = str(q["id"])
                prompt, opt_list = build_prompt(ru, q)
                response_text = gpt_reason_and_answer(prompt)

                # --- Parse choice exactly ---
                text = response_text.strip().lower()
                chosen = None
                prefix = "final answer:"
                if prefix in text:
                    tail = text.split(prefix, 1)[1].strip()
                    tail = " ".join(tail.split())
                    for opt in opt_list:
                        if tail == opt.lower():
                            chosen = opt
                            break
                if not chosen:
                    for opt in opt_list:
                        if f" {opt.lower()} " in f" {text} ":
                            chosen = opt
                            break
                if not chosen:
                    chosen = random.choice(opt_list)  # fallback

                writer.writerow({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "RUs_id": ru["RUs_id"],
                    "question_id": qid,
                    "response": chosen
                })

            if i % 50 == 0 or i == len(rus_units):
                print(f"   ✅ Completed {i}/{len(rus_units)} RUs")

    print("\n✅ All RUs finished answering all questions.")
    print(f"📄 Results appended to: {RESULTS_CSV}")
    print("===========================================")

# ======================================================
# Entrypoint
# ======================================================
if __name__ == "__main__":
    print("🚀 Running media survey for Reflector Units (RUs)...")
    run_media_RUs()
