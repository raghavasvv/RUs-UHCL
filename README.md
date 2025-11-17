# UHCL_CAPSTONE: Reflector Units Pipeline

This repository contains the full pipeline for generating, running, and evaluating Reflector Units (RUs).
Reflector Units are synthetic agents equipped with memory, reflection, and planning components, designed to simulate human survey behavior.
This system supports cloud-based LLMs, local LLMs (Ollama), OCEAN personality profiling, behavioral replications, media-question responses, and human vs agent comparisons.

The codebase is structured so that any developer can install and run the complete workflow end-to-end.

---

## Folder Structure

```
UHCL_CAPSTONE/
│
├── pipeline/
│     ├── memory_manager.py
│     ├── reflection_manager.py
│     ├── plan_manager.py
│     ├── initialise_RU's.py
│     ├── CLOUD_API/
│     │       └── run_RUS_cloud.py
│     ├── LOCAL_LLM/
│     │       └── run_RUS_LLM.py
│     ├── 5_STUDIES/
│     │       ├── ames_and_fiske.py
│     │       ├── cooney_et_al.py
│     │       ├── halevy_halali.py
│     │       ├── rai_et_al.py
│     │       ├── schilke_et_al.py
│     ├── HUMAN_VS_HUMAN/
│     ├── MEDIA/
│
├── questions/
│     ├── OCEAN.json
│     ├── Psychometrics.json
│     ├── media_questions.json
│     └── additional_question_files_here.json
│
├── RUS/
│     └── synthetic_RUS.json
│
├── memory/
│     └── RUS_XXXX.json
├── reflections/
├── plans/
│
├── human/
│     ├── human_responses_phase1.csv
│     ├── human_responses_phase2.csv
│     ├── human_ocean_1000_1.json
│     ├── human_ocean_1000_2.json
│
├── results/
│     └── study_results/*.csv
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Requirements

Python version 3.10 or later is recommended.

Install all dependencies using:

```
pip install -r requirements.txt
```

Main libraries used:

```
openai
pandas
numpy
scipy
statsmodels
tqdm
python-dotenv
matplotlib
requests
jsonlines
ollama
```

---

## Environment Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY="your_api_key_here"
MODEL_NAME="gpt-4o-mini"
```

For local LLM execution via Ollama:

1. Install Ollama from: [https://ollama.ai](https://ollama.ai)
2. Pull a model:

```
ollama pull llama3
```

---

## Running the Pipeline:

## Adding Custom Question Files

You may create new question sets and place them inside `questions/`.

Example:

```
questions/
    my_custom_questions.json
```

To use the new question file, update the question path in your RU execution script.

For cloud:

```
pipeline/CLOUD_API/run_RUS_cloud.py
```

For local LLM:

```
pipeline/LOCAL_LLM/run_RUS_LLM.py
```

Modify the line:

```python
QUESTIONS_PATH = "questions/my_custom_questions.json"
```

Ensure the JSON format matches existing question files.

## Initialising New RUs

Users may generate a completely new set of RUs by running:

```
python "pipeline/initialise_RU's.py"
```

This script generates:

* A new `synthetic_RUS.json`
* Fresh memory files
* Fresh reflection files
* Fresh plan files

Developers running this project for the first time or attempting custom experiments should initialize new RUs before running their pipelines.

### 1. Running the OCEAN Personality Survey

Cloud:

```
python pipeline/CLOUD_API/run_OCEAN_cloud.py
```

Local LLM:

```
python pipeline/LOCAL_LLM/run_OCEAN_LLM.py
```

Output is stored under:

```
results/ocean/
memory/
reflections/
plans/
```

---

### 2. Running RUs with Cloud API

```
python pipeline/CLOUD_API/run_RUS_cloud.py
```

This script:

* Loads RUs
* Loads question set
* Applies memory, reflection, and planning
* Saves updated RU states

---

### 3. Running RUs with Local LLM

```
python pipeline/LOCAL_LLM/run_RUS_LLM.py
```

Requires Ollama and a locally installed model such as Llama-3.

---

### 4. Running Replication Studies

Each replication study is executed independently.

Examples:

```
python pipeline/5_STUDIES/ames_and_fiske.py
python pipeline/5_STUDIES/cooney_et_al.py
python pipeline/5_STUDIES/halevy_halali.py
python pipeline/5_STUDIES/rai_et_al.py
python pipeline/5_STUDIES/schilke_et_al.py
```

Outputs are placed in:

```
results/study_results/
```

---

### 5. Running Media-Response Study

```
python pipeline/MEDIA/run_media_responses.py
```

---

### 6. Human vs Agent Comparison

```
python pipeline/HUMAN_VS_HUMAN/compare_localvshuman.py
```

---

## Initialising New RUs

Users may generate a completely new set of RUs by running:

```
python "pipeline/initialise_RU's.py"
```

This script generates:

* A new `synthetic_RUS.json`
* Fresh memory files
* Fresh reflection files
* Fresh plan files

Developers running this project for the first time or attempting custom experiments should initialize new RUs before running their pipelines.

---


---

## Output Directory Structure

Outputs from various modules will appear in:

```
results/study_results/       Study results CSVs
results/ocean/               OCEAN phase results
memory/                      RU memory files
reflections/                 Reflection logs
plans/                       Plan logs
```

---

## Troubleshooting

1. API key errors
   Ensure `.env` contains a valid OpenAI API key.

2. Local LLM not found
   Pull a local model via:

   ```
   ollama pull llama3
   ```

3. Windows path issues
   Use quotes for apostrophes:

   ```
   python "pipeline/initialise_RU's.py"
   ```

4. Very large memory/reflection files
   Reinitialize RUs if files exceed several MB.