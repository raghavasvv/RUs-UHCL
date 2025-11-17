import sys
from pathlib import Path

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parents[2]
sys.path.append(str(PROJECT_ROOT))

import json

RUS_PATH = PROJECT_ROOT / "RUS" / "synthetic_RUS.json"
with open(RUS_PATH, "r") as f:
    rus_units = json.load(f)

# Assign IDs as Ames & Fiske script does
for i, rus in enumerate(rus_units, start=1):
    rus["RU_id"] = f"RUS_{i:04d}"

# Show first 15 IDs
print("\nFirst 15 RU IDs:")
for ru in rus_units[:15]:
    print(ru["RU_id"])
