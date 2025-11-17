import json
from pathlib import Path

class PlanManager:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ru_id):
        return self.dir / f"{ru_id}.json"

    def load(self, ru_id):
        """Always return list."""
        path = self._path(ru_id)

        if not path.exists():
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except:
            return []

        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data

        return []

    def append(self, ru_id, entry):
        """
        Append plan entry.
        Final format: [ {"next_action": "..."} ]
        """
        path = self._path(ru_id)
        plans = self.load(ru_id)

        # Normalize
        if isinstance(entry, dict) and "next_action" in entry:
            new_plan = entry
        else:
            new_plan = {"next_action": str(entry)}

        plans.append(new_plan)

        # Save
        with open(path, "w") as f:
            json.dump(plans, f, indent=2)
