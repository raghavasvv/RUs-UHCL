import json
from pathlib import Path

class ReflectionManager:
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
        Append reflection entry.
        Final format: [ {"insight": "..."} ]
        """
        path = self._path(ru_id)
        reflections = self.load(ru_id)

        # Normalize
        if isinstance(entry, dict) and "insight" in entry:
            new_entry = entry
        else:
            new_entry = {"insight": str(entry)}

        reflections.append(new_entry)

        # Save
        with open(path, "w") as f:
            json.dump(reflections, f, indent=2)
