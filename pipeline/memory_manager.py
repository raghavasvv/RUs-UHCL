import json
from pathlib import Path

class MemoryManager:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ru_id):
        return self.dir / f"{ru_id}.json"

    def load(self, ru_id):
        """Always return a LIST."""
        path = self._path(ru_id)

        if not path.exists():
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except:
            return []

        # Normalize
        if isinstance(data, dict):
            return [data]  
        if isinstance(data, list):
            return data

        return []

    def append(self, ru_id, entry):
        """
        FORCE append new q/a item into the existing list
        """
        path = self._path(ru_id)
        memory = self.load(ru_id)

        # Entry is already correct format
        if "q" in entry and "a" in entry:
            new_item = entry
        else:
            # Convert study task → q/a format
            cond = entry.get("condition", "unknown")
            rating = entry.get("rating", None)

            new_item = {
                "q": f"{cond} harm",
                "a": rating
            }

        # Append
        memory.append(new_item)

        # DEBUG
        print(f" Memory appended → {ru_id}: {new_item}")
        print(f" Saving to → {path}")

        # Save
        with open(path, "w") as f:
            json.dump(memory, f, indent=2)
