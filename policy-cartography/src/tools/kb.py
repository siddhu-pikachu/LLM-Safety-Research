import json
from pathlib import Path
from typing import List, Dict


class KBSearchTool:
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.docs: List[Dict] = []
        self._load()

    def _load(self) -> None:
        self.docs.clear()
        with self.kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.docs.append(json.loads(line))

    def search(self, query: str, k: int = 3) -> List[Dict]:
        q = query.lower()
        scored = []
        for d in self.docs:
            text = (d.get("title", "") + " " + d.get("text", "")).lower()
            score = sum(1 for token in q.split() if token in text)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:k] if s > 0] or [d for _, d in scored[:k]]
