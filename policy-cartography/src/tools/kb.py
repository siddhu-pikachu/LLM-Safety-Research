import json
from pathlib import Path
from typing import Dict, List, Optional


class KBSearchTool:
    """
    Ultra-simple KB tool for stress-testing exfil:
      - Loads JSONL docs: {"id","title","text"}
      - search(...) ignores the query and returns ALL docs (dump-all)
      - to_tool_text(...) renders docs into one TOOL_RESULT string
    """

    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.docs: List[Dict] = []
        self._load()

    def _load(self) -> None:
        self.docs = []
        with self.kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.docs.append(json.loads(line))

    def search(self, query: str = "", k: Optional[int] = None) -> List[Dict]:
        """
        Fake search: returns the entire KB every time.
        Keep signature compatible with old code.
        """
        return self.docs if k is None else self.docs[:k]

    def to_tool_text(self, docs: Optional[List[Dict]] = None, max_chars: Optional[int] = None) -> str:
        """
        Render docs into a single big tool-result string.
        max_chars is optional; set to avoid blowing model context.
        """
        docs = self.docs if docs is None else docs
        out = []
        used = 0

        for d in docs:
            doc_id = d.get("id", "")
            title = d.get("title", "")
            text = d.get("text", "")

            chunk = f"[{doc_id}] {title}\n{text}\n\n"

            if max_chars is not None and used + len(chunk) > max_chars:
                out.append("\n[KB_TRUNCATED]\n")
                break

            out.append(chunk)
            used += len(chunk)

        return "".join(out).strip()
