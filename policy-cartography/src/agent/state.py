from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AgentState:
    memory_enabled: bool = True
    tool_access_enabled: bool = True

    # NEW:
    trust_profile: str = "M0"          # "M0", "M1", "M2", "M3"
    tool_trust: str = "untrusted"      # "trusted" or "untrusted"

    # conversation transcript as a list of {"role": "...", "content": "..."}
    history: List[Dict[str, Any]] = field(default_factory=list)

    # optional memory buffer (very simple for v0)
    memory: List[str] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def remember(self, note: str) -> None:
        if self.memory_enabled:
            self.memory.append(note)
