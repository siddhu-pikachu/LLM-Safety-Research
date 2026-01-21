# provider.py
# Promptfoo Python provider: call_api(prompt, options, context) -> {"output": "...", "sessionId": "..."}
#
# Key design goals:
# 1) Make multi-turn REAL even if Promptfoo runs a new Python process per turn:
#    - Persist AgentState + turn index on disk under outputs/promptfoo/sessions/<session_id>.pkl
# 2) Make session_id stable across turns:
#    - Prefer vars.sessionId (set via promptfoo transformVars)
#    - Else use Promptfoo's targetConversationId / redTeamingChatConversationId (found in context["test"]["provider"])
# 3) Keep it easy to debug:
#    - Write outputs/promptfoo/context_sample.json once per process
#    - Log errors in a simple JSONL file (outputs/promptfoo/promptfoo_runs.jsonl)

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml  # requires PyYAML installed

from src.targets.ollama_target import OllamaTarget
from src.tools.kb import KBSearchTool
from src.agent.state import AgentState
from src.agent.loop import run_episode

# -----------------------
# Files / directories
# -----------------------
LOG_DIR = Path(os.environ.get("PF_LOG_DIR", "outputs/promptfoo"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "promptfoo_runs.jsonl"
CONTEXT_SAMPLE_PATH = LOG_DIR / "context_sample.json"

SESSION_DIR = LOG_DIR / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Cached runtime objects
# (safe to cache; if process restarts, they re-init)
# -----------------------
_CFG: Optional[Dict[str, Any]] = None
_TARGET: Optional[OllamaTarget] = None
_KB: Optional[KBSearchTool] = None
_MODEL: Optional[str] = None
_KB_VARIANT: Optional[str] = None

_WROTE_CONTEXT_SAMPLE = False


# -----------------------
# Small utilities
# -----------------------
def _append_jsonl(record: Dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_write_context_sample_once(context: Dict[str, Any]) -> None:
    global _WROTE_CONTEXT_SAMPLE
    if _WROTE_CONTEXT_SAMPLE:
        return
    try:
        CONTEXT_SAMPLE_PATH.write_text(
            json.dumps(context, indent=2, default=str),
            encoding="utf-8",
        )
        _WROTE_CONTEXT_SAMPLE = True
    except Exception as e:
        # Don't fail the run if debug write fails.
        _append_jsonl(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "debug_write_failed",
                "error": repr(e),
            }
        )


def _load_cfg() -> Dict[str, Any]:
    """
    Load the same config sweep.py uses.
    Looks for configs/base.yaml then base.yaml.
    """
    global _CFG
    if _CFG is not None:
        return _CFG

    candidates = [Path("configs/base.yaml"), Path("base.yaml")]
    for p in candidates:
        if p.exists():
            _CFG = yaml.safe_load(p.read_text(encoding="utf-8"))
            break

    if _CFG is None:
        raise FileNotFoundError(
            "Could not find configs/base.yaml or base.yaml. "
            "Run Promptfoo from repo root (policy-cartography/) so configs/base.yaml is visible."
        )
    return _CFG


def _init_runtime(kb_variant_override: Optional[str] = None, model_override: Optional[str] = None) -> None:
    """
    Initialize cached runtime objects (Ollama target + KB tool).
    """
    global _TARGET, _KB, _MODEL, _KB_VARIANT

    cfg = _load_cfg()

    model = (model_override or cfg.get("model") or "").strip()
    if not model:
        raise ValueError("Config missing 'model'.")

    base_url = (cfg.get("ollama_base_url") or "").strip()
    if not base_url:
        raise ValueError("Config missing 'ollama_base_url'.")

    kb_variant = (kb_variant_override or cfg.get("kb_variant") or "B").strip()
    kb_path = Path(f"data/kb_{kb_variant}.jsonl")
    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path} (kb_variant={kb_variant})")

    if _TARGET is None:
        _TARGET = OllamaTarget(base_url=base_url)

    if _KB is None or _KB_VARIANT != kb_variant:
        _KB = KBSearchTool(str(kb_path))
        _KB_VARIANT = kb_variant

    _MODEL = model


def _deterministic_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get_session_id(context: Dict[str, Any]) -> str:
    """
    Return a stable session id across turns for the same redteam conversation.
    Priority order:
      1) vars.sessionId (from transformVars)
      2) test.provider.targetConversationId
      3) test.provider.redTeamingChatConversationId
      4) test.id
      5) deterministic hash of test object
    """
    vars_ = context.get("vars") or {}

    sid = vars_.get("sessionId") or vars_.get("session_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()

    test = context.get("test") or {}
    provider_meta = test.get("provider") or {}

    tcid = provider_meta.get("targetConversationId")
    if isinstance(tcid, str) and tcid.strip():
        return tcid.strip()

    rcid = provider_meta.get("redTeamingChatConversationId")
    if isinstance(rcid, str) and rcid.strip():
        return rcid.strip()

    tid = test.get("id")
    if isinstance(tid, str) and tid.strip():
        return tid.strip()

    # last resort: stable hash of test
    if test:
        return _deterministic_hash(test)

    # absolute last resort
    return str(uuid.uuid4())


def _session_path(session_id: str) -> Path:
    # keep filenames safe
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))[:120]
    if not safe:
        safe = _deterministic_hash(session_id)
    return SESSION_DIR / f"{safe}.pkl"


def _load_session(session_id: str) -> Optional[Tuple[AgentState, int]]:
    """
    Load (AgentState, turn_idx) from disk. Returns None if missing/unreadable.
    """
    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            state, turn_idx = pickle.load(f)
        if not isinstance(turn_idx, int):
            turn_idx = int(turn_idx)
        return state, turn_idx
    except Exception as e:
        _append_jsonl(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "session_load_failed",
                "session_id": session_id,
                "path": str(p),
                "error": repr(e),
            }
        )
        return None


def _save_session(session_id: str, state: AgentState, turn_idx: int) -> None:
    """
    Save (AgentState, turn_idx) to disk.
    """
    p = _session_path(session_id)
    try:
        with p.open("wb") as f:
            pickle.dump((state, int(turn_idx)), f)
    except Exception as e:
        _append_jsonl(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "session_save_failed",
                "session_id": session_id,
                "path": str(p),
                "error": repr(e),
            }
        )


def _new_state_from_context(context: Dict[str, Any]) -> AgentState:
    """
    Create a fresh AgentState based on config + promptfoo vars.
    Kept intentionally small.
    """
    cfg = _load_cfg()
    vars_ = context.get("vars") or {}

    memory_enabled = bool(vars_.get("memory_enabled", cfg.get("memory_enabled", True)))
    tool_access_enabled = bool(vars_.get("tool_access_enabled", cfg.get("tool_access_enabled", True)))

    state = AgentState(memory_enabled=memory_enabled, tool_access_enabled=tool_access_enabled)

    # Optional: trust profile/tool trust (if your AgentState supports them)
    trust_profile = vars_.get("trust_profile")
    if trust_profile is not None and hasattr(state, "trust_profile"):
        state.trust_profile = str(trust_profile)

    tool_trust = vars_.get("tool_trust")
    if tool_trust is not None and hasattr(state, "tool_trust"):
        state.tool_trust = str(tool_trust)

    return state


# -----------------------
# Promptfoo entrypoint
# -----------------------
def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]):
    """
    Promptfoo provider entrypoint.
    Must return {"output": "..."}.
    We also return {"sessionId": "..."} to make session continuity explicit.
    """
    t0 = time.time()
    request_id = str(uuid.uuid4())
    vars_ = context.get("vars") or {}

    _safe_write_context_sample_once(context)

    kb_variant_override = vars_.get("kb_variant")
    model_override = vars_.get("model")

    _init_runtime(
        kb_variant_override=str(kb_variant_override) if kb_variant_override else None,
        model_override=str(model_override) if model_override else None,
    )

    assert _TARGET is not None and _KB is not None and _MODEL is not None

    # --- stable session id ---
    session_id = _get_session_id(context)

    # --- disk-backed session ---
    loaded = _load_session(session_id)
    if loaded is None:
        state = _new_state_from_context(context)
        turn_idx = 0
    else:
        state, turn_idx = loaded

    # --- run one agent step (your loop updates state in-place) ---
    record = run_episode(
        target=_TARGET,
        model=_MODEL,
        state=state,
        kb=_KB,
        user_prompt=prompt,
    )

    # bump and persist
    _save_session(session_id, state, turn_idx + 1)

    output_text = record.get("model_final_output")
    if not isinstance(output_text, str):
        output_text = str(output_text)

    latency_ms = int((time.time() - t0) * 1000)

    # Log only useful fields for debugging/analysis
    _append_jsonl(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "session_id": session_id,
            "turn_index": turn_idx,
            "prompt": prompt,
            "vars": vars_,
            "kb_variant": _KB_VARIANT,
            "model": _MODEL,
            "latency_ms": latency_ms,
            "tool_used": record.get("tool_used"),
            "tool_forced": record.get("tool_forced"),
            "tool_requested_by_model": record.get("tool_requested_by_model"),
            "tool_query": record.get("tool_query"),
            "tool_docs": record.get("tool_docs"),
        }
    )

    # Returning sessionId helps promptfoo keep continuity if it supports it in your version.
    return {"output": output_text, "sessionId": session_id}
