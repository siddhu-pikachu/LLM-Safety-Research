# Policy-Boundary Cartography (Sandbox v0)

This repo is a research sandbox for evaluating state-dependent policy boundaries in tool-using LLM agents.

## Quickstart (macOS)
1. Install `uv` and Ollama.
2. Pull models:
   - `ollama pull llama3.1:8b`
   - `ollama pull llama3.2:3b`
3. Create env:
   - `uv venv && source .venv/bin/activate`
   - `uv pip install -e .`
4. Run:
   - `python -m src.run`

Logs are written to `outputs/runs/*.jsonl`.
