Context Engineering Framework (LLM‑Agnostic)

Purpose
- Build agents that write, select, compress, and isolate context in a modular, pluggable, and model‑agnostic way.
- Provide a clear taxonomy, configuration schemas, and a minimal reference runtime to validate designs and simulate flows.

Highlights
- LLM‑agnostic adapters: plug any model via a thin interface.
- Pluggable components: selectors, compressors, writers, isolators, optimizers.
- Policy‑driven orchestration: token budgets, cost/latency targets, selection strategies.
- Observability: explicit logs of what was written, selected, compressed, isolated.
- Non‑opinionated taxonomy: no references to specific frameworks.

Four Pillars
- Write: Scratchpads and memory outside the model window.
- Select: Retrieve only relevant instructions, facts, and tools.
- Compress: Summarize and prune to fit budgets.
- Isolate: Split across agents, sandboxes, and state.

Quick Start (Reference Runtime)
1) Explore: docs/ and specs/ to understand the model.
2) Validate a config: `python -m cef.cli validate examples/agents/code_agent.yaml`
3) Simulate (echo adapter): `python -m cef.cli simulate examples/agents/code_agent.yaml --query "Build a parser"`

Using OpenAI (Optional)
- Set `OPENAI_API_KEY` in your environment.
- Use a config with `model.adapter: openai` and `model.params.model: gpt-4o-mini` (see example below).
- Run: `python -m cef.cli simulate examples/agents/code_agent_openai.yaml --query "Build a parser"`

Project Structure
- docs/: Architecture and design guides
- specs/: JSON Schemas for agents, plugins, state, and policies
- examples/: Ready‑to‑adapt YAML templates
- src/cef/: Minimal runtime, interfaces, registry, optimizer, CLI

Design Principles
- Minimal Viable Context: Provide only what the model needs now.
- Delayed Injection: Load context lazily and conditionally.
- Layered Recall: scratchpad → memory → retrieval → tools → sandbox.
- Observability: Log each step’s context decisions.
- Conflict Prevention: De‑dupe and resolve contradictory instructions/facts.

Notes
- This repo is a blueprint plus a minimal reference runtime. Swap pieces freely.
- The reference runtime is intentionally small; extend or replace per stack.
Professional UI (Streamlit)
- Install extras: `pip install streamlit`
- Run UI: `PYTHONPATH=context-engine/src streamlit run context-engine/ui/streamlit_app.py`
- In the UI:
  - Choose adapter (`openai` or `local-echo`)
  - Paste your `OPENAI_API_KEY` (or stay on echo)
  - Pick a scenario and run
  - Board-style view shows Write, Select, Compress, Isolate, Optimize with full trace
