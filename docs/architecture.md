Architecture Overview

Goals
- Decouple context management concerns into composable components.
- Optimize for token/cost budgets while preserving task performance.
- Remain portable across LLM providers and execution environments.

Core Concepts
- Orchestrator: Drives the lifecycle per request/turn using policies.
- Runtime State: Structured object containing plan, scratchpad refs, selected items, tool history, and output draft.
- Components (pluggable):
  - Writers: Scratchpad, memory updaters, state writers.
  - Selectors: Instruction, fact, tool selectors (RAG, keyword, rules).
  - Compressors: Summarizers, pruners, deduplicators.
  - Isolators: Sub‑agent wrappers, sandbox runners, payload splitters.
  - Optimizer: Chooses components/levels to meet budgets and objectives.
  - LLM Adapter: Uniform interface to any model.

Lifecycle (per request)
1. Receive query and hydrate Runtime State.
2. Select relevant instructions, facts, and tools.
3. Write initial plan/scratchpad; update memories when applicable.
4. If near budgets, compress past context; prune irrelevancies.
5. Isolate heavy work (sandbox/sub‑agent) and pass structured summaries.
6. Generate with LLM using minimized, curated context.
7. Write results; update long‑term memory; clean ephemeral artifacts.

Optimization Loop
- Inputs: token budget, latency target, cost ceiling, quality preference.
- Process: estimate → plan selection/compression → simulate → adjust.
- Outputs: chosen components, their parameters, and injection set.

Observability
- For each step, log: written items, selected sources, compression stats, isolation boundaries, and budgets used.

