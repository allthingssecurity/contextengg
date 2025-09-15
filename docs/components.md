Components and Responsibilities

Writers
- ScratchpadWriter: persists ephemeral work artifacts for the current task.
- MemoryWriter: synthesizes episodic/semantic/procedural updates post‑turn.
- StateWriter: manages the Runtime State fields to keep nodes in sync.

Selectors
- InstructionSelector: retrieves procedural rules from rule stores.
- FactSelector: retrieves facts via search/RAG/graph queries.
- ToolSelector: ranks tools by semantics + metadata (success rate, domain).

Compressors
- Summarizer: distills key decisions, facts, and open loops.
- Pruner: removes low‑value or resolved content.
- Deduper: de‑duplicates redundant context blocks.

Isolators
- SubAgent: runs a specialized agent with its own window and summary handoff.
- SandboxRunner: executes code/data tasks, returns structured outputs.
- PayloadSplitter: partitions context across phases/agents.

Optimizer
- Budgeter: estimates tokens/cost; allocates budgets per step.
- Reranker: reorders candidates by utility under constraints.
- StrategyChooser: tunes component parameters and thresholds.

LLM Adapter (Model‑Agnostic)
- Methods: generate, count_tokens, batch_generate, metadata.
- Guarantees: no provider‑specific semantics leak upward.

Registry
- Discovers components via manifests, config, or import hooks.
- Supports versioned components and capability tags.

