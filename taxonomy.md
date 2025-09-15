Taxonomy (Framework-Agnostic)

- Write: ScratchpadWriter, MemoryWriter, StateWriter
- Select: InstructionSelector, FactSelector, ToolSelector
- Compress: Summarizer, Pruner, Deduper
- Isolate: SubAgent, SandboxRunner, PayloadSplitter
- Optimize: Budgeter, Reranker, StrategyChooser
- Adapt: LLMAdapter (model-agnostic)
- Observe: Event log of write/select/compress/isolate decisions

