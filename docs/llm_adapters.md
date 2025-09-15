LLM Adapter Contract

Purpose
- Provide a uniform interface across LLM providers and local models.
- Centralize token accounting, batching, and model metadata.

Interface (conceptual)
- generate(prompt: str, *, system: str | None, stop: list[str] | None, max_tokens: int | None, temperature: float | None, extra: dict | None) -> Generation
- count_tokens(text: str | list[str]) -> int | list[int]
- batch_generate(prompts: list[Prompt]) -> list[Generation]
- model_info() -> { name, context_window, pricing, supports_tools, … }

Guidelines
- Avoid provider‑specific fields at higher layers; pass via `extra` if needed.
- Always count tokens and return usage in Generation metadata.
- Keep adapters thin; move policy to orchestrator/optimizer.

