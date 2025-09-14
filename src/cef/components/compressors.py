from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..interfaces import Component
from ..llm import ADAPTERS


@dataclass
class DecisionSummarizer:
    kind: str = "compressor"
    name: str = "summarize.decision"

    def run(self, *, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        preserve = params.get("preserve", ["actions", "facts", "open_loops"])  # not strictly used in this stub
        # Build a compact prompt from selected context and ask the model to summarize facts.
        model = (env or {}).get("model")
        if model is None:
            return {}
        facts = (state.get("selected", {}) or {}).get("facts", [])
        instructions = (state.get("selected", {}) or {}).get("instructions", [])
        if not facts and not instructions:
            return {}
        prompt = (
            "Summarize the following context into crisp bullets. Preserve key actions, critical facts, and open questions.\n\n"
            f"Instructions:\n{chr(10).join(instructions)}\n\n"
            f"Facts:\n{chr(10).join(facts)}\n\n"
            "Return at most 10 bullets."
        )
        gen = model.generate(prompt, max_tokens=256, temperature=0)
        # Replace facts with their summary to reduce footprint
        return {"selected": {"facts": [gen.text]}}

