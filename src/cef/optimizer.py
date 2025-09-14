from __future__ import annotations
from typing import Any, Dict


class SimpleOptimizer:
    """Heuristic optimizer that enforces basic budgets and toggles compression.
    This is a placeholder for more advanced strategies.
    """

    def choose(self, *, state: Dict[str, Any], policies: Dict[str, Any]) -> Dict[str, Any]:
        budgets = policies.get("budgets", {})
        compression = policies.get("compression", {})
        input_used = state.get("budgets", {}).get("input_tokens", 0)
        max_in = budgets.get("max_input_tokens", 4000)
        ratio = input_used / max_in if max_in else 0.0
        should_summarize = ratio >= float(compression.get("summarize_when_ratio", 0.9))
        return {
            "should_summarize": should_summarize,
            "remaining_input_budget": max(0, max_in - input_used),
        }

