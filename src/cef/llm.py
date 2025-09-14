from __future__ import annotations
from typing import Any, Dict, List, Optional
from .interfaces import Generation, LLMAdapter
try:
    from .llm_openai import OpenAIAdapter  # optional if openai installed
except Exception:  # pragma: no cover
    OpenAIAdapter = None  # type: ignore


class EchoAdapter(LLMAdapter):
    """A minimal, local adapter used for validation and simulation.
    It returns the prompt and simple token accounting.
    """

    def __init__(self, context_window: int = 16000) -> None:
        self._info = {
            "name": "local-echo",
            "context_window": context_window,
            "pricing": {"input": 0.0, "output": 0.0},
            "supports_tools": False,
        }

    def generate(self, prompt: str, *, system: Optional[str] = None, stop: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                 extra: Optional[Dict[str, Any]] = None) -> Generation:
        text = (system + "\n\n" if system else "") + prompt
        usage = {
            "input_tokens": self.count_tokens(text),
            "output_tokens": min(max_tokens or 64, 64),
        }
        return Generation(text=f"[echo]\n{text}", usage=usage, meta={"adapter": self._info["name"]})

    def count_tokens(self, text: str | List[str]) -> int | List[int]:
        def _estimate(s: str) -> int:
            # naive token estimate: ~4 chars/token
            return max(1, len(s) // 4)
        if isinstance(text, list):
            return [_estimate(t) for t in text]
        return _estimate(text)

    def model_info(self) -> Dict[str, Any]:
        return dict(self._info)


ADAPTERS = {
    "local-echo": EchoAdapter,
    **({"openai": OpenAIAdapter} if OpenAIAdapter else {}),
}
