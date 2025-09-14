from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # SDK v1+
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore

from .interfaces import Generation, LLMAdapter


class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Install 'openai>=1.0.0'.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self._client = OpenAI(api_key=key)
        self._model = model
        self._info = {
            "name": f"openai:{model}",
            "context_window": None,
            "pricing": {},
            "supports_tools": True,
        }

    def generate(self, prompt: str, *, system: Optional[str] = None, stop: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                 extra: Optional[Dict[str, Any]] = None) -> Generation:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if stop is not None:
            kwargs["stop"] = stop
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if extra:
            kwargs.update(extra)

        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", None) if u is not None else None,
            "output_tokens": getattr(u, "completion_tokens", None) if u is not None else None,
            "total_tokens": getattr(u, "total_tokens", None) if u is not None else None,
        }
        return Generation(text=text, usage=usage, meta={"adapter": self._info["name"]})

    def count_tokens(self, text: str | List[str]) -> int | List[int]:
        # Heuristic fallback; for stricter accounting, integrate tiktoken with model name.
        def _estimate(s: str) -> int:
            return max(1, len(s) // 4)
        if isinstance(text, list):
            return [_estimate(t) for t in text]
        return _estimate(text)

    def model_info(self) -> Dict[str, Any]:
        return dict(self._info)
