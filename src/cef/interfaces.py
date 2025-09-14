from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class Generation:
    text: str
    usage: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMAdapter(Protocol):
    def generate(self, prompt: str, *, system: Optional[str] = None, stop: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                 extra: Optional[Dict[str, Any]] = None) -> Generation: ...

    def count_tokens(self, text: str | List[str]) -> int | List[int]: ...

    def model_info(self) -> Dict[str, Any]: ...


@runtime_checkable
class Component(Protocol):
    kind: str
    name: str

    def run(self, *, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None,
            env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...


@dataclass
class RuntimeState:
    plan: str = ""
    memory_refs: List[str] = field(default_factory=list)
    scratchpad_ref: Optional[str] = None
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    output_draft: str = ""
    selected: Dict[str, List[str]] = field(default_factory=lambda: {"instructions": [], "facts": [], "tools": []})
    budgets: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

