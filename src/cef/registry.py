from __future__ import annotations
import importlib
from typing import Any, Dict


class Registry:
    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        self._components[name] = component

    def get(self, name: str) -> Any:
        if name not in self._components:
            raise KeyError(f"Component not found: {name}")
        return self._components[name]

    def load_entrypoint(self, entrypoint: str) -> Any:
        # Entry point form: package.module:ClassName
        mod_name, cls_name = entrypoint.split(":")
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)

