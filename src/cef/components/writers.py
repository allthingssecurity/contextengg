from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ScratchpadFileWriter:
    kind: str = "writer"
    name: str = "scratchpad.file"

    def run(self, *, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        base_dir = (env or {}).get("base_dir", "")
        path = params.get("path", ".cache/scratchpad.txt")
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plan = state.get("plan") or "Plan: analyze -> select -> compress -> generate -> review"
        with open(path, "w", encoding="utf-8") as f:
            f.write(plan + "\n")
        return {"scratchpad_ref": path, "plan": plan}
