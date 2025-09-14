from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..interfaces import Component
from ..llm_openai import OpenAIAdapter  # type: ignore


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())


@dataclass
class InstructionKeywordSelector:
    kind: str = "selector"
    name: str = "instruction.keyword"

    def run(self, *, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        keywords = set([k.lower() for k in params.get("keywords", [])])
        base_dir = (env or {}).get("base_dir", "")
        files = ((env or {}).get("stores", {}).get("rules", {}) or {}).get("files", [])
        selected: List[str] = []
        for fp in files:
            if not os.path.isabs(fp):
                fp = os.path.join(base_dir, fp)
            if not os.path.exists(fp):
                continue
            content = _read_text(fp)
            if not keywords:
                selected.append(content.strip())
            else:
                toks = set(_tokenize(content))
                if any(k in toks for k in keywords):
                    selected.append(content.strip())
        # If nothing matched, add a minimal rule line
        if not selected:
            selected = ["Follow project rules and style guide."]
        return {"selected": {"instructions": selected}}


@dataclass
class FactKeywordSelector:
    kind: str = "selector"
    name: str = "fact.keyword"

    def run(self, *, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        base_dir = (env or {}).get("base_dir", "")
        index_path = ((env or {}).get("stores", {}).get("retrieval", {}) or {}).get("index")
        if index_path and not os.path.isabs(index_path):
            index_path = os.path.join(base_dir, index_path)
        top_k = int(params.get("top_k", 5))
        if not index_path or not os.path.exists(index_path):
            return {"selected": {"facts": []}}
        with open(index_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        query = (env or {}).get("query", "")
        q_toks = set(_tokenize(query))
        scored = []
        explain = []
        for it in items:
            text = (it.get("title", "") + "\n" + it.get("text", "")).lower()
            score = sum(1 for t in q_toks if t in text)
            if score:
                scored.append((score, it))
                explain.append({"id": it.get("id"), "title": it.get("title"), "score": score})
        scored.sort(key=lambda x: x[0], reverse=True)
        facts = [s[1]["text"] for s in scored[:top_k]]
        explain.sort(key=lambda e: e["score"], reverse=True)
        return {"selected": {"facts": facts}, "__explain__": {"ranking": explain[:top_k]}}


@dataclass
class FactEmbeddingSelector:
    kind: str = "selector"
    name: str = "fact.embedding"

    def run(self, *, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Uses OpenAI embeddings to rank a small local index.
        params = params or {}
        base_dir = (env or {}).get("base_dir", "")
        index_path = ((env or {}).get("stores", {}).get("retrieval", {}) or {}).get("index")
        if index_path and not os.path.isabs(index_path):
            index_path = os.path.join(base_dir, index_path)
        model = params.get("embedding_model", "text-embedding-3-small")
        top_k = int(params.get("top_k", 5))
        if not index_path or not os.path.exists(index_path):
            return {"selected": {"facts": []}}
        with open(index_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        query = (env or {}).get("query", "")

        # Adapter is only used to confirm key presence; embeddings use OpenAI SDK directly via llm_openai
        if OpenAIAdapter is None:
            # Fallback to keyword if OpenAI not available
            return FactKeywordSelector().run(state=state, params=params, env=env)

        # Lazy import to avoid circular deps
        from openai import OpenAI  # type: ignore
        import math

        client = OpenAI()
        def embed(text: str) -> List[float]:
            r = client.embeddings.create(model=model, input=text)
            return r.data[0].embedding  # type: ignore

        def cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(y*y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        q_in = query if len(query) <= 4000 else query[:4000]
        qv = embed(q_in)
        scored: List[tuple[float, Dict[str, Any]]] = []
        for it in items:
            txt = it.get("title", "") + "\n" + it.get("text", "")
            dv = embed(txt)
            s = cosine(qv, dv)
            scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        facts = [s[1]["text"] for s in scored[:top_k]]
        return {"selected": {"facts": facts}}


@dataclass
class ToolKeywordSelector:
    kind: str = "selector"
    name: str = "tool.rag"

    def run(self, *, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        base_dir = (env or {}).get("base_dir", "")
        registry_path = ((env or {}).get("stores", {}).get("tools", {}) or {}).get("registry")
        if registry_path and not os.path.isabs(registry_path):
            registry_path = os.path.join(base_dir, registry_path)
        top_n = int(params.get("top_n", 4))
        if not registry_path or not os.path.exists(registry_path):
            return {"selected": {"tools": []}}
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        query = (env or {}).get("query", "").lower()
        results = []
        explain = []
        for t in reg.get("tools", []):
            name = t.get("name", "")
            desc = (t.get("description", "") + " " + " ".join([t.get("domain", "")])).lower()
            score = 0
            for token in _tokenize(query):
                if token in name.lower():
                    score += 2
                if token in desc:
                    score += 1
            score += int(100 * float(t.get("success_rate", 0)))  # prefer higher success
            results.append((score, t))
            explain.append({"name": name, "score": score, "domain": t.get("domain")})
        results.sort(key=lambda x: x[0], reverse=True)
        tools = [r[1]["name"] for r in results[:top_n]]
        explain.sort(key=lambda e: e["score"], reverse=True)
        return {"selected": {"tools": tools}, "__explain__": {"ranking": explain[:top_n]}}
