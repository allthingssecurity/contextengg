from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from .interfaces import RuntimeState
from .llm import ADAPTERS
from .optimizer import SimpleOptimizer
from .components import COMPONENTS


class Orchestrator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        adapter_name = config["model"]["adapter"]
        adapter_cls = ADAPTERS.get(adapter_name)
        if not adapter_cls:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        self.model = adapter_cls(**config["model"].get("params", {}))
        self.optimizer = SimpleOptimizer()

    def _count(self, text: str) -> int:
        try:
            return int(self.model.count_tokens(text))
        except Exception:
            return max(1, len(text) // 4)

    def run_once(self, *, query: str, prior_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = RuntimeState()
        logs: List[str] = []
        trace: List[Dict[str, Any]] = []

        # Seed from prior state (multi-turn carryover)
        if prior_state:
            prev_sel = prior_state.get("selected", {}) or {}
            for k in ("instructions", "facts", "tools"):
                items = prev_sel.get(k, [])
                if items:
                    state.selected[k].extend([x for x in items if x not in state.selected[k]])
            if prior_state.get("plan"):
                state.plan = prior_state.get("plan")
            if prior_state.get("scratchpad_ref"):
                state.scratchpad_ref = prior_state.get("scratchpad_ref")
            trace.append({
                "phase": "write",
                "component": "carryover.state",
                "kind": "writer",
                "added": {"instructions": prev_sel.get("instructions", []), "facts": prev_sel.get("facts", []), "tools": prev_sel.get("tools", [])},
                "state_updates": {"plan": state.plan, "scratchpad_ref": state.scratchpad_ref},
                "explain": "Merged prior turn context for continuity",
            })

        # Preprocess query (dedupe/summarize if excessively long)
        query, qtrace = self._preprocess_query(query)
        if qtrace:
            trace.append(qtrace)

        # Execute pipeline components
        env = {
            "stores": self.config.get("stores", {}),
            "query": query,
            "model": self.model,
            "policies": self.config.get("policies", {}),
            "base_dir": self.config.get("__base_dir__"),
        }

        compressors: List[Dict[str, Any]] = []
        for step in self.config.get("pipeline", []):
            ref = step.get("ref")
            params = step.get("params", {})
            inject_keys = step.get("inject", [])
            kind = step.get("kind")
            if kind == "compressor":
                compressors.append(step)
                continue
            if kind == "optimizer":
                # Orchestrator owns optimization
                continue
            cls = COMPONENTS.get(ref)
            if not cls:
                logs.append(f"Skipping unknown component: {ref}")
                continue
            comp = cls()  # type: ignore
            out = comp.run(state=state.__dict__, params=params, env=env)
            # Merge outputs
            if "selected" in out:
                for k, v in out["selected"].items():
                    # union inject
                    lst = state.selected.get(k, [])
                    lst.extend([x for x in v if x not in lst])
                    state.selected[k] = lst
            # debug log per step
            added_f = len(out.get("selected", {}).get("facts", []))
            added_t = len(out.get("selected", {}).get("tools", []))
            added_i = len(out.get("selected", {}).get("instructions", []))
            logs.append(f"Ran {ref} (kind={kind}): +instructions={added_i}, +facts={added_f}, +tools={added_t}")
            trace.append({
                "phase": "select" if kind == "selector" else ("write" if kind == "writer" else kind),
                "component": ref,
                "kind": kind,
                "added": {
                    "instructions": out.get("selected", {}).get("instructions", []),
                    "facts": out.get("selected", {}).get("facts", []),
                    "tools": out.get("selected", {}).get("tools", []),
                },
                "state_updates": {k: v for k, v in out.items() if k not in ("selected", "__explain__")},
                "explain": out.get("__explain__"),
            })
            # Pass through other top-level fields
            for k, v in out.items():
                if k == "selected":
                    continue
                setattr(state, k, v)

        logs.append(
            f"Selected instructions: {len(state.selected['instructions'])}, facts: {len(state.selected['facts'])}, tools: {len(state.selected['tools'])}"
        )

        # Optimizer: decide compression
        system_text, user_text = self._build_prompt(query, state)
        in_tokens = self._count(system_text + "\n\n" + user_text)
        state.budgets["input_tokens"] = in_tokens
        decision = self.optimizer.choose(state=state.__dict__, policies=self.config.get("policies", {}))
        logs.append(f"Optimizer decision: {json.dumps(decision)}")
        trace.append({
            "phase": "optimize",
            "component": "optimizer",
            "kind": "optimizer",
            "decision": decision,
            "budgets": {"input_tokens": in_tokens},
        })

        if decision.get("should_summarize"):
            # Run summarizer if present in pipeline (lookup by ref)
            for step in self.config.get("pipeline", []):
                if step.get("kind") == "compressor" and step.get("ref") == "summarize.decision":
                    cls = COMPONENTS.get("summarize.decision")
                    if cls:
                        comp = cls()  # type: ignore
                        out = comp.run(state=state.__dict__, params=step.get("params", {}), env=env)
                        if "selected" in out:
                            for k, v in out["selected"].items():
                                state.selected[k] = v
                        trace.append({
                            "phase": "compress",
                            "component": "summarize.decision",
                            "kind": "compressor",
                            "result": out.get("selected", {}),
                        })
                        logs.append("Applied summarization to fit budgets")
                        break
            # Rebuild prompt after compression
            system_text, user_text = self._build_prompt(query, state)

        # Isolation note (placeholder)
        logs.append("Isolation: no sandbox run in simulation")

        # Generate
        max_out = self.config.get("policies", {}).get("budgets", {}).get("max_output_tokens", 512)
        gen = self.model.generate(user_text, system=system_text, max_tokens=max_out)
        state.output_draft = gen.text
        logs.append("Generated output via adapter")
        trace.append({
            "phase": "generate",
            "component": "llm",
            "kind": "adapter",
            "prompt_tokens": in_tokens,
            "output_tokens": gen.usage.get("output_tokens"),
            "max_output_tokens": max_out,
        })

        # Finalize
        state.logs.extend(logs)
        # Build attribution from trace
        fact_rank = []
        tool_rank = []
        # heuristic mapping from tools to keywords
        tool_keywords = {
            "event_mesh_check": ["event mesh", "queue", "queues", "dlq", "dead-letter", "idempotency", "retry"],
            "btp_logs": ["logs", "latency", "p95", "error rate", "monitoring"],
            "xsuaa_token": ["xsuaa", "jwt", "token", "scope", "role"],
            "destinations_list": ["destination", "destinations", "connectivity"],
            "cpi_monitor": ["cpi", "integration suite", "iflow", "alert"],
            "ctms_transport": ["transport", "promotion", "ctms"],
        }
        try:
            for t in trace:
                if t.get("phase") == "select" and t.get("kind") == "selector":
                    comp = t.get("component") or ""
                    exp = t.get("explain") or {}
                    ranking = exp.get("ranking") or []
                    if comp.startswith("fact") and ranking:
                        fact_rank = ranking
                    if comp == "tool.rag" and ranking:
                        tool_rank = ranking
        except Exception:
            pass

        # attribute each fact to a likely tool (heuristic) and/or KB item when available
        facts_attr = []
        for idx, f in enumerate(state.selected.get("facts", [])):
            f_low = f.lower()
            tool_match = None
            for tool_name, kws in tool_keywords.items():
                if any(kw in f_low for kw in kws):
                    if tool_name in state.selected.get("tools", []):
                        tool_match = tool_name
                        break
            kb_match = None
            if fact_rank:
                kb_entry = fact_rank[min(idx, len(fact_rank)-1)] if len(fact_rank) else None
                if isinstance(kb_entry, dict):
                    kb_match = kb_entry.get("title") or kb_entry.get("id")
            facts_attr.append({"text": f, "tool": tool_match, "kb": kb_match})

        prompt_breakdown = {
            "system": system_text,
            "user": user_text,
            "sources": {
                "instructions": state.selected.get("instructions", []),
                "facts": state.selected.get("facts", []),
                "tools": state.selected.get("tools", []),
                "plan": state.plan,
            },
            "sources_meta": {
                "instructions_sources": (self.config.get("stores", {}).get("rules", {}) or {}).get("files", []),
                "facts_sources": fact_rank,
                "tools_sources": tool_rank,
            },
            "facts_attribution": facts_attr,
        }
        return {
            "output": gen.text,
            "usage": gen.usage,
            "logs": logs,
            "state": state.__dict__,
            "model": self.model.model_info(),
            "trace": trace,
            "effective_query": query,
            "prompt": prompt_breakdown,
        }

    def _preprocess_query(self, query: str):
        # Remove excessive repetition and optionally summarize very long inputs
        original = query
        # Collapse consecutive duplicate sentences
        parts = [p.strip() for p in query.split(".")]
        deduped: List[str] = []
        prev = None
        repeat_count = 0
        for p in parts:
            if not p:
                continue
            if prev is not None and p == prev:
                repeat_count += 1
                continue
            else:
                if prev is not None and repeat_count > 0:
                    deduped.append(f"{prev}. (repeated {repeat_count}x)")
                if prev is not None and repeat_count == 0:
                    deduped.append(prev + ".")
                prev = p
                repeat_count = 0
        if prev is not None:
            if repeat_count > 0:
                deduped.append(f"{prev}. (repeated {repeat_count}x)")
            else:
                deduped.append(prev + ".")

        compact = " ".join(deduped).strip()
        # If compaction failed to reduce meaningfully, keep original
        candidate = compact if len(compact) < len(original) else original
        in_tokens = self._count(candidate)
        threshold = int(self.config.get("policies", {}).get("compression", {}).get("pre_summarize_query_over_tokens", 6000))
        if in_tokens <= threshold:
            return candidate, {
                "phase": "compress",
                "component": "query.preprocessor",
                "kind": "compressor",
                "result": {
                    "before_tokens": self._count(original),
                    "after_tokens": in_tokens,
                    "collapsed_repetitions": len(original) - len(candidate),
                },
                "after_preview": candidate[:300] + ("…" if len(candidate) > 300 else ""),
            }

        # Summarize the background to fit budget
        prompt = (
            "The following query is verbose with repeated background context.\n"
            "Rewrite it into a concise version preserving only the task, key constraints, and any critical details.\n"
            "Cap at ~500 tokens.\n\nQuery:\n" + candidate
        )
        try:
            gen = self.model.generate(prompt, max_tokens=500, temperature=0)
            summarized = gen.text
            return summarized, {
                "phase": "compress",
                "component": "query.summarizer",
                "kind": "compressor",
                "result": {
                    "before_tokens": in_tokens,
                    "after_tokens": self._count(summarized),
                },
                "after_preview": summarized[:300] + ("…" if len(summarized) > 300 else ""),
            }
        except Exception:
            # Fallback to truncated content
            truncated = candidate[: min(len(candidate), 12000)]
            return truncated, {
                "phase": "compress",
                "component": "query.truncate",
                "kind": "compressor",
                "result": {
                    "before_tokens": in_tokens,
                    "after_tokens": self._count(truncated),
                },
                "after_preview": truncated[:300] + ("…" if len(truncated) > 300 else ""),
            }

    def _build_prompt(self, query: str, state: RuntimeState) -> tuple[str, str]:
        instructions = "\n".join(state.selected.get("instructions", [])).strip()
        facts = "\n".join(state.selected.get("facts", [])).strip()
        system_sections: List[str] = []
        preamble = (
            "You are an expert enterprise architect. Provide a polished, executive-ready deliverable. "
            "Do not mention internal process, context sources, or headings like instructions/facts/tools/plan. "
            "Do not restate the question; go straight to the answer."
        )
        system_sections.append(preamble)
        if instructions:
            system_sections.append("House Rules:\n" + instructions)
        if facts:
            system_sections.append(
                "Reference Notes (do not cite; for your reasoning only):\n" + facts
            )
        system_sections.append(
            "Output Requirements:\n- Clear structure with concise sections.\n- No meta commentary.\n- No references to internal steps (write/select/compress/isolate/optimize).\n- Deliverables: Objectives, Phases (with owners/gates), Risks & Mitigations, Evidence/SLAs, Next Steps."
        )
        system_text = "\n\n".join([s for s in system_sections if s.strip()])
        user_text = query.strip()
        return system_text, user_text
