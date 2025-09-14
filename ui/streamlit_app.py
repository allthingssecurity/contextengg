from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml

from cef.runtime import Orchestrator
from cef.llm import ADAPTERS


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_agent(config_path: Path, query: str, api_key: str | None, prior_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    # base dir resolving
    p = config_path.resolve()
    base = p.parent
    for anc in p.parents:
        if anc.name == "examples":
            base = anc.parent
            break
    cfg["__base_dir__"] = str(base)

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    orch = Orchestrator(cfg)
    return orch.run_once(query=query, prior_state=prior_state)


def run_baseline(config_path: Path, query: str, api_key: str | None) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    adapter_name = cfg["model"]["adapter"]
    params = cfg["model"].get("params", {})
    Adapter = ADAPTERS.get(adapter_name)
    if not Adapter:
        raise RuntimeError(f"Adapter not found: {adapter_name}")
    model = Adapter(**params)
    system = (
        "You are an expert enterprise architect. Provide a concise, executive-ready answer. "
        "No meta commentary."
    )
    max_out = cfg.get("policies", {}).get("budgets", {}).get("max_output_tokens", 512)
    gen = model.generate(query, system=system, max_tokens=max_out)
    return {"output": gen.text, "usage": gen.usage, "model": model.model_info()}


# --------- Simple quality scoring heuristics (UI-only) ---------
STOPWORDS = set(
    "the a an and or of to for in on with by as is are be this that those these from into over under at it its their our your you we they".split()
)


def _tokens(s: str) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", s)]


def _keywords(s: str, limit: int = 30) -> set[str]:
    toks = [t for t in _tokens(s) if t not in STOPWORDS and len(t) > 3]
    # frequency-based selection
    from collections import Counter
    top = [w for w, _ in Counter(toks).most_common(limit)]
    return set(top)


def score_output(output: str, selected: Dict[str, list[str]] | None = None, query_text: str | None = None) -> Dict[str, float]:
    # Section coverage
    want = ["objectives", "phases", "risks", "mitigations", "evidence", "sla", "next steps"]
    o_low = output.lower()
    section_hits = sum(1 for w in want if w in o_low)
    section_cov = section_hits / len(want)

    # Actionability: count bullets and imperative verbs
    verbs = ["implement", "define", "configure", "deploy", "monitor", "test", "mitigate", "schedule", "assign", "review", "measure", "gate"]
    lines = [ln.strip() for ln in output.splitlines()]
    bullets = sum(1 for ln in lines if ln.startswith(('- ', '* ', '1.', '2.', '3.')))
    verb_hits = sum(1 for ln in lines for v in verbs if ln.lower().startswith(v) or f" {v} " in ln.lower())
    actionability = min(1.0, (bullets * 0.02) + (verb_hits * 0.02))

    # Relevance: keyword overlap with query and with selected context (max of the two)
    out_kw = _keywords(output)
    overlap_q = 0.0
    if query_text:
        q_kw = _keywords(query_text)
        overlap_q = len(q_kw & out_kw) / max(1, len(q_kw))
    overlap_c = 0.0
    if selected:
        ref_text = "\n".join(selected.get("instructions", []) + selected.get("facts", []))
        ref_kw = _keywords(ref_text)
        overlap_c = len(ref_kw & out_kw) / max(1, len(ref_kw))

    overlap = max(overlap_q, overlap_c)

    # If API key is present, augment relevance with embedding similarity
    try:
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            import math
            client = OpenAI()
            def emb(t: str):
                r = client.embeddings.create(model="text-embedding-3-small", input=t[:4000])
                return r.data[0].embedding
            vec_out = emb(output)
            sims = []
            if query_text:
                vec_q = emb(query_text)
                sims.append(_cos(vec_out, vec_q))
            if selected:
                ref_text = "\n".join(selected.get("instructions", []) + selected.get("facts", []))
                if ref_text.strip():
                    vec_ref = emb(ref_text[:4000])
                    sims.append(_cos(vec_out, vec_ref))
            if sims:
                overlap = max(overlap, max(sims))
    except Exception:
        pass

    # Conciseness: prefer <= 800 tokens (approx chars/4)
    approx_tokens = max(1, len(output) // 4)
    conciseness = 1.0 if approx_tokens <= 800 else max(0.0, 1.0 - (approx_tokens - 800) / 800)

    # Composite (weighted)
    composite = round(0.35 * section_cov + 0.35 * actionability + 0.2 * overlap + 0.1 * conciseness, 3)

    return {
        "section_coverage": round(section_cov, 3),
        "actionability": round(actionability, 3),
        "relevance_overlap": round(overlap, 3),
        "conciseness": round(conciseness, 3),
        "composite": composite,
    }


def _cos(a: list[float], b: list[float]) -> float:
    try:
        import math
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    except Exception:
        return 0.0


st.set_page_config(page_title="Context Engineering Demo", layout="wide")
st.title("Context Engineering Framework — Live Demo")
st.caption("Write • Select • Compress • Isolate • Optimize")

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    adapter = st.selectbox("Adapter", ["openai", "local-echo"], index=0)
with col2:
    scenario = st.selectbox(
        "Scenario",
        [
            "BTP CAP Service Hardening",
            "Event Mesh Reliability",
            "BTP API Management Governance",
            "Kyma Multi-Region DR",
            "CPI Observability",
            "SAP S/4HANA Cutover Plan",
            "SAP Authorization Redesign (SoD)",
            "SAP IDoc Stabilization & Monitoring",
            "SAP Fiori Rollout Governance",
            "BW/4HANA Integration for Analytics",
            "Plugin Auth Refactor",
            "Research Roadmap (Ingestion • Schema • Backfill)",
            "Customer Support Triage (RAG + Tools)",
        ],
        index=0,
    )
with col3:
    api_key = st.text_input("OpenAI API Key", type="password")

base = Path(__file__).parent.parent
examples = base / "examples" / "agents"
openai_cfg = examples / "code_agent_openai.yaml"
echo_cfg = examples / "code_agent.yaml"

default_queries = {
    "BTP CAP Service Hardening": (
        "Harden a SAP BTP CAP multi-tenant service: define XSUAA scopes/roles, configure Destination service and connectivity, "
        "add rate limits and circuit breakers, and set up blue-green deploy on Cloud Foundry. Provide risks, mitigations, and evidence."
    ),
    "Event Mesh Reliability": (
        "Design Event Mesh reliability: queues/topics, QoS, DLQ strategy, retry/backoff, idempotency keys, and consumer autoscaling. "
        "Include monitoring, alerts, and runbook."
    ),
    "BTP API Management Governance": (
        "Establish BTP API Management governance: spike arrest, quota, API key/JWT policy, IP allowlists, masking, audit logging; "
        "define lifecycle via CTMS with environments and promotion gates."
    ),
    "Kyma Multi-Region DR": (
        "Create a Kyma multi-region DR plan: active/active or active/passive, HPA/VPA, backup/restore cadence, traffic failover, and config drift detection."
    ),
    "CPI Observability": (
        "Implement SAP Integration Suite (CPI) observability: monitoring dashboards, tracing, alerting rules, error categories, and retry/backoff strategy."
    ),
    "SAP S/4HANA Cutover Plan": (
        "Create a cutover plan for an S/4HANA go-live with BW/4HANA integration. "
        "Include transport sequencing, data migration rehearsals, downtime window, backout plan, and hypercare."
    ),
    "SAP Authorization Redesign (SoD)": (
        "Redesign PFCG roles for least privilege and mitigate SoD risks. Include firefighter access, recertification cadence, and audit evidence."
    ),
    "SAP IDoc Stabilization & Monitoring": (
        "Stabilize IDoc interfaces: monitoring dashboards (WE02/WE05), alerting, retry strategy, partner profiles, status handling, and error taxonomy."
    ),
    "SAP Fiori Rollout Governance": (
        "Plan Fiori rollout: catalogs/groups, app governance, launchpad performance (cache buster/CDN), OData throttling, and SSO integration."
    ),
    "BW/4HANA Integration for Analytics": (
        "Design BW/4HANA and S/4 integration via ODP/SLT, with delta handling, data reconciliation controls, and lineage for audit."
    ),
    "Plugin Auth Refactor": (
        "You are building a plugin-based auth layer across 6 microservices with conflicting token formats. "
        "Propose a phased refactor plan, migration steps, and fallback strategy while keeping SLAs. "
        "Include test strategy and rollback triggers."
    ),
    "Research Roadmap (Ingestion • Schema • Backfill)": (
        "Given 200 pages of notes and API docs, propose a 3-phase roadmap to add streaming ingestion, "
        "schema evolution, and backfill with zero downtime. Include risk matrix, tool selection, and acceptance criteria."
    ),
    "Customer Support Triage (RAG + Tools)": (
        "Design a support triage agent that retrieves KB articles, summarizes logs, and suggests actions, with tool routing to ticketing and status APIs."
    ),
}

if "conv_state" not in st.session_state:
    st.session_state.conv_state = None
    st.session_state.turn = 0

query = st.text_area("Query (raw)", value=default_queries[scenario], height=120)
follow_up = st.text_input("Follow-up (optional, next turn)", value="")

c1, c2, c3 = st.columns([1,1,2])
with c1:
    start = st.button("Start")
with c2:
    next_turn = st.button("Next Turn")
with c3:
    reset = st.button("Reset Conversation")

if reset:
    st.session_state.conv_state = None
    st.session_state.turn = 0
    st.rerun()

run = start or next_turn

if run:
    cfg_path = openai_cfg if adapter == "openai" else echo_cfg
    if adapter == "openai" and not api_key:
        st.error("Please provide an OpenAI API Key or switch to local-echo adapter.")
        st.stop()
    # Determine turn input
    effective_query = query if st.session_state.turn == 0 or start else (follow_up or query)
    compare = st.checkbox("Compare with baseline (no context engineering)", value=True)
    with st.spinner("Running simulation…"):
        res = run_agent(
            cfg_path,
            effective_query,
            api_key if adapter == "openai" else None,
            prior_state=st.session_state.conv_state,
        )
        base = None
        if compare:
            base = run_baseline(cfg_path, effective_query, api_key if adapter == "openai" else None)
        st.session_state.conv_state = res.get("state")
        st.session_state.turn += 1

    # Header metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Adapter", res["model"]["name"])
    with m2:
        st.metric("Input Tokens", res["usage"].get("input_tokens", "-"))
    with m3:
        st.metric("Output Tokens", res["usage"].get("output_tokens", "-"))
    with m4:
        st.metric("Summarized?", "Yes" if any(t.get("phase") == "compress" for t in res.get("trace", [])) else "No")

    # Effective Query (sanitized)
    st.subheader("Effective Query (sanitized)")
    qp = next((t for t in res.get("trace", []) if t.get("component", "").startswith("query.")), None)
    if qp:
        before = qp.get("result", {}).get("before_tokens")
        after = qp.get("result", {}).get("after_tokens")
        st.caption(f"Normalized input: {before} → {after} tokens")
        st.code(res.get("effective_query", ""), language=None)
    else:
        st.code(res.get("effective_query", ""), language=None)

    # Professional board-style sections
    st.subheader("Context Pipeline — Storyline")
    for t in res.get("trace", []):
        phase = t.get("phase")
        if phase == "compress" and t.get("component", "").startswith("query."):
            before = t.get("result", {}).get("before_tokens")
            after = t.get("result", {}).get("after_tokens")
            if before and after and after < before:
                st.markdown(f"- Compress: Query reduced from {before} → {after} tokens (removed repetition)")
            else:
                st.markdown("- Compress: Query normalized (no reduction needed)")
        elif phase == "select" and t.get("kind") == "selector":
            added = t.get("added", {})
            ins, fac, tls = len(added.get("instructions", [])), len(added.get("facts", [])), len(added.get("tools", []))
            title = t.get("component")
            st.markdown(f"- Select: {title} added +{ins} instructions, +{fac} facts, +{tls} tools")
        elif phase == "write":
            st.markdown("- Write: Scratchpad updated and plan persisted")
        elif phase == "optimize":
            dec = t.get("decision", {})
            st.markdown(f"- Optimize: summarize={dec.get('should_summarize')} remaining_input={dec.get('remaining_input_budget')}")
        elif phase == "generate":
            st.markdown("- Generate: Model produced response within budget")

    with st.expander("See Detailed Trace (for engineers)"):
        st.json(res.get("trace", []))

    # Select details: what was chosen
    st.subheader("Selected Context")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Instructions**")
        for i, it in enumerate(res["state"]["selected"]["instructions"][:3]):
            st.code(it, language=None)
        if len(res["state"]["selected"]["instructions"]) > 3:
            st.caption(f"(+{len(res['state']['selected']['instructions']) - 3} more)")
    with s2:
        st.markdown("**Facts**")
        for i, it in enumerate(res["state"]["selected"]["facts"][:3]):
            st.code(it, language=None)
        if len(res["state"]["selected"]["facts"]) > 3:
            st.caption(f"(+{len(res['state']['selected']['facts']) - 3} more)")
    with s3:
        st.markdown("**Tools**")
        st.write(", ".join(res["state"]["selected"]["tools"]))

    # Optimizer decision and budgets
    st.subheader("Budgets & Decision")
    opt = next((t for t in res.get("trace", []) if t.get("phase") == "optimize"), None)
    cols = st.columns(3)
    with cols[0]:
        st.metric("Input Tokens (pre-gen)", opt.get("budgets", {}).get("input_tokens") if opt else "-")
    with cols[1]:
        st.metric("Summarize?", str(opt.get("decision", {}).get("should_summarize")) if opt else "-")
    with cols[2]:
        st.metric("Remaining Budget", opt.get("decision", {}).get("remaining_input_budget") if opt else "-")

    # Outputs + Quality comparison
    st.subheader("Outputs")
    if 'base' in locals() and base is not None:
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Baseline (No Context Engineering)**")
            st.caption(f"Tokens: in={base['usage'].get('input_tokens','-')} out={base['usage'].get('output_tokens','-')}")
            st.text(base["output"]) 
        with cB:
            st.markdown("**Context‑Engineered**")
            st.caption(f"Tokens: in={res['usage'].get('input_tokens','-')} out={res['usage'].get('output_tokens','-')}")
            st.text(res["output"])

        st.subheader("Quality Scores (Heuristics)")
        base_scores = score_output(base["output"], None, query_text=effective_query)
        eng_scores = score_output(res["output"], res["state"].get("selected", {}), query_text=effective_query)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Sections (base)", base_scores["section_coverage"])
            st.metric("Sections (engineered)", eng_scores["section_coverage"])
        with c2:
            st.metric("Actionable (base)", base_scores["actionability"])
            st.metric("Actionable (engineered)", eng_scores["actionability"])
        with c3:
            st.metric("Relevant (base)", base_scores["relevance_overlap"])
            st.metric("Relevant (engineered)", eng_scores["relevance_overlap"])
        with c4:
            st.metric("Concise (base)", base_scores["conciseness"])
            st.metric("Concise (engineered)", eng_scores["conciseness"])
        with c5:
            st.metric("Composite (base)", base_scores["composite"])
            st.metric("Composite (engineered)", eng_scores["composite"])

        st.caption("Heuristics: sections={Objectives, Phases, Risks/Mitigations, Evidence/SLAs, Next Steps}; actionability=bullets + verbs; relevance=overlap with selected context; conciseness=<=800 tokens.")
    else:
        st.text(res["output"])
