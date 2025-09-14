from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import StreamingResponse
from pathlib import Path
import yaml

import sys
BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cef.runtime import Orchestrator  # type: ignore
from cef.llm import ADAPTERS  # type: ignore



def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_agent(cfg_path: Path, query: str, api_key: Optional[str], prior_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = load_yaml(cfg_path)
    # resolve base dir
    p = cfg_path.resolve()
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


def run_baseline(cfg_path: Path, query: str, api_key: Optional[str]) -> Dict[str, Any]:
    cfg = load_yaml(cfg_path)
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


app = FastAPI(title="Context Engineering Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/api/examples")
def examples():
    scenarios = {
        "BTP CAP Service Hardening": "Harden a SAP BTP CAP multi-tenant service: define XSUAA scopes/roles, configure Destination service and connectivity, add rate limits and circuit breakers, and set up blue-green deploy on Cloud Foundry. Provide risks, mitigations, and evidence.",
        "Event Mesh Reliability": "Design Event Mesh reliability: queues/topics, QoS, DLQ strategy, retry/backoff, idempotency keys, and consumer autoscaling. Include monitoring, alerts, and runbook.",
        "BTP API Management Governance": "Establish BTP API Management governance: spike arrest, quota, API key/JWT policy, IP allowlists, masking, audit logging; define lifecycle via CTMS with environments and promotion gates.",
        "Kyma Multi-Region DR": "Create a Kyma multi-region DR plan: active/active or active/passive, HPA/VPA, backup/restore cadence, traffic failover, and config drift detection.",
        "CPI Observability": "Implement SAP Integration Suite (CPI) observability: monitoring dashboards, tracing, alerting rules, error categories, and retry/backoff strategy.",
    }
    return {"scenarios": scenarios}


@app.post("/api/run")
def api_run(
    payload: Dict[str, Any] = Body(...),
    x_openai_key: Optional[str] = Header(default=None)
):
    adapter = payload.get("adapter", "openai")
    compare = bool(payload.get("compare", True))
    query = payload.get("query", "")
    prior_state = payload.get("prior_state")
    cfg_path = BASE / "examples" / "agents" / ("code_agent_openai.yaml" if adapter == "openai" else "code_agent.yaml")
    api_key = payload.get("api_key") or x_openai_key
    try:
        res = run_agent(cfg_path, query, api_key, prior_state)
        base = run_baseline(cfg_path, query, api_key) if compare else None
        return {"engineered": res, "baseline": base}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/run_stream")
def api_run_stream(
    payload: Dict[str, Any] = Body(...),
    x_openai_key: Optional[str] = Header(default=None)
):
    adapter = payload.get("adapter", "openai")
    compare = bool(payload.get("compare", True))
    query = payload.get("query", "")
    prior_state = payload.get("prior_state")
    cfg_path = BASE / "examples" / "agents" / ("code_agent_openai.yaml" if adapter == "openai" else "code_agent.yaml")
    api_key = payload.get("api_key") or x_openai_key

    def iter_chunks():
        try:
            res = run_agent(cfg_path, query, api_key, prior_state)
            # send meta first
            import json, time
            meta = {"type":"meta","engineered": {k: v for k, v in res.items() if k != "output"}}
            yield json.dumps(meta) + "\n"
            # stream output in chunks
            out = res.get("output", "")
            step = 60
            for i in range(0, len(out), step):
                chunk = out[i:i+step]
                yield json.dumps({"type":"delta","text": chunk}) + "\n"
                time.sleep(0.02)
            if compare:
                base = run_baseline(cfg_path, query, api_key)
            else:
                base = None
            yield json.dumps({"type":"done","baseline": base}) + "\n"
        except Exception as e:
            import json
            yield json.dumps({"type":"error","error": str(e)}) + "\n"

    return StreamingResponse(iter_chunks(), media_type="application/json")


WEB_DIR = BASE / "web"


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/static/{path:path}")
def static_files(path: str):
    file_path = WEB_DIR / path
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "Not found"})
