from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from .runtime import Orchestrator


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    # Light validation (schema validation can be added by user stack)
    required = ["name", "model", "pipeline", "policies"]
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"Missing required keys: {missing}")
        return 2
    print("Config looks structurally valid.")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    orch = Orchestrator(cfg)
    # attach base_dir for relative paths resolution
    # Resolve project base dir (parent of 'examples' if present; else config's parent)
    p = Path(args.config).resolve()
    base = p.parent
    for anc in p.parents:
        if anc.name == "examples":
            base = anc.parent
            break
    cfg["__base_dir__"] = str(base)
    orch = Orchestrator(cfg)
    res = orch.run_once(query=args.query)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=== Output ===")
        print(res["output"])  # noqa
        print("\n=== Usage ===")
        print(res["usage"])  # noqa
        print("\n=== Logs ===")
        for line in res["logs"]:
            print("-", line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cef")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate an agent config")
    p_val.add_argument("config", help="Path to YAML config")
    p_val.set_defaults(func=cmd_validate)

    p_sim = sub.add_parser("simulate", help="Simulate a single run")
    p_sim.add_argument("config", help="Path to YAML config")
    p_sim.add_argument("--query", required=True, help="User query")
    p_sim.add_argument("--json", action="store_true", help="Output JSON")
    p_sim.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
