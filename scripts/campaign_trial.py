"""Run one bounded construction campaign trial from a frozen spec."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from rsi_programmer.campaign_worker import run_campaign


def _json(value_or_path: str | Path) -> dict[str, Any]:
    path = Path(value_or_path)
    try:
        raw = path.read_text() if path.is_file() else str(value_or_path)
        value = json.loads(raw)
    except OSError:
        value = json.loads(str(value_or_path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--baseline-recipe")
    parser.add_argument("--selected-recipe")
    args = parser.parse_args()
    spec = _json(args.spec)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.audit:
            if not args.baseline_recipe or not args.selected_recipe:
                raise ValueError("audit requires both frozen recipe files")
            result = run_campaign(
                spec, root, audit=True,
                baseline_recipe=_json(args.baseline_recipe),
                selected_recipe=_json(args.selected_recipe))
        else:
            result = run_campaign(spec, root)
    except Exception as error:
        result = {"status": "infrastructure_failure",
                  "phase": "audit" if args.audit else "development",
                  "error_type": type(error).__name__,
                  "error": str(error)[:2000]}
        (root / "result.json").write_text(
            json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 2
    print(json.dumps({"status": result["status"],
                      "phase": result["phase"],
                      "development": result.get("development"),
                      "verdict": result.get("verdict")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
