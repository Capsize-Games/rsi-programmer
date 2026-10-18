"""Describe completed frozen outputs without rerunning or fitting models."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    """Retain exact descriptive contrasts and source-specific scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous descriptive evidence")
    summary = json.loads((args.bundle / "summary.json").read_text())
    pooled = {row["arm"]: row for row in summary["pooled"]}
    snn = pooled["plastic_snn"]
    frozen = pooled["frozen_snn"]
    readout = pooled["readout_snn"]
    ann = pooled["plastic_ann"]
    ema = pooled["ema"]
    contrasts = {
        "online_mse_reduction_vs_frozen": 1 - snn["mse"] / frozen["mse"],
        "encoder_mse_reduction_vs_readout": (
            1 - snn["mse"] / readout["mse"]),
        "encoder_processing_cpu_ratio": snn["processing_cpu_seconds"]
        / readout["processing_cpu_seconds"],
        "snn_vs_ann_recall_difference": snn["recall"] - ann["recall"],
        "snn_vs_ann_sham_alarm_difference": (
            snn["sham_false_alarms_per_minute"]
            - ann["sham_false_alarms_per_minute"]),
        "snn_vs_ema_mse_ratio": snn["mse"] / ema["mse"],
        "snn_vs_ema_processing_cpu_ratio": snn["processing_cpu_seconds"]
        / ema["processing_cpu_seconds"],
    }
    counters: dict[str, dict[str, int]] = {}
    with gzip.open(args.bundle / "streams.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            target = counters.setdefault(row["arm"], {})
            for name, count in row["work"].items():
                if type(count) is not int:
                    raise ValueError("work counts must be integers")
                target[name] = target.get(name, 0) + count
    per_speed = [{key: row[key] for key in (
        "arm", "seed", "speed", "detected", "events", "recall",
        "sham_false_alarms", "sham_false_alarms_per_minute")}
        for row in summary["per_speed_seed"] if row["arm"] == "plastic_snn"]
    result: dict[str, Any] = {
        "analysis": "Descriptive post-run contrasts; no fitting or tuning",
        "summary_sha256": hashlib.sha256(
            (args.bundle / "summary.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (args.bundle / "manifest.json").read_bytes()).hexdigest(),
        "contrasts": contrasts, "pooled_work": counters,
        "plastic_snn_by_speed_seed": per_speed,
        "all_seed_gates": summary["all_seed_gates"],
        "calibration_design_concern": {
            "window_hz": 5, "quantile": 0.995,
            "ideal_continuous_iid_raw_exceedances_per_minute": (
                5 * 60 * (1 - 0.995)),
            "ideal_iid_alarm_rate_with_five_window_cooldown": (
                300 * 0.005 / (1 + 4 * 0.005)),
            "required_sham_alarms_per_minute": 1,
            "scope": "An idealized design check, not an estimate of this "
            "correlated source or proof of the observed failure's cause. "
            "The frozen operating point is preserved.",
        },
        "scope": "Repeated clips and seeds are correlated. CPU is "
        "software model/detector processing, not device power or latency. "
        "No independent physical transitions, faults or chip measurement.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)
                           + "\n")
    print(json.dumps(contrasts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
