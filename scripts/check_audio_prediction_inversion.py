"""Retrospectively score prediction complements without updating models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rsi_programmer.audio_features import normalize_bands
from rsi_programmer.sensing_predictor import SensingPredictor

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("adaptive_snn", "fixed_snn", "adaptive_ann", "ema", "persistence")


def _empty() -> dict[str, dict[str, float | int]]:
    return {arm: {"original_sse": 0.0, "flipped_sse": 0.0,
                  "both_flipped_sse": 0.0, "targets": 0} for arm in ARMS}


def _score(totals: dict[str, Any], arm: str, prediction: list[float],
           target: list[float]) -> None:
    if len(prediction) != 6 or len(target) != 6:
        raise ValueError("expected six prediction and target bands")
    values = [*prediction, *target]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("prediction and target must be bounded and finite")
    row = totals[arm]
    row["original_sse"] += sum((p - y) ** 2
                               for p, y in zip(prediction, target))
    row["flipped_sse"] += sum((1.0 - p - y) ** 2
                              for p, y in zip(prediction, target))
    row["both_flipped_sse"] += sum(((1.0 - p) - (1.0 - y)) ** 2
                                   for p, y in zip(prediction, target))
    row["targets"] += 6


def _summarize(totals: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm, row in totals.items():
        count = row["targets"]
        if not count:
            raise ValueError("no scored targets")
        original = row["original_sse"] / count
        flipped = row["flipped_sse"] / count
        both = row["both_flipped_sse"] / count
        if not math.isclose(original, both, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("complementing both must preserve squared error")
        result[arm] = {
            **row, "original_mse": original, "flipped_prediction_mse": flipped,
            "both_flipped_mse": both,
            "flipped_prediction_mse_reduction_percent": (
                100 * (original - flipped) / original if original else None),
        }
    return result


def measure(session: Path) -> dict[str, Any]:
    """Use the original private trace and checkpoint-30 parameter copies."""
    spec_path = ROOT / "fixtures/audio_output_inversion_v1.json"
    spec = json.loads(spec_path.read_text())
    published = json.loads((ROOT / "artifacts/audio-live-session-20261018"
                            / "minute-030/report.json").read_text())
    rows = []
    digest = hashlib.sha256()
    with (session / "feature_trace.jsonl").open("rb") as stream:
        for index in range(spec["prefix_frames"]):
            raw = stream.readline()
            if not raw.endswith(b"\n"):
                raise ValueError("original trace prefix is incomplete")
            digest.update(raw)
            row = json.loads(raw)
            if row["i"] != index:
                raise ValueError("original trace index mismatch")
            rows.append(row)
    live = _empty()
    for position, row in enumerate(rows[400:]):
        if row["phase"] != "monitor" or row["j"] != position:
            raise ValueError("monitor sequence mismatch")
        if position < 25:
            continue
        for arm in ARMS:
            item = row["arms"][arm]
            if not item["scored"]:
                raise ValueError("unexpected unscored monitor row")
            _score(live, arm, item["prediction_before"], row["normalized"])
    live_result = _summarize(live)
    for arm in ARMS:
        expected = published["selected_cumulative"]["arms"][arm]["mse"]
        if not math.isclose(live_result[arm]["original_mse"], expected,
                            rel_tol=1e-12):
            raise ValueError("original live MSE does not reproduce report")

    snapshot_path = session / "snapshots/monitor-000009000.json"
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)["snapshot"]
    normalizer = snapshot["normalizer"]
    probe = [normalize_bands(row["features"]["band_dbfs"], normalizer)
             for row in rows[300:400]]
    models: dict[str, list[SensingPredictor]] = {}
    before: dict[str, list[str]] = {}
    for arm in ("adaptive_snn", "fixed_snn", "adaptive_ann"):
        bank = snapshot["models"][arm]
        models[arm] = []
        for band, saved in enumerate(bank["bands"]):
            model = SensingPredictor(
                seed=bank["seed"] + band, spiking=bank["spiking"],
                learning=False, encoder_learning=False)
            model.restore(saved)
            models[arm].append(model)
        before[arm] = [m.weight_hash() for m in models[arm]]
    totals = _empty()
    prior: dict[str, list[float]] = {}
    alpha = 1.0 - math.exp(-0.2 / 5.0)
    for position, target in enumerate(probe):
        if position >= 25:
            for arm in ARMS:
                _score(totals, arm, prior[arm], target)
        for arm in ARMS:
            if arm in models:
                prior[arm] = [
                    model.start_session(0.2, value) if position == 0
                    else model.advance((position + 1) * 0.2, value)
                    for model, value in zip(models[arm], target)]
            elif arm == "persistence" or position == 0:
                prior[arm] = target[:]
            else:
                prior[arm] = [alpha * y + (1 - alpha) * p
                              for y, p in zip(target, prior[arm])]
    result = _summarize(totals)
    expected_probe = published["selected_checkpoints"][-1]["arms"]
    for arm in ARMS:
        if not math.isclose(result[arm]["original_mse"],
                            expected_probe[arm]["mse"], rel_tol=1e-12):
            raise ValueError("original probe MSE does not reproduce report")
    for arm, bank in models.items():
        if [m.weight_hash() for m in bank] != before[arm]:
            raise ValueError("evaluation changed parameter hashes")
        for model in bank:
            model.end_session()
    return {
        "scope": "Retrospective first30 minutes and checkpoint30 probe",
        "protocol_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "original_trace_prefix_sha256": digest.hexdigest(),
        "private_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "live": live_result, "sealed_probe": result,
        "probe_training": False, "parameters_unchanged": True,
        "original_metrics_reproduce_published_report": True,
        "both_complemented_mse_invariant": True,
        "interpretation": "A chart sign flip changes no predictions. "
                          "Prediction complements need separate confirmation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve an existing diagnostic")
    result = measure(args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("live", "sealed_probe")},
                     indent=2))


if __name__ == "__main__":
    main()
