"""Independent replay audit for the frozen fan-sensing pilot."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from scripts.sensing_baseline_reference import ReferenceBaseline
    from scripts.sensing_reference import NEURAL_ARMS, ReferencePredictor
except ModuleNotFoundError:
    from sensing_baseline_reference import ReferenceBaseline
    from sensing_reference import NEURAL_ARMS, ReferencePredictor

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "sensing_fan_v1.json"
FIXTURE_SHA256 = (
    "b604d5cb2a88f9dcc7278c0f3ffd6e7437ccf798956958283235e8663a31d191")
CSV_SHA256 = (
    "296a94eaf40643e1188012a4ee37d0b2c397bbc04decd39cfbe9952b8c841674")
ARCHIVE_SHA256 = (
    "00557d031124defb8cebc6ae62cb1833e4cadde546c6c2b1997a8925d92816b0")
FIRST_DT = 0.2
ARMS = ("plastic_snn", "readout_snn", "frozen_snn", "plastic_ann",
        "persistence", "ema", "nlms")
SEEDS = (0, 1, 2)
TRAIN_SPEEDS = (20, 35, 50, 65, 80, 95)
TEST_SPEEDS = (25, 30, 40, 45, 55, 60, 70, 75, 85, 90, 100)
TOL = 1e-12


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _close(actual: Any, expected: float, label: str) -> None:
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=TOL,
                                abs_tol=TOL)):
        raise ValueError(f"{label} mismatch")


def _equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} keys mismatch")
        for key, value in expected.items():
            _equal(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if (not isinstance(actual, (list, tuple))
                or len(actual) != len(expected)):
            raise ValueError(f"{label} length mismatch")
        for index, value in enumerate(expected):
            _equal(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise ValueError(f"{label} mismatch")
    elif isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise ValueError(f"{label} mismatch")
    elif isinstance(expected, float):
        _close(actual, expected, label)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label} mismatch")



def load_protocol() -> dict[str, Any]:
    payload = FIXTURE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != FIXTURE_SHA256:
        raise ValueError("sensing protocol hash mismatch")
    return json.loads(payload)


def _load_source(
    protocol: dict[str, Any],
) -> dict[tuple[int, int], list[float]]:
    dataset = protocol["dataset"]
    csv_path = ROOT / dataset["csv"]
    archive_path = ROOT / dataset["archive"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("source archive hash mismatch")
    raw = csv_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("source CSV hash mismatch")
    rows: dict[tuple[int, int], list[float]] = {}
    order: list[tuple[int, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != (
                "wconfid", "pctid", "x", "y", "z"):
            raise ValueError("source columns mismatch")
        for row in reader:
            key = (int(row["wconfid"]), int(row["pctid"]))
            if not order or order[-1] != key:
                if key in rows:
                    raise ValueError("source groups are not contiguous")
                order.append(key)
                rows[key] = []
            value = float(row["x"])
            if not math.isfinite(value):
                raise ValueError("non-finite source observation")
            rows[key].append(value)
    expected = [(c, s) for c in (1, 2, 3) for s in range(20, 101, 5)]
    if order != expected or any(len(v) != 3000 for v in rows.values()):
        raise ValueError("source group layout mismatch")
    return rows


def _windows(values: list[float]) -> list[float]:
    result = []
    for start in range(0, len(values) - 9, 10):
        block = values[start:start + 10]
        result.append(math.sqrt(sum(x * x for x in block) / 10.0))
    return result


def _fit_transform(source: dict[tuple[int, int], list[float]],
                   protocol: dict[str, Any]
                   ) -> tuple[
                       float, float, dict[tuple[int, int], list[float]]]:
    split, inp = protocol["split"], protocol["input"]
    train_lo, train_hi = split["training_raw_range"]
    training = [feature for speed in split["training_speeds"]
                for feature in _windows(source[(1, speed)][train_lo:train_hi])]
    center = statistics.median(training)
    deviations = [abs(value - center) for value in training]
    scale = max(1.4826 * statistics.median(deviations), 1e-6)
    all_windows = {key: _windows(values) for key, values in source.items()}
    normalized = {
        key: [0.5 + 0.5 * (x - center) / (scale + abs(x - center))
              for x in values]
        for key, values in all_windows.items()
    }
    return center, scale, normalized


def _make_sequences(protocol: dict[str, Any],
                    normalized: dict[tuple[int, int], list[float]]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = protocol["split"]
    lo, hi = (split["training_raw_range"][0] // 10,
              split["training_raw_range"][1] // 10)
    clo, chi = (split["calibration_raw_range"][0] // 10,
                split["calibration_raw_range"][1] // 10)
    train = [{"speed": speed, "values": normalized[(1, speed)][lo:hi]}
             for speed in split["training_speeds"]]
    calibration = [{"speed": speed,
                    "values": normalized[(1, speed)][clo:chi]}
                   for speed in split["training_speeds"]]
    return train, calibration


def _stream_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct evaluator-side stream metadata from the pinned fixture."""
    spec = protocol["streams"]
    width = spec["block_samples"]
    rows: list[dict[str, Any]] = []
    for speed in protocol["split"]["test_speeds"]:
        for order_index, order in enumerate(spec["positive_orders"]):
            quarters = spec["positive_quarters"]
            slices = [{"config": config, "speed": speed,
                       "start": width * quarter,
                       "stop": width * (quarter + 1)}
                      for config, quarter in zip(order, quarters)]
            rows.append({"speed": speed, "kind": "positive",
                         "order": order_index, "source_slices": slices,
                         "events": list(spec["positive_events"])})
        for order_index, order in enumerate(spec["sham_orders"]):
            slices = [{"config": spec["sham_configuration"],
                       "speed": speed, "start": width * quarter,
                       "stop": width * (quarter + 1)}
                      for quarter in order]
            rows.append({"speed": speed, "kind": "sham",
                         "order": order_index, "source_slices": slices,
                         "events": []})
    return rows


def _expected_stream_values(
    slices: list[dict[str, int]],
    normalized: dict[tuple[int, int], list[float]],
) -> list[float]:
    result: list[float] = []
    for item in slices:
        lo = item["start"] // 10
        hi = item["stop"] // 10
        result.extend(normalized[(item["config"], item["speed"])][lo:hi])
    return result


def _new_model(arm: str, seed: int) -> ReferencePredictor | ReferenceBaseline:
    if arm in NEURAL_ARMS:
        return ReferencePredictor(seed, arm)
    return ReferenceBaseline(arm)


def _metrics(model: ReferencePredictor | ReferenceBaseline
             ) -> dict[str, int]:
    return dict(model.metrics)


def _counter_delta(before: dict[str, int], after: dict[str, int]
                   ) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


def _train_model(
    arm: str,
    seed: int,
    training: list[dict[str, Any]],
) -> tuple[ReferencePredictor | ReferenceBaseline, list[dict[str, Any]]]:
    model = _new_model(arm, seed)
    sequences = []
    for sequence in training:
        values = sequence["values"]
        if not values:
            raise ValueError("empty training recording")
        windows, tail = _process_values(model, values, None, 5)
        sequences.append({"speed": sequence["speed"], "windows": windows,
                          **tail})
    return model, sequences


def _frozen_from_plastic(
    plastic: ReferencePredictor, seed: int,
) -> ReferencePredictor:
    frozen = ReferencePredictor(seed, "frozen_snn")
    frozen.w, frozen.b = plastic.w[:], plastic.b[:]
    frozen.rw, frozen.ob = plastic.rw[:], plastic.ob
    frozen.metrics = frozen._zero_metrics()
    frozen._reset_state()
    return frozen


def _process_values(
    model: ReferencePredictor | ReferenceBaseline,
    values: list[float],
    threshold: float | None,
    first_scored: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(values) < 6:
        raise ValueError("every sensing sequence needs scored features")
    predictions: list[dict[str, Any]] = []
    last_alarm: int | None = None
    before_metrics = _metrics(model)
    for index, value in enumerate(values):
        timestamp = FIRST_DT * (index + 1)
        before_hash = model.weight_hash()
        if index == 0:
            prediction_before = None
            score = None
            alarm = False
            next_prediction = model.start(timestamp, value)
        else:
            prediction_before = model.pending
            if prediction_before is None:
                raise ValueError("reference predictor lost pending forecast")
            score = abs(value - prediction_before)
            scored = index >= first_scored
            alarm = bool(threshold is not None and scored
                         and score > threshold
                         and (last_alarm is None or index - last_alarm >= 5))
            if alarm:
                last_alarm = index
            next_prediction = model.advance(timestamp, value)
        after_hash = model.weight_hash()
        predictions.append({
            "i": index, "timestamp": timestamp, "value": value,
            "prediction_before": prediction_before, "score": score,
            "scored": index >= first_scored, "alarm": alarm,
            "prediction_next": next_prediction,
            "model_hash_before": before_hash,
            "model_hash_after": after_hash,
        })
    model.end()
    work = _counter_delta(before_metrics, _metrics(model))
    return predictions, {"end_weight_hash": model.weight_hash(),
                         "end_snapshot_hash": _digest(model.snapshot()),
                         "work": work}


def _calibrate_values(
    model: ReferencePredictor | ReferenceBaseline,
    values: list[float],
    first_scored: int,
) -> tuple[list[float], dict[str, Any]]:
    windows, tail = _process_values(model, values, None, first_scored)
    scores = [row["score"] for row in windows if row["scored"]]
    return scores, {"windows": windows, **tail}


def _threshold(scores: list[float], protocol: dict[str, Any]) -> float:
    if not scores:
        raise ValueError("calibration produced no scored values")
    calibration = protocol["calibration"]
    index = math.ceil(calibration["quantile"] * len(scores)) - 1
    floor = float(calibration["threshold"].split("(")[1].split(",")[0])
    return max(floor, sorted(scores)[index])


def _clone_model(model: ReferencePredictor | ReferenceBaseline
                 ) -> ReferencePredictor | ReferenceBaseline:
    result = copy.deepcopy(model)
    if isinstance(result, ReferencePredictor):
        result._reset_state()
    else:
        result.reset_state()
    return result


def _check_fields(actual: Any, expected: dict[str, Any], label: str) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} must be an object")
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f"{label}.{key} missing")
        _equal(actual[key], value, f"{label}.{key}")


def _check_sequences(
    actual: Any,
    expected: list[dict[str, Any]],
    label: str,
) -> None:
    """Compare deterministic sequence evidence and validate measured costs."""
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValueError(f"{label} sequence count mismatch")
    cost_fields = {"processing_cpu_seconds", "provenance_cpu_seconds"}
    for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)):
        item_label = f"{label}[{index}]"
        if not isinstance(actual_item, dict):
            raise ValueError(f"{item_label} must be an object")
        if set(actual_item) != set(expected_item) | cost_fields:
            raise ValueError(f"{item_label} keys mismatch")
        _check_fields(actual_item, expected_item, item_label)
        _check_runtime_costs(actual_item, item_label)


def _hashes_present(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(_hashes_present(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_hashes_present(item))
    elif isinstance(value, str) and len(value) == 64:
        found.add(value)
    return found


def _current_source_manifest() -> dict[str, str]:
    paths: set[Path] = set()
    for directory in ("rsi_programmer", "tests", "scripts", "fixtures"):
        paths.update(path for path in (ROOT / directory).rglob("*")
                     if path.suffix in {".py", ".json", ".js"})
    paths.update(ROOT / name for name in (
        "requirements.txt", "justfile", ".github/workflows/ci.yml"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in sorted(paths)}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("arm"), row.get("seed"), row.get("speed"),
            row.get("kind"), row.get("order"))


def _source_streams(protocol: dict[str, Any],
                    normalized: dict[tuple[int, int], list[float]]) \
        -> list[dict[str, Any]]:
    result = []
    for stream in _stream_specs(protocol):
        result.append({**stream, "values": _expected_stream_values(
            stream["source_slices"], normalized)})
    return result


def _make_training_models(
    protocol: dict[str, Any],
    train_sequences: list[dict[str, Any]],
    records_by_key: dict[tuple[str, int], dict[str, Any]],
) -> dict[tuple[str, int], ReferencePredictor | ReferenceBaseline]:
    trained: dict[tuple[str, int], ReferencePredictor | ReferenceBaseline] = {}
    for seed in protocol["models"]["seeds"]:
        for arm in ("plastic_snn", "readout_snn", "plastic_ann",
                    "persistence", "ema", "nlms"):
            entry = records_by_key[(arm, seed)]
            model, sequences = _train_model(arm, seed, train_sequences)
            expected_entry = {
                "trained_weight_hash": model.weight_hash(),
                "trained_snapshot_hash": _digest(model.snapshot()),
            }
            _check_fields(entry, expected_entry, f"training[{arm},{seed}]")
            _check_sequences(
                entry.get("sequences"), sequences, f"training[{arm},{seed}]")
            trained[(arm, seed)] = model
        plastic = trained[("plastic_snn", seed)]
        if not isinstance(plastic, ReferencePredictor):
            raise ValueError("plastic SNN reference is not neural")
        frozen = _frozen_from_plastic(plastic, seed)
        frozen_entry = records_by_key[("frozen_snn", seed)]
        _check_fields(frozen_entry, {
            "inherited_from": "plastic_snn", "sequences": [],
            "trained_weight_hash": frozen.weight_hash(),
            "trained_snapshot_hash": _digest(frozen.snapshot()),
        }, f"training[frozen_snn,{seed}]")
        trained[("frozen_snn", seed)] = frozen
    return trained


def _audit_calibration(
    protocol: dict[str, Any], calibration_rows: list[dict[str, Any]],
    trained: dict[tuple[str, int], ReferencePredictor | ReferenceBaseline],
    calibration_sequences: list[dict[str, Any]],
) -> dict[tuple[str, int], float]:
    keyed = {(row["arm"], row["seed"]): row for row in calibration_rows}
    if len(keyed) != len(calibration_rows):
        raise ValueError("duplicate calibration arm/seed row")
    if set(keyed) != set(trained):
        raise ValueError(
            "calibration arm/seed matrix is incomplete or has extras")
    thresholds: dict[tuple[str, int], float] = {}
    first_scored = protocol["calibration"]["first_scored_index"]
    for (arm, seed), source_model in trained.items():
        row = keyed[(arm, seed)]
        expected_sequences = []
        all_scores: list[float] = []
        for sequence in calibration_sequences:
            model = _clone_model(source_model)
            scores, tail = _calibrate_values(
                model, sequence["values"], first_scored)
            result = {"speed": sequence["speed"], **tail}
            all_scores.extend(scores)
            expected_sequences.append(result)
        threshold = _threshold(all_scores, protocol)
        _check_fields(row, {"scores": all_scores, "threshold": threshold},
                      f"calibration[{arm},{seed}]")
        _check_sequences(row.get("sequences"), expected_sequences,
                         f"calibration[{arm},{seed}]")
        thresholds[(arm, seed)] = threshold
    return thresholds


def _audit_stream_rows(
    protocol: dict[str, Any],
    stream_rows: list[dict[str, Any]],
    trained: dict[tuple[str, int], ReferencePredictor | ReferenceBaseline],
    thresholds: dict[tuple[str, int], float],
    source_streams: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_count = (len(ARMS) * len(SEEDS)
                      * protocol["streams"]["streams_per_seed_arm"])
    if len(stream_rows) != expected_count:
        raise ValueError("unexpected number of stream rows")
    rows_by_key = {_row_key(row): row for row in stream_rows}
    if len(rows_by_key) != len(stream_rows):
        raise ValueError("duplicate stream identity")
    aggregate: dict[str, Any] = {}
    first_scored = protocol["streams"]["first_scored_index"]
    expected_events = protocol["streams"]["positive_events"]
    for seed in protocol["models"]["seeds"]:
        for arm in protocol["models"]["arms"]:
            alarms_by_kind = {"positive": [], "sham": []}
            events_by_kind = {"positive": [], "sham": []}
            delay_values: list[float] = []
            false_alarms = 0
            positive_stable_windows = 0
            sham_stable_windows = 0
            for source in source_streams:
                kind = source["kind"]
                key = (arm, seed, source["speed"], kind, source["order"])
                row = rows_by_key.get(key)
                if row is None:
                    raise ValueError(f"missing stream row {key}")
                threshold = thresholds[(arm, seed)]
                model = _clone_model(trained[(arm, seed)])
                windows, tail = _process_values(
                    model, source["values"], threshold, first_scored)
                _check_fields(row, {
                    "source_slices": source["source_slices"],
                    "events": source["events"],
                    "threshold": threshold,
                    "windows": windows,
                    **tail,
                }, f"stream[{key}]")
                alarms = [entry["i"] for entry in windows if entry["alarm"]]
                outcome = _metrics_for_stream(windows, source["events"])
                delay_values.extend(outcome["delay_seconds"])
                false_alarms += outcome["false_alarms"]
                if kind == "positive":
                    positive_stable_windows += (
                        300 - first_scored - len(source["events"]) * 5)
                else:
                    sham_stable_windows += 300 - first_scored
                alarms_by_kind[kind].append(len(alarms))
                events_by_kind[kind].append(outcome)
                expected_metrics = _metrics_for_stream(windows,
                                                       source["events"])
                _check_fields(row, {"metrics": expected_metrics},
                              f"stream-metrics[{key}]")
                _check_runtime_costs(row, f"stream[{key}]")
            agg_key = f"{arm}:{seed}"
            total_events = sum(
                item["events"] for item in events_by_kind["positive"])
            total_hits = sum(
                item["detected"] for item in events_by_kind["positive"])
            sham_alarms = sum(alarms_by_kind["sham"])
            aggregate[agg_key] = {
                "positive_streams": len(events_by_kind["positive"]),
                "sham_streams": len(events_by_kind["sham"]),
                "events": total_events, "hits": total_hits,
                "recall": total_hits / total_events if total_events else None,
                "false_alarms": false_alarms,
                "positive_stable_minutes": positive_stable_windows / 300.0,
                "sham_stable_minutes": sham_stable_windows / 300.0,
                "sham_false_alarms_per_minute": (
                    sham_alarms / (sham_stable_windows / 300.0)
                    if sham_stable_windows else None),
                "mean_detection_delay_seconds": (
                    sum(delay_values) / len(delay_values)
                    if delay_values else None),
            }
            if total_events != protocol["evaluation"][
                    "expected_events_per_seed_arm"]:
                raise ValueError("event count does not match fixture")
    return aggregate


def _event_replay(events: list[int], alarms: list[int]) -> dict[str, Any]:
    horizon = 5
    matched: set[int] = set()
    delays: list[float] = []
    for event in events:
        hit = next((alarm for alarm in alarms
                    if event <= alarm < event + horizon
                    and alarm not in matched), None)
        if hit is not None:
            matched.add(hit)
            delays.append(0.2 * (hit + 1 - event))
    false = [alarm for alarm in alarms if alarm not in matched]
    return {"events": len(events), "detected": len(matched),
            "false_alarms": len(false), "delay_seconds": delays}


def _metrics_for_stream(windows: list[dict[str, Any]],
                        events: list[int]) -> dict[str, Any]:
    outcome = _event_replay(events, [row["i"] for row in windows
                                     if row["alarm"]])
    stable = [row for row in windows if row["scored"] and not any(
        event <= row["i"] < event + 5 for event in events)]
    stable_seconds = len(stable) * FIRST_DT
    if stable_seconds <= 0:
        raise ValueError("no stable scored windows")
    scores = [row["score"] for row in windows if row["scored"]]
    outcome.update({
        "stable_seconds": stable_seconds,
        "false_alarms_per_minute": outcome["false_alarms"]
        * 60.0 / stable_seconds,
        "scored_windows": len(scores),
        "squared_error_sum": sum(score * score for score in scores),
    })
    return outcome


def _check_runtime_costs(record: dict[str, Any], label: str) -> None:
    for name in ("processing_cpu_seconds", "provenance_cpu_seconds"):
        value = record.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{label}.{name} is invalid")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["kind"] == "positive"]
    shams = [row for row in rows if row["kind"] == "sham"]
    events = sum(row["metrics"]["events"] for row in positive)
    detected = sum(row["metrics"]["detected"] for row in positive)
    sham_false = sum(row["metrics"]["false_alarms"] for row in shams)
    sham_seconds = sum(row["metrics"]["stable_seconds"] for row in shams)
    targets = sum(row["metrics"]["scored_windows"] for row in rows)
    return {
        "streams": len(rows), "events": events, "detected": detected,
        "recall": detected / events if events else None,
        "sham_false_alarms": sham_false,
        "sham_stable_seconds": sham_seconds,
        "sham_false_alarms_per_minute": (
            sham_false * 60.0 / sham_seconds if sham_seconds else None),
        "positive_false_alarms": sum(
            row["metrics"]["false_alarms"] for row in positive),
        "delay_seconds": [delay for row in positive
                          for delay in row["metrics"]["delay_seconds"]],
        "mse": sum(row["metrics"]["squared_error_sum"] for row in rows)
        / targets if targets else None,
        "processing_cpu_seconds": sum(row["processing_cpu_seconds"]
                                       for row in rows),
        "provenance_cpu_seconds": sum(row["provenance_cpu_seconds"]
                                       for row in rows),
    }


def _expected_summary(stream_rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = list(ARMS)
    expected = {(arm, seed, speed, kind, order) for arm in arms
                for seed in SEEDS for speed in TEST_SPEEDS
                for kind in ("positive", "sham") for order in (0, 1)}
    identities = [_row_key(row) for row in stream_rows]
    if len(identities) != 924 or set(identities) != expected:
        raise ValueError("stream identity matrix is incomplete")
    per_seed, per_speed_seed = [], []
    for arm in arms:
        for seed in SEEDS:
            selected = [row for row in stream_rows
                        if row["arm"] == arm and row["seed"] == seed]
            per_seed.append({"arm": arm, "seed": seed,
                             **_aggregate_rows(selected)})
            for speed in TEST_SPEEDS:
                subset = [row for row in selected if row["speed"] == speed]
                per_speed_seed.append({"arm": arm, "seed": seed,
                                       "speed": speed,
                                       **_aggregate_rows(subset)})
    by_key = {(row["arm"], row["seed"]): row for row in per_seed}
    gates = []
    for seed in SEEDS:
        ours = by_key[("plastic_snn", seed)]
        frozen = by_key[("frozen_snn", seed)]
        ann = by_key[("plastic_ann", seed)]
        practical = (ours["recall"] >= 0.8
                     and ours["sham_false_alarms_per_minute"] <= 1.0)
        eligible = [arm for arm in ("persistence", "ema", "nlms")
                    if by_key[(arm, seed)][
                        "sham_false_alarms_per_minute"] <= 1.0]
        gates.append({
            "seed": seed, "practical": practical,
            "adaptation": practical
            and ours["recall"] >= frozen["recall"] + 0.10
            and ours["sham_false_alarms_per_minute"]
            <= frozen["sham_false_alarms_per_minute"],
            "snn_specific": practical
            and ours["recall"] >= ann["recall"] + 0.10
            and ours["sham_false_alarms_per_minute"]
            <= ann["sham_false_alarms_per_minute"],
            "competitive": practical and all(
                ours["recall"] >= by_key[(arm, seed)]["recall"]
                for arm in eligible),
            "eligible_cheap_baselines": eligible,
        })
    return {
        "stream_rows": len(stream_rows), "per_seed": per_seed,
        "per_speed_seed": per_speed_seed,
        "pooled": [{"arm": arm, **_aggregate_rows(
            [row for row in stream_rows if row["arm"] == arm])}
                   for arm in arms],
        "gates_by_seed": gates,
        "all_seed_gates": {key: all(gate[key] for gate in gates)
                            for key in ("practical", "adaptation",
                                        "snn_specific", "competitive")},
        "device_gate": "unmeasured",
        "interpretation": "Stitched clips from one fan; correlated "
        "repetitions, not independent recording or device replicates. "
        "A competitive gate with no eligible cheap baseline is vacuous.",
        "streams": [{key: row[key] for key in (
            "arm", "seed", "speed", "kind", "order", "metrics",
            "processing_cpu_seconds", "provenance_cpu_seconds")}
                    for row in stream_rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_bundle(args.bundle)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


def audit_bundle(bundle: Path) -> dict[str, Any]:
    """Independently reconstruct and verify every frozen-result row."""
    protocol = load_protocol()
    manifest = _load_json(bundle / "manifest.json")
    found_hashes = _hashes_present(manifest)
    for name, digest in (("protocol", FIXTURE_SHA256),
                         ("source CSV", CSV_SHA256),
                         ("source archive", ARCHIVE_SHA256)):
        if digest not in found_hashes:
            raise ValueError(f"manifest does not pin {name} SHA-256")
    if manifest.get("source_files") != _current_source_manifest():
        raise ValueError("source manifest does not match current source tree")
    source = _load_source(protocol)
    center, scale, normalized = _fit_transform(source, protocol)
    training_sequences, calibration_sequences = _make_sequences(
        protocol, normalized)
    training_rows = _load_json(bundle / "training.json")
    calibration_rows = _load_json(bundle / "calibration.json")
    stream_rows = read_jsonl_gz(bundle / "streams.jsonl.gz")
    summary_row = _load_json(bundle / "summary.json")
    if (not isinstance(training_rows, list)
            or not isinstance(calibration_rows, list)):
        raise ValueError("training/calibration files must contain JSON arrays")
    training_by_key = {(row["arm"], row["seed"]): row
                       for row in training_rows}
    if len(training_by_key) != len(training_rows):
        raise ValueError("duplicate training arm/seed row")
    expected_keys = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(training_by_key) != expected_keys:
        raise ValueError("training matrix is incomplete or contains extras")
    trained = _make_training_models(
        protocol, training_sequences, training_by_key)
    thresholds = _audit_calibration(
        protocol, calibration_rows, trained, calibration_sequences)
    stream_specs = _source_streams(protocol, normalized)
    aggregate = _audit_stream_rows(
        protocol, stream_rows, trained, thresholds, stream_specs)
    expected_summary = _expected_summary(stream_rows)
    _check_fields(summary_row, expected_summary, "summary")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest file hashes are missing")
    artifact_names = ("training.json", "calibration.json",
                      "streams.jsonl.gz", "summary.json")
    if set(files) != set(artifact_names):
        raise ValueError(
            "manifest artifact hash set is incomplete or has extras")
    for name in artifact_names:
        actual_hash = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if files.get(name) != actual_hash:
            raise ValueError(f"artifact hash mismatch for {name}")
    git_head = manifest.get("git_head")
    if (not isinstance(git_head, str) or len(git_head) != 40
            or any(char not in "0123456789abcdef" for char in git_head)):
        raise ValueError("manifest git_head is not a full commit hash")
    expected_cost_scope = (
        "Processing covers model, residual and alarm calls, including "
        "session reset. Provenance hashes are separate. Full process "
        "includes data/copies/evaluator/output/provenance; no device-energy "
        "measurement.")
    if manifest.get("cost_scope") != expected_cost_scope:
        raise ValueError("manifest cost_scope does not match runner contract")
    sequence_records = [
        sequence
        for record in training_rows + calibration_rows
        for sequence in record["sequences"]
    ]
    measured_cpu = sum(
        record[name] for record in sequence_records + stream_rows
        for name in ("processing_cpu_seconds", "provenance_cpu_seconds"))
    prep_cpu = manifest.get("data_preparation_cpu_seconds")
    process_cpu = manifest.get("full_process_cpu_seconds")
    wall_seconds = manifest.get("wall_seconds")
    for name, value in (("data preparation CPU", prep_cpu),
                        ("full process CPU", process_cpu),
                        ("wall time", wall_seconds)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"manifest {name} is invalid")
    if process_cpu + TOL < prep_cpu + measured_cpu:
        raise ValueError("full process CPU is below its measured components")
    _check_fields(manifest, {
        "dataset": protocol["dataset"],
        "normalization": {
            "center": center, "scale": scale,
            "training_feature_count": 900,
            "fit_groups": [[1, speed, 0, 1500] for speed in TRAIN_SPEEDS],
        },
        "stream_rows": protocol["evaluation"]["expected_stream_rows"],
        "windows": protocol["evaluation"]["expected_stream_rows"] * 300,
        "source_unchanged": True, "trained_sources_unchanged": True,
    }, "manifest")
    _check_fields(manifest, {
        "protocol_sha256": FIXTURE_SHA256,
        "stream_rows": len(stream_rows),
    }, "manifest")
    return {
        "protocol_sha256": FIXTURE_SHA256,
        "csv_sha256": CSV_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "source_groups": len(source),
        "normalization_center": center,
        "normalization_scale": scale,
        "training_sequences": len(training_sequences),
        "calibration_sequences_per_arm_seed": len(calibration_sequences),
        "stream_rows": len(stream_rows),
        "aggregate": aggregate,
        "summary_verified": True,
        "scope": "Independent replay of recorded software outputs; no "
        "physical transition or device-energy claim",
    }


if __name__ == "__main__":
    raise SystemExit(main())
