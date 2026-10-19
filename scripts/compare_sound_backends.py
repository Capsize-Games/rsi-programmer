"""Compare persistent SNN and nearest-example sound memories.

The checked-in protocol fixes synthetic training examples, held-out probes,
stages, and unknown controls before any model is run. Reports contain only
aggregate counts and timings; audio and feature vectors remain in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from scripts.check_sound_learning_demo import (
    DURATION,
    LABEL_ACTIONS,
    SAMPLE_RATE,
    make_sound,
)
from rsi_programmer.sound_features import extract_sound


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "fixtures" / "sound_comparison_v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "1f0be06f30f3010b1cd75d10fab4c8c676ab137c56f1e2dd046cfe4111839b83"
)
LABEL_ORDER = ("bell", "knock", "whistle", "chime")


def canonical_json(value: Any) -> bytes:
    """Encode JSON data deterministically for the frozen protocol digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def protocol_digest(protocol: dict[str, Any]) -> str:
    """Hash the protocol without its self-referential digest field."""
    payload = dict(protocol)
    payload.pop("protocol_sha256", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Read and validate the checked-in comparison schedule."""
    with Path(path).open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    validate_protocol(protocol)
    digest = protocol_digest(protocol)
    if (digest != EXPECTED_PROTOCOL_SHA256
            or protocol.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256):
        raise ValueError("protocol SHA-256 differs from the frozen version")
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject changed labels, schedules, seed overlap, or stage counts."""
    if protocol.get("name") != "desktop-creature-sound-backend-comparison-v1":
        raise ValueError("unexpected protocol name")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported protocol schema")
    if protocol.get("seed") != 0:
        raise ValueError("protocol seed differs")
    generator = protocol.get("generator", {})
    if (generator.get("module") != "scripts.check_sound_learning_demo"
            or generator.get("function") != "make_sound"
            or generator.get("sample_rate") != SAMPLE_RATE
            or generator.get("duration_seconds") != DURATION
            or generator.get("families") != list(LABEL_ORDER)):
        raise ValueError("sound generator configuration differs")
    labels = protocol.get("labels")
    if not isinstance(labels, dict) or tuple(labels) != LABEL_ORDER:
        raise ValueError("label order or label set differs from protocol")
    for label, offset in zip(LABEL_ORDER, (100, 200, 300, 400)):
        entry = labels[label]
        if entry.get("action") != LABEL_ACTIONS[label]:
            raise ValueError(f"unexpected action for {label}")
        if entry.get("training_seeds") != list(range(offset, offset + 3)):
            raise ValueError(f"unexpected training seeds for {label}")
    stages = protocol.get("stages")
    expected_stages = [
        ("initial_unknown", []),
        ("three_labels", ["bell", "knock", "whistle"]),
        ("four_labels", list(LABEL_ORDER)),
        ("after_reload", list(LABEL_ORDER)),
    ]
    if not isinstance(stages, list) or len(stages) != len(expected_stages):
        raise ValueError("protocol stage count differs")
    for stage, (stage_id, expected_labels) in zip(stages, expected_stages):
        if stage.get("id") != stage_id or stage.get("labels") != expected_labels:
            raise ValueError("protocol stage sequence differs")
    if stages[-1].get("reload") is not True:
        raise ValueError("last protocol stage must reload persistent memory")
    probes = protocol.get("probes", {})
    if probes.get("families") != list(LABEL_ORDER):
        raise ValueError("probe family order differs")
    if probes.get("parameter_seeds") != [0, 1, 2]:
        raise ValueError("probe parameter seeds differ")
    if probes.get("replicates_per_parameter_seed") != 4:
        raise ValueError("probe replicate count differs")
    if probes.get("first_seed") != 10000:
        raise ValueError("probe seed range differs")
    if probes.get("family_seed_stride") != 100:
        raise ValueError("probe family seed stride differs")
    controls = probes.get("controls", {})
    if set(controls) != {
        "silence", "noise", "far_tone_low", "far_tone_high"
    }:
        raise ValueError("unknown-control schedule differs")
    if (controls["silence"].get("kind") != "silence"
            or controls["noise"] != {
                "kind": "noise", "seed": 19000, "sigma": 0.04
            }
            or controls["far_tone_low"] != {
                "kind": "pure_tone", "seed": 19100,
                "frequency_hz": 5000, "gain": 0.35
            }
            or controls["far_tone_high"] != {
                "kind": "pure_tone", "seed": 19200,
                "frequency_hz": 6200, "gain": 0.35
            }):
        raise ValueError("unknown-control parameters differ")
    evaluation = protocol.get("evaluation", {})
    if (evaluation.get("aggregate_only") is not True
            or evaluation.get("queries_are_read_only") is not True
            or evaluation.get("same_family_queries_reused_at_each_stage")
            is not True
            or evaluation.get("minimum_queries_per_family") != 12
            or evaluation.get("allow_nan") is not False):
        raise ValueError("protocol must require aggregate-only reports")
    train_seeds = {
        seed for item in labels.values() for seed in item["training_seeds"]
    }
    probe_seeds = {case["seed"] for case in probe_cases(protocol)}
    if train_seeds & probe_seeds:
        raise ValueError("training and probe seeds must be disjoint")
    if len(probe_seeds) != len(LABEL_ORDER) * 12:
        raise ValueError("protocol must define twelve probes per family")


def probe_cases(protocol: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield deterministic family probes in their fixed evaluation order."""
    probes = protocol["probes"]
    start = probes["first_seed"]
    stride = probes["family_seed_stride"]
    replicates = probes["replicates_per_parameter_seed"]
    for family_index, family in enumerate(probes["families"]):
        family_start = start + family_index * stride
        for parameter_seed in probes["parameter_seeds"]:
            for replicate in range(replicates):
                seed = family_start + parameter_seed * replicates + replicate
                yield {
                    "family": family,
                    "expected": family,
                    "parameter_seed": parameter_seed,
                    "replicate": replicate,
                    "seed": seed,
                }


def stage_probe_sequences(
    protocol: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    """Return the fixed probe seed sequence assigned to every stage."""
    seeds = tuple(case["seed"] for case in probe_cases(protocol))
    return {stage["id"]: seeds for stage in protocol["stages"]}


def _make_control(name: str, control: dict[str, Any]) -> dict[str, Any]:
    """Generate one deterministic non-family input descriptor."""
    count = round(SAMPLE_RATE * DURATION)
    kind = control["kind"]
    if kind == "silence":
        samples = [0.0] * count
    elif kind == "noise":
        rng = random.Random(control["seed"])
        samples = [rng.gauss(0.0, control["sigma"]) for _ in range(count)]
    elif kind == "pure_tone":
        frequency = control["frequency_hz"]
        gain = control["gain"]
        phase_rng = random.Random(control["seed"])
        phase = phase_rng.uniform(0.0, 2.0 * math.pi)
        samples = [gain * math.sin(
            2.0 * math.pi * frequency * index / SAMPLE_RATE + phase
        ) for index in range(count)]
    else:
        raise ValueError(f"unsupported control kind: {kind}")
    return extract_sound(samples, SAMPLE_RATE)


def _outcome(expected: str | None, recognized: str | None) -> str:
    if recognized is None:
        return "rejected"
    return "correct" if expected == recognized else "wrong"


def _count_result(
    counts: dict[str, int], expected: str | None, recognized: str | None
) -> None:
    counts[_outcome(expected, recognized)] += 1


def _new_stage_counts() -> dict[str, Any]:
    return {
        "snn": {key: 0 for key in ("correct", "wrong", "rejected")},
        "matcher": {key: 0 for key in ("correct", "wrong", "rejected")},
        "per_class": {},
        "queries": 0,
        "snn_inference_wall_seconds": 0.0,
        "matcher_inference_wall_seconds": 0.0,
        "paired_record_wall_seconds": 0.0,
        "controls": {},
    }


def _record_known_probes(
    memory: Any,
    stage: dict[str, Any],
    descriptors: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    for case, descriptor in descriptors:
        snn_started = time.perf_counter()
        snn_result = memory.recognize(descriptor)
        stage["snn_inference_wall_seconds"] += (
            time.perf_counter() - snn_started
        )
        matcher_started = time.perf_counter()
        baseline_result = memory.recognize_baseline(descriptor)
        stage["matcher_inference_wall_seconds"] += (
            time.perf_counter() - matcher_started
        )
        record_started = time.perf_counter()
        memory.record_probe(case["expected"], descriptor)
        stage["paired_record_wall_seconds"] += (
            time.perf_counter() - record_started
        )
        snn_label = snn_result.get("label")
        baseline_label = baseline_result.get("label")
        class_counts = stage["per_class"].setdefault(case["family"], {
            "snn": {key: 0 for key in ("correct", "wrong", "rejected")},
            "matcher": {key: 0 for key in ("correct", "wrong", "rejected")},
            "queries": 0,
        })
        _count_result(stage["snn"], case["expected"], snn_label)
        _count_result(stage["matcher"], case["expected"], baseline_label)
        _count_result(class_counts["snn"], case["expected"], snn_label)
        _count_result(class_counts["matcher"], case["expected"], baseline_label)
        stage["queries"] += 1
        class_counts["queries"] += 1


def _recognize_unknown(
    memory: Any,
    stage: dict[str, Any],
    name: str,
    descriptor: dict[str, Any],
) -> None:
    snn_started = time.perf_counter()
    snn_result = memory.recognize(descriptor)
    stage["snn_inference_wall_seconds"] += (
        time.perf_counter() - snn_started
    )
    matcher_started = time.perf_counter()
    baseline_result = memory.recognize_baseline(descriptor)
    stage["matcher_inference_wall_seconds"] += (
        time.perf_counter() - matcher_started
    )
    stage["controls"][name] = {
        "snn_false_accept": snn_result.get("label") is not None,
        "snn_recognized": snn_result.get("label"),
        "snn_reason": snn_result.get("reason"),
        "matcher_false_accept": baseline_result.get("label") is not None,
        "matcher_recognized": baseline_result.get("label"),
        "matcher_reason": baseline_result.get("reason"),
    }


def _model_hash(summary: dict[str, Any]) -> str | None:
    learning = summary.get("learning", {})
    if not isinstance(learning, dict):
        return None
    value = learning.get("model_hash")
    return value if isinstance(value, str) else None


def _safe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate learning metrics while excluding private detail."""
    allowed = ("labels", "total_examples", "probe_count", "revision",
               "fingerprint", "label_fingerprints")
    result = {key: summary[key] for key in allowed if key in summary}
    learning = summary.get("learning")
    if isinstance(learning, dict):
        result["learning"] = _aggregate_metrics(learning)
        if isinstance(learning.get("model_hash"), str):
            result["learning"]["model_hash"] = learning["model_hash"]
    return result


def _aggregate_metrics(value: Any) -> Any:
    """Keep numeric aggregate metrics and discard strings, paths, and arrays."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return _sanitize(value)
    if isinstance(value, dict):
        return {str(key): _aggregate_metrics(item)
                for key, item in value.items()
                if isinstance(item, (dict, int, float, bool)) or item is None}
    if isinstance(value, list):
        return [_aggregate_metrics(item) for item in value
                if isinstance(item, (dict, int, float, bool)) or item is None]
    return None


def _sanitize(value: Any) -> Any:
    """Convert values to strict JSON primitives and drop non-finite floats."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _record_unknown_family(
    memory: Any,
    stage: dict[str, Any],
    case: dict[str, Any],
    descriptor: dict[str, Any],
) -> None:
    _recognize_unknown(
        memory, stage, f"family_{case['family']}", descriptor
    )
    item = stage["unknown_family_controls"].setdefault(case["family"], {
        "queries": 0,
        "snn_false_accepts": 0,
        "matcher_false_accepts": 0,
    })
    result = stage["controls"].pop(f"family_{case['family']}")
    item["queries"] += 1
    item["snn_false_accepts"] += int(result["snn_false_accept"])
    item["matcher_false_accepts"] += int(result["matcher_false_accept"])


def _summary_hashes(summary: dict[str, Any]) -> dict[str, str | None]:
    return {
        "model": _model_hash(summary),
        "examples": summary.get("fingerprint"),
    }


def _run_seed_comparison(
    protocol: dict[str, Any],
    memory_path: Path,
    model_seed: int,
    probe_descriptors: list[tuple[dict[str, Any], dict[str, Any]]],
    controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run all stages for one independently seeded SNN and matcher pair."""
    from rsi_programmer.sound_comparison import SoundComparison

    memory = SoundComparison(memory_path, seed=model_seed)
    train = protocol["labels"]
    report: dict[str, Any] = {
        "model_seed": model_seed,
        "stages": [],
        "teaching": {
            "wall_seconds": 0.0,
            "snn_replay_fit_wall_seconds": 0.0,
            "matcher_and_persistence_wall_seconds": 0.0,
            "examples_by_label": {},
        },
    }
    old_label_fingerprints: dict[str, str] = {}

    def teach(label: str) -> None:
        entry = train[label]
        for seed in entry["training_seeds"]:
            descriptor = extract_sound(make_sound(label, seed), SAMPLE_RATE)
            before = memory.summary().get("learning", {}).get(
                "train_seconds", 0.0
            )
            started = time.perf_counter()
            memory.teach(label, descriptor, entry["action"])
            elapsed = time.perf_counter() - started
            after = memory.summary().get("learning", {}).get(
                "train_seconds", before
            )
            snn_fit = max(0.0, float(after) - float(before))
            report["teaching"]["wall_seconds"] += elapsed
            report["teaching"]["snn_replay_fit_wall_seconds"] += snn_fit
            report["teaching"]["matcher_and_persistence_wall_seconds"] += max(
                0.0, elapsed - snn_fit
            )
            counts = report["teaching"]["examples_by_label"]
            counts[label] = counts.get(label, 0) + 1

    def evaluate(
        stage_id: str,
        known_labels: set[str],
        unknown_families: set[str] | None = None,
        include_controls: bool = False,
    ) -> None:
        stage = _new_stage_counts()
        stage["unknown_family_controls"] = {}
        before = _summary_hashes(memory.summary())
        for case, descriptor in probe_descriptors:
            if case["family"] in known_labels:
                _record_known_probes(memory, stage, [(case, descriptor)])
            elif unknown_families and case["family"] in unknown_families:
                _record_unknown_family(memory, stage, case, descriptor)
        if include_controls:
            for name, descriptor in controls.items():
                _recognize_unknown(memory, stage, name, descriptor)
        after_summary = memory.summary()
        after = _summary_hashes(after_summary)
        stage["id"] = stage_id
        stage["model_hash_before_probes"] = before["model"]
        stage["model_hash_after_probes"] = after["model"]
        stage["model_unchanged_during_probes"] = (
            before["model"] == after["model"]
        )
        stage["example_fingerprint_before_probes"] = before["examples"]
        stage["example_fingerprint_after_probes"] = after["examples"]
        stage["examples_unchanged_during_probes"] = (
            before["examples"] == after["examples"]
        )
        stage["examples_by_label"] = {
            item["name"]: item["examples"]
            for item in after_summary.get("labels", [])
        }
        stage["summary"] = _safe_summary(after_summary)
        report["stages"].append(_sanitize(stage))

    all_labels = set(LABEL_ORDER)
    evaluate("initial_unknown", set(), unknown_families=all_labels)
    for label in protocol["stages"][1]["labels"]:
        teach(label)
    old_label_fingerprints = dict(memory.summary().get(
        "label_fingerprints", {}
    ))
    evaluate("three_labels", set(LABEL_ORDER[:3]), unknown_families={"chime"})

    teach("chime")
    evaluate("four_labels", all_labels, include_controls=True)
    four_summary = memory.summary()
    report["old_label_retention"] = {
        label: four_summary.get("label_fingerprints", {}).get(label)
        == fingerprint
        for label, fingerprint in old_label_fingerprints.items()
    }
    three_stage = next(
        item for item in report["stages"] if item["id"] == "three_labels"
    )
    four_stage = next(
        item for item in report["stages"] if item["id"] == "four_labels"
    )
    report["forgetting_delta"] = {
        label: {
            backend: {
                "correct_count_delta": (
                    four_stage["per_class"][label][backend]["correct"]
                    - three_stage["per_class"][label][backend]["correct"]
                ),
                "accuracy_delta": (
                    four_stage["per_class"][label][backend]["correct"]
                    / four_stage["per_class"][label]["queries"]
                    - three_stage["per_class"][label][backend]["correct"]
                    / three_stage["per_class"][label]["queries"]
                ),
            }
            for backend in ("snn", "matcher")
        }
        for label in LABEL_ORDER[:3]
    }

    before_reload = _summary_hashes(memory.summary())
    label_hashes_before_reload = memory.summary().get(
        "label_fingerprints", {}
    )
    reloaded = SoundComparison(memory_path, seed=model_seed)
    after_reload_summary = reloaded.summary()
    after_reload = _summary_hashes(after_reload_summary)
    report["reload_hashes"] = {
        "model_before": before_reload["model"],
        "model_after": after_reload["model"],
        "examples_before": before_reload["examples"],
        "examples_after": after_reload["examples"],
        "model_hash_matches": before_reload["model"] == after_reload["model"],
        "example_hash_matches": (
            before_reload["examples"] == after_reload["examples"]
        ),
    }
    report["reload_label_retention"] = {
        label: after_reload_summary.get("label_fingerprints", {}).get(label)
        == fingerprint
        for label, fingerprint in label_hashes_before_reload.items()
    }
    memory = reloaded
    evaluate("after_reload", all_labels, include_controls=True)
    reload_stage = report["stages"][-1]
    report["reload_results_match"] = _stage_results_match(
        four_stage, reload_stage
    )
    report["final_learning_metrics"] = _safe_summary(
        memory.summary()
    ).get("learning", {})
    report["parameter_change_evidence"] = {
        "model_hash_after_reload": _model_hash(memory.summary()),
        "reported_learning_metrics": report["final_learning_metrics"],
    }
    report["total_training_examples"] = sum(
        report["teaching"]["examples_by_label"].values()
    )
    return _sanitize(report)


def _stage_results_match(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare saved and reloaded outcomes by class and backend."""
    result: dict[str, Any] = {}
    for label in LABEL_ORDER:
        result[label] = {
            backend: first["per_class"][label][backend]
            == second["per_class"][label][backend]
            for backend in ("snn", "matcher")
        }
    result["controls"] = first["controls"] == second["controls"]
    return result


def _pool_stage_results(
    per_seed: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sum count and timing fields while preserving per-class outcomes."""
    stage_ids = [stage["id"] for stage in next(iter(per_seed.values()))["stages"]]
    pooled: list[dict[str, Any]] = []
    for stage_id in stage_ids:
        stages = [next(stage for stage in run["stages"]
                       if stage["id"] == stage_id)
                  for run in per_seed.values()]
        combined = _new_stage_counts()
        combined["id"] = stage_id
        combined["model_runs"] = len(stages)
        for backend in ("snn", "matcher"):
            for outcome in ("correct", "wrong", "rejected"):
                combined[backend][outcome] = sum(
                    stage[backend][outcome] for stage in stages
                )
        for field in ("queries", "snn_inference_wall_seconds",
                      "matcher_inference_wall_seconds",
                      "paired_record_wall_seconds"):
            combined[field] = sum(stage[field] for stage in stages)
        for label in LABEL_ORDER:
            records = [stage["per_class"][label]
                       for stage in stages if label in stage["per_class"]]
            if not records:
                continue
            merged = {"queries": sum(item["queries"] for item in records)}
            for backend in ("snn", "matcher"):
                merged[backend] = {
                    outcome: sum(item[backend][outcome] for item in records)
                    for outcome in ("correct", "wrong", "rejected")
                }
            combined["per_class"][label] = merged
        unknown_labels = set().union(*(
            set(stage["unknown_family_controls"]) for stage in stages
        ))
        combined["unknown_family_controls"] = {}
        for label in unknown_labels:
            rows = [stage["unknown_family_controls"][label]
                    for stage in stages
                    if label in stage["unknown_family_controls"]]
            combined["unknown_family_controls"][label] = {
                key: sum(row[key] for row in rows)
                for key in ("queries", "snn_false_accepts",
                            "matcher_false_accepts")
            }
        control_names = set().union(*(set(stage["controls"]) for stage in stages))
        combined["controls"] = {}
        for name in control_names:
            rows = [stage["controls"][name] for stage in stages
                    if name in stage["controls"]]
            combined["controls"][name] = {
                "snn_false_accepts": sum(
                    int(row["snn_false_accept"]) for row in rows
                ),
                "matcher_false_accepts": sum(
                    int(row["matcher_false_accept"]) for row in rows
                ),
                "queries": len(rows),
            }
        pooled.append(_sanitize(combined))
    return pooled


def run_comparison(
    protocol: dict[str, Any], memory_path: Path
) -> dict[str, Any]:
    """Run all fixed probe cases under each protocol parameter seed."""
    validate_protocol(protocol)
    probe_descriptors = [
        (case, extract_sound(make_sound(case["family"], case["seed"]),
                             SAMPLE_RATE))
        for case in probe_cases(protocol)
    ]
    controls = {
        name: _make_control(name, control)
        for name, control in protocol["probes"]["controls"].items()
    }
    per_seed: dict[str, dict[str, Any]] = {}
    for model_seed in protocol["probes"]["parameter_seeds"]:
        seed_path = memory_path.parent / f"seed-{model_seed}" / memory_path.name
        per_seed[str(model_seed)] = _run_seed_comparison(
            protocol, seed_path, model_seed, probe_descriptors, controls
        )
    return _sanitize({
        "protocol_name": protocol["name"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "notice": "Synthetic feature probes only; no microphone accuracy claim.",
        "model_seeds": protocol["probes"]["parameter_seeds"],
        "queries_per_seed": len(probe_descriptors),
        "per_seed": per_seed,
        "pooled_stages": _pool_stage_results(per_seed),
        "total_training_examples_per_seed": 12,
        "total_teaching_wall_seconds_all_seeds": sum(
            run["teaching"]["wall_seconds"] for run in per_seed.values()
        ),
        "total_snn_replay_fit_wall_seconds_all_seeds": sum(
            run["teaching"]["snn_replay_fit_wall_seconds"]
            for run in per_seed.values()
        ),
        "total_matcher_and_persistence_wall_seconds_all_seeds": sum(
            run["teaching"]["matcher_and_persistence_wall_seconds"]
            for run in per_seed.values()
        ),
    })


def _write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "sound_comparison_report.json"
    output_path.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    ) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.private_dir is not None:
        private_dir = args.private_dir.resolve()
        private_dir.mkdir(parents=True, exist_ok=False)
        report = run_comparison(protocol, private_dir / "memory.json")
    else:
        with tempfile.TemporaryDirectory(prefix="sound-comparison-") as temp:
            report = run_comparison(protocol, Path(temp) / "memory.json")
    output_path = _write_report(args.output_dir, report)
    print(json.dumps({"report": str(output_path),
                      "protocol_sha256": EXPECTED_PROTOCOL_SHA256},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
