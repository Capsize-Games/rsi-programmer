"""Unit tests for the end-to-end cost ledger (stdlib only)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from rsi_programmer.cost_ledger import PHASES, CostLedger


class CostLedgerTests(unittest.TestCase):
    def test_unknown_phase_rejected(self) -> None:
        ledger = CostLedger()
        with self.assertRaises(ValueError):
            ledger.record("flying", 1.0, 1.0)
        with self.assertRaises(ValueError):
            ledger.phase_stats("flying")

    def test_negative_sample_rejected(self) -> None:
        ledger = CostLedger()
        with self.assertRaises(ValueError):
            ledger.record("fit", -0.5, 0.1)

    def test_measure_records_non_negative_sample(self) -> None:
        ledger = CostLedger()
        with ledger.measure("search"):
            total = sum(range(1000))
        self.assertGreaterEqual(total, 0)
        stats = ledger.phase_stats("search")
        self.assertEqual(stats["n_cpu_seconds"], 1.0)
        self.assertGreaterEqual(stats["total_cpu_seconds"], 0.0)

    def test_repeated_measurements_report_spread(self) -> None:
        ledger = CostLedger()
        ledger.record("fit", 2.0, 2.5)
        ledger.record("fit", 4.0, 4.5)
        stats = ledger.phase_stats("fit")
        self.assertEqual(stats["n_cpu_seconds"], 2.0)
        self.assertAlmostEqual(stats["total_cpu_seconds"], 6.0)
        self.assertAlmostEqual(stats["mean_cpu_seconds"], 3.0)
        self.assertGreater(stats["stdev_cpu_seconds"], 0.0)
        self.assertAlmostEqual(stats["minimum_cpu_seconds"], 2.0)
        self.assertAlmostEqual(stats["maximum_cpu_seconds"], 4.0)

    def test_single_sample_spread_is_zero(self) -> None:
        ledger = CostLedger()
        ledger.record("mining", 1.5, 1.0)
        stats = ledger.phase_stats("mining")
        self.assertEqual(stats["stdev_cpu_seconds"], 0.0)

    def test_totals_sum_every_phase(self) -> None:
        ledger = CostLedger()
        ledger.record("fit", 40.0, 41.0)
        ledger.record("search", 10.0, 12.0)
        ledger.record("backend_execution", 15.0, 30.0)
        totals = ledger.totals()
        self.assertAlmostEqual(
            totals["measured_total_cpu_seconds"], 65.0)
        self.assertAlmostEqual(
            totals["measured_total_wall_seconds"], 83.0)

    def test_amortized_spreads_total_over_volume(self) -> None:
        ledger = CostLedger()
        ledger.record("fit", 42.0, 44.0)
        ledger.record("search", 28.0, 30.0)
        result = ledger.amortized(1000)
        self.assertEqual(result["volume"], 1000)
        self.assertAlmostEqual(
            result["cpu_per_request_seconds"], 0.07)
        self.assertAlmostEqual(
            result["per_phase_cpu_per_request_seconds"]["fit"], 0.042)
        with self.assertRaises(ValueError):
            ledger.amortized(0)

    def test_run_executes_command_and_records_legacy_keys(self) -> None:
        ledger = CostLedger()
        code = ledger.run("search", [sys.executable, "-c", "pass"])
        self.assertEqual(code, 0)
        body = ledger.to_dict()
        entry = body["command_costs"]["search"]
        for key in ("command", "exit_code", "wall_seconds",
                    "cpu_seconds", "whole_process_cpu_seconds",
                    "largest_child_rss_kib",
                    "accounting_source_sha256"):
            self.assertIn(key, entry)
        self.assertGreaterEqual(entry["cpu_seconds"], 0.0)

    def test_to_dict_extends_artifact_shape(self) -> None:
        ledger = CostLedger()
        ledger.record("fit", 42.406, 44.0)
        ledger.declare_excluded("Historical donor library")
        ledger.declare_excluded("Historical donor library")
        body = ledger.to_dict(volume=1000)
        self.assertIn("cost_definition", body)
        self.assertEqual(body["excluded_costs"],
                         ["Historical donor library"])
        self.assertIn("fit", body["phases"])
        self.assertIn("measured_total_cpu_seconds", body)
        self.assertEqual(body["amortized"]["volume"], 1000)
        self.assertNotIn("amortized", ledger.to_dict())

    def test_write_json_round_trip(self) -> None:
        ledger = CostLedger()
        ledger.record("qualification", 3.0, 3.5)
        with tempfile.TemporaryDirectory() as directory:
            path = ledger.write_json(Path(directory) / "cost.json")
            body = json.loads(path.read_text())
        self.assertAlmostEqual(
            body["measured_total_cpu_seconds"], 3.0)

    def test_from_legacy_ingests_frozen_summary_shape(self) -> None:
        legacy = {
            "cost_definition": "legacy definition",
            "excluded_costs": ["Historical creation"],
            "command_costs": {
                "search": {"cpu_seconds": 55.8,
                           "whole_process_cpu_seconds": 55.83,
                           "wall_seconds": 56.4},
                "python": {"cpu_seconds": 3.9,
                           "whole_process_cpu_seconds": 3.95,
                           "wall_seconds": 12.8},
                "curlee": {"cpu_seconds": 11.8,
                           "whole_process_cpu_seconds": 11.83,
                           "wall_seconds": 18.5}},
            "fresh_fit_cpu_seconds": 42.406}
        ledger = CostLedger.from_legacy(legacy)
        self.assertAlmostEqual(
            ledger.totals()["measured_total_cpu_seconds"],
            55.83 + 3.95 + 11.83)
        self.assertEqual(
            ledger.to_dict()["excluded_costs"], ["Historical creation"])
        self.assertIn("search", ledger.to_dict()["command_costs"])
        phases = ledger.to_dict()["phases"]
        self.assertIn("backend_execution", phases)
        self.assertAlmostEqual(
            phases["backend_execution"]["total_cpu_seconds"], 15.78)
        self.assertAlmostEqual(
            phases["search"]["total_cpu_seconds"], 55.83 - 42.406)
        self.assertAlmostEqual(
            phases["fit"]["total_cpu_seconds"], 42.406)

    def test_all_nine_phases_recognized(self) -> None:
        self.assertEqual(len(PHASES), 9)
        for phase in ("donor_library_creation", "fit", "mining",
                      "qualification", "oracle_labelling", "search",
                      "emission", "backend_execution",
                      "candidate_evaluation"):
            ledger = CostLedger()
            ledger.record(phase, 0.0, 0.0)
            self.assertIn(phase, ledger.to_dict()["phases"])


if __name__ == "__main__":
    raise SystemExit(unittest.main())
