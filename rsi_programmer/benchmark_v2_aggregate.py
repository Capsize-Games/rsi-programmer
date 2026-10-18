"""Fixed summaries and decision checks for benchmark v2."""
from __future__ import annotations

from collections import defaultdict
import random
from typing import Any

def aggregate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]
                   ) -> dict[str, Any]:
    """Summarize every frozen cell without selecting metrics post hoc."""
    family_seed: dict[tuple[str, int, str, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    for row in rows:
        values = family_seed[(row["control"], row["budget"],
                              row["family"], row["seed"])]
        values[0] += int(row["accepted"])
        values[1] += 1
        search = row.get("search", {})
        values[2] += float(search.get("signature_evaluations", 0))
        values[3] += float(row.get("wall_seconds_including_wrapper", 0))
        values[4] += float(search.get("snn_steps", 0))
    family_seed_rows = []
    family_fractions: dict[tuple[str, int, str], list[float]] = defaultdict(
        list,
    )
    overall: dict[tuple[str, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    for (control, budget, family, seed), values in sorted(
            family_seed.items()):
        successes, count, scans, wall, steps = values
        fraction = successes / count if count else 0.0
        family_fractions[(control, budget, family)].append(fraction)
        family_seed_rows.append({
            "control": control, "budget": budget, "family": family,
            "seed": seed, "successes": int(successes),
            "tasks": int(count), "success_fraction": fraction,
            "candidate_signatures": int(scans), "wall_seconds": wall,
            "snn_steps": int(steps),
        })
        total = overall[(control, budget, seed)]
        for index, value in enumerate(values):
            total[index] += value
    family_means = [{
        "control": control, "budget": budget, "family": family,
        "seed_fractions": fractions,
        "mean_seed_fraction": sum(fractions) / len(fractions),
    } for (control, budget, family), fractions in
        sorted(family_fractions.items())]
    overall_seed_rows = []
    overall_fractions: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (control, budget, seed), values in sorted(overall.items()):
        successes, count, scans, wall, steps = values
        fraction = successes / count if count else 0.0
        overall_fractions[(control, budget)].append(fraction)
        overall_seed_rows.append({
            "control": control, "budget": budget, "seed": seed,
            "successes": int(successes), "tasks": int(count),
            "success_fraction": fraction,
            "candidate_signatures": int(scans), "wall_seconds": wall,
            "snn_steps": int(steps),
        })
    overall_means = [{
        "control": control, "budget": budget,
        "seed_fractions": fractions,
        "mean_seed_fraction": sum(fractions) / len(fractions),
    } for (control, budget), fractions in sorted(overall_fractions.items())]
    controls = list(protocol["controls"])
    budgets = list(protocol["evaluation"]["candidate_scan_budgets"])
    families = list(protocol["corpus"]["families"])
    scores = {(item["control"], item["budget"]):
              item["mean_seed_fraction"] for item in overall_means}
    matched_targets = protocol["decision_rule"]["matched_success_targets"]
    matched = []
    for control in controls:
        for target in matched_targets:
            reached = [budget for budget in budgets
                       if scores.get((control, budget), 0.0) >= target]
            matched.append({
                "control": control, "target": target,
                "lowest_budget": min(reached) if reached else None,
                "unreached_censored_at": None if reached else max(budgets),
            })
    checks = []
    for budget in protocol["decision_rule"]["primary_budgets"]:
        trained = scores.get(("trained_snn", budget), 0.0)
        best_other = max((scores.get((control, budget), 0.0)
                          for control in controls
                          if control != "trained_snn"), default=0.0)
        stronger_families = 0
        for family in families:
            trained_family = next((item["mean_seed_fraction"]
                                   for item in family_means
                                   if item["control"] == "trained_snn"
                                   and item["budget"] == budget
                                   and item["family"] == family), 0.0)
            other = max((item["mean_seed_fraction"] for item in family_means
                         if item["budget"] == budget
                         and item["family"] == family
                         and item["control"] != "trained_snn"),
                        default=0.0)
            stronger_families += int(trained_family > other)
        checks.append({
            "budget": budget,
            "trained_snn_mean_fraction": trained,
            "best_other_mean_fraction": best_other,
            "overall_margin": trained - best_other,
            "families_where_trained_snn_is_higher": stronger_families,
        })
    rule = protocol["decision_rule"]
    passes = all(
        check["overall_margin"] >= rule["minimum_margin"]
        and check["families_where_trained_snn_is_higher"]
        >= rule["minimum_families_higher"]
        for check in checks
    )
    uncertainty = _paired_bootstrap(rows, protocol)
    return {
        "family_seed_rows": family_seed_rows,
        "family_means_across_seeds": family_means,
        "overall_seed_rows": overall_seed_rows,
        "overall_means_across_seeds": overall_means,
        "matched_success_budgets": matched,
        "paired_task_bootstrap": uncertainty,
        "trained_snn_advantage": {
            "rule_would_pass": passes,
            "checks": checks,
            "scope": rule["claim_scope"],
        },
    }


def _paired_bootstrap(rows: list[dict[str, Any]],
                      protocol: dict[str, Any]) -> dict[str, Any]:
    """Stratified paired task bootstrap, averaged over training seeds."""
    spec = protocol["uncertainty"]
    draws = spec["draws"]
    rng = random.Random(spec["seed"])
    family_order = list(protocol["corpus"]["families"])
    tasks = {family: sorted({row["task_id"] for row in rows
                             if row["family"] == family})
             for family in family_order}
    task_scores: dict[tuple[int, str, str], float] = {}
    for budget in protocol["evaluation"]["candidate_scan_budgets"]:
        for family in family_order:
            for task_id in tasks[family]:
                for control in protocol["controls"]:
                    successes = [int(row["accepted"]) for row in rows
                                 if row["budget"] == budget
                                 and row["family"] == family
                                 and row["task_id"] == task_id
                                 and row["control"] == control]
                    task_scores[(budget, task_id, control)] = (
                        sum(successes) / len(successes) if successes else 0.0
                    )
    comparisons = []
    for budget in protocol["decision_rule"]["primary_budgets"]:
        for control in protocol["controls"]:
            if control == "trained_snn":
                continue
            observed: list[float] = []
            for family in family_order:
                values = [task_scores[(budget, task_id, "trained_snn")]
                          - task_scores[(budget, task_id, control)]
                          for task_id in tasks[family]]
                observed.extend(values)
            mean_difference = sum(observed) / len(observed) if observed else 0
            samples = []
            for _ in range(draws):
                sampled_differences = []
                for family in family_order:
                    family_tasks = tasks[family]
                    if not family_tasks:
                        continue
                    sampled = [rng.choice(family_tasks)
                               for _ in family_tasks]
                    sampled_differences.extend(
                        task_scores[(budget, task_id, "trained_snn")]
                        - task_scores[(budget, task_id, control)]
                        for task_id in sampled
                    )
                samples.append(sum(sampled_differences)
                               / len(sampled_differences))
            samples.sort()
            comparisons.append({
                "budget": budget, "control": control,
                "mean_paired_difference": mean_difference,
                "confidence_level": spec["confidence_level"],
                "confidence_interval": [
                    samples[int(0.025 * (draws - 1))],
                    samples[int(0.975 * (draws - 1))],
                ],
            })
    return {
        "method": "family-stratified paired task bootstrap",
        "task_unit": "evaluation task; average across training seeds first",
        "draws": draws, "seed": spec["seed"],
        "confidence_level": spec["confidence_level"],
        "descriptive_only": True,
        "comparisons": comparisons,
    }
