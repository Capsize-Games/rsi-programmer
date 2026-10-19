"""Plot the saved paired sound comparison without running models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def chart_data(report: dict[str, Any]) -> dict[str, Any]:
    """Derive plotted values from pooled integer counts and elapsed work."""
    stages = [stage for stage in report["pooled_stages"] if stage["queries"]]
    four = next(stage for stage in stages if stage["id"] == "four_labels")
    timed_queries = (
        four["queries"]
        + sum(item["queries"]
              for item in four["unknown_family_controls"].values())
        + sum(item["queries"] for item in four["controls"].values())
    )
    return {
        "stages": [{
            "id": stage["id"], "queries": stage["queries"],
            "snn_correct": stage["snn"]["correct"],
            "matcher_correct": stage["matcher"]["correct"],
            "snn_percent": 100 * stage["snn"]["correct"] / stage["queries"],
            "matcher_percent": (100 * stage["matcher"]["correct"]
                                / stage["queries"]),
        } for stage in stages],
        "inference_stage": "four_labels",
        "inference_queries": timed_queries,
        "inference_scope": "Known families plus unknown/control clips",
        "snn_ms_per_query": (1000 * four["snn_inference_wall_seconds"]
                             / timed_queries),
        "matcher_ms_per_query": (
            1000 * four["matcher_inference_wall_seconds"] / timed_queries),
    }


def main() -> None:
    """Write one figure and the exact aggregate values used to plot it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    data = chart_data(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [1.65, 1]})
    colors = {"snn": "#168575", "matcher": "#c77a17"}
    positions = list(range(len(data["stages"])))
    for backend, offset, label in (("snn", -0.18, "SNN + support guard"),
                                    ("matcher", 0.18, "Example matcher")):
        values = [stage[backend + "_percent"] for stage in data["stages"]]
        axes[0].bar([p + offset for p in positions], values, width=0.34,
                    color=colors[backend], label=label)
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Correctly named known sounds (%)")
    axes[0].set_xticks(positions, [
        "3 sounds\n108 decisions", "4 sounds\n144 decisions",
        "After reload\n144 decisions"])
    axes[0].set_title("Learning and retention: accuracy ties")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    for index, stage in enumerate(data["stages"]):
        axes[0].text(index, 103,
                     f"{stage['snn_correct']}/{stage['queries']} each",
                     ha="center", fontsize=9)
    times = [data["snn_ms_per_query"], data["matcher_ms_per_query"]]
    axes[1].barh([1, 0], times,
                 color=[colors["snn"], colors["matcher"]], height=0.55)
    axes[1].set_yticks([1, 0], ["SNN + guard", "Matcher"])
    axes[1].set_xlim(0, max(times) * 1.32)
    axes[1].set_xlabel("Milliseconds per decision (lower is better)")
    axes[1].set_title("Decision time: known + control clips")
    for position, value in zip([1, 0], times):
        axes[1].text(value + 0.04, position, f"{value:.2f}", va="center")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SNN creature versus the previous matcher", fontsize=15)
    fig.text(0.02, 0.03,
             "Generated sounds: 12 queries per family, reused across three "
             "SNN seeds. No microphone accuracy or chip-energy claim.",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(args.output_dir / "learning-retention.png", dpi=150)
    plt.close(fig)
    (args.output_dir / "chart-data.json").write_text(json.dumps(
        data, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
