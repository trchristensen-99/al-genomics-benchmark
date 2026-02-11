#!/usr/bin/env python3
"""
Plot performance curves for Yeast and K562 across different data fractions.

Reads JSON evaluation results from `results/evaluations/` with filenames like:
    yeast_0.010_seed303_eval.json

Each JSON file is expected to contain a list of dicts, one per test set, e.g.:
    {
      "test_type": "in_distribution" | "snv" | "ood",
      "dataset": "yeast" | "k562",
      "pearson_r": float,
      ...
    }

For each (dataset, test_type), we aggregate Pearson R across seeds at each
training fraction, then plot:
  - x-axis: training data fraction (log scale, shown as %)
  - y-axis: Pearson R (0–1)
  - line: mean Pearson R
  - shaded region: ±1 std dev (no error bars)

Outputs:
  - plots/yeast_performance_curves.png
  - plots/k562_performance_curves.png (if K562 results exist)
"""

from pathlib import Path
import re
import json
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


RESULTS_DIR = Path("results") / "evaluations"
PLOTS_DIR = Path("plots")

TEST_TYPES = [
    ("in_distribution", "In-Distribution"),
    ("snv", "SNV"),
    ("ood", "OOD"),
]

DATASET_COLORS = {
    "yeast": "#2E86AB",
    "k562": "#E67E22",
}


def load_results(results_dir: Path) -> Dict[str, Dict[str, Dict[float, List[float]]]]:
    """
    Load evaluation JSON files and organize by:
        data[dataset][test_type][fraction] -> list[pearson_r]
    """
    data: Dict[str, Dict[str, Dict[float, List[float]]]] = {}

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return data

    pattern = re.compile(
        r"^(?P<dataset>[a-zA-Z0-9_]+)_(?P<fraction>[0-9.]+)_seed(?P<seed>.+)_eval\.json$"
    )

    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"No evaluation JSON files found in {results_dir}")
        return data

    print(f"Found {len(files)} evaluation files in {results_dir}")

    for path in files:
        m = pattern.match(path.name)
        if not m:
            # Skip files that don't follow the expected naming convention
            print(f"Skipping file with unexpected name format: {path.name}")
            continue

        dataset = m.group("dataset")
        try:
            fraction = float(m.group("fraction"))
        except ValueError:
            print(f"Could not parse fraction from {path.name}, skipping")
            continue

        try:
            with path.open("r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        if isinstance(results, dict):
            results_list: List[Dict[str, Any]] = [results]
        else:
            results_list = results

        for entry in results_list:
            test_type = entry.get("test_type")
            pearson = entry.get("pearson_r")

            if test_type is None or pearson is None:
                continue

            data.setdefault(dataset, {}) \
                .setdefault(test_type, {}) \
                .setdefault(fraction, []) \
                .append(float(pearson))

    return data


def _format_fraction_ticks(x: float, _pos: int) -> str:
    """Format log-scale x ticks as percentages."""
    return f"{x * 100:.1f}%"


def plot_dataset_performance(
    dataset: str,
    dataset_data: Dict[str, Dict[float, List[float]]],
    output_path: Path,
) -> None:
    """Plot performance curves for a single dataset across all test types."""
    print(f"\nPlotting performance curves for dataset: {dataset}")

    fig, axes = plt.subplots(
        1,
        len(TEST_TYPES),
        figsize=(5 * len(TEST_TYPES), 4),
        sharey=True,
    )

    if len(TEST_TYPES) == 1:
        axes = [axes]

    color = DATASET_COLORS.get(dataset, "#2E86AB")

    for ax, (test_key, test_label) in zip(axes, TEST_TYPES):
        test_data = dataset_data.get(test_key, {})

        if not test_data:
            ax.set_title(f"{test_label} (no data)")
            ax.set_xlabel("Training data fraction (%)")
            continue

        fractions = sorted(test_data.keys())
        means = [float(np.mean(test_data[f])) for f in fractions]
        stds = [float(np.std(test_data[f])) for f in fractions]

        # Plot mean Pearson R
        ax.plot(
            fractions,
            means,
            "o-",
            linewidth=2,
            markersize=6,
            label=dataset,
            color=color,
            alpha=0.85,
        )

        # Shaded std dev region
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(
            fractions,
            lower,
            upper,
            alpha=0.15,
            color=color,
        )

        # Log-scale x-axis
        ax.set_xscale("log")
        ax.set_xlim(min(fractions) * 0.8, max(fractions) * 1.2)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_fraction_ticks))

        # Y-axis limits for Pearson R
        ax.set_ylim(0.0, 1.0)

        ax.set_title(test_label, fontsize=12)
        ax.set_xlabel("Training data fraction (%)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Pearson R", fontsize=11)

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.suptitle(f"{dataset.upper()} performance across data fractions", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.94])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    data = load_results(RESULTS_DIR)
    if not data:
        return

    for dataset, dataset_data in sorted(data.items()):
        output_path = PLOTS_DIR / f"{dataset}_performance_curves.png"
        plot_dataset_performance(dataset, dataset_data, output_path)


if __name__ == "__main__":
    main()

