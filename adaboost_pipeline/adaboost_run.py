from pathlib import Path
import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

# This code is used to run the adaboost_core.py within the adroit cluster.

# Records the directory the script was launched from.
# This is used as the starting point for locating the AdaBoost project files.
ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# This locates the project directory containing adaboost_core.py.
# The search first checks th current directory and its parents, then searches downward.
# This permits the script to run on the cluster for variable launch directories.
def locate_project_dir():
    candidates = [ROOT, *ROOT.parents]

    for candidate in candidates:
        if (candidate / "adaboost_core.py").exists():
            return candidate

    for candidate in candidates:
        for match in candidate.rglob("adaboost_core.py"):
            parent = match.parent
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent

    raise FileNotFoundError(
        "Could not locate the project folder. Place adaboost_run.py and adaboost_core.py in the same folder before running."
    )

# Identify the project directory, adds it to the import path, and changes the working directory to it.

# This ensures that the rpeort and figures are written to the project directory.
project_dir = locate_project_dir()
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))
os.chdir(project_dir)

# Import the core AdaBoost experiment and the report-writing functions.
from adaboost_core import run_default_adaboost_experiments, write_adaboost_report, adaboost_report


# The script accepts an optional number of boosting rounds and an optional output directory.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the AdaBoost extension experiments, write the AdaBoost text report, and save AdaBoost figures."
    )
    parser.add_argument(
        "rounds",
        nargs="?",
        type=int,
        default=25,
        help="Optional number of boosting rounds. Default: %(default)s.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_adaboost",
        help="Directory in which the AdaBoost report and figures are written. Default: %(default)s.",
    )
    return parser.parse_args()

# This function suppresses the plotting code by saving the figure and then closing it.
# This saves a single diagnostic plot to the disk each time it's run.
def save_plot(path, title, x_values, y_values, y_label):
    plt.figure(figsize=(7, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# This saves the AdaBoost diagonistics for each default dataset used.

# For each dataset, this writes:
# 1. The weighted weak-learner error by round,
# 2. The entropy of sample weights by round,
# 3. The hard-example mass by round,
# 4. The minimum sample margin by round.

def save_all_adaboost_figures(results, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in results.items():
        result = payload["result"]
        rounds_axis = np.arange(1, result["rounds"] + 1)
        state_axis = np.arange(0, result["rounds"] + 1)

        save_plot(
            figures_dir / f"{name}_weighted_error.png",
            f"{name}: weighted weak-learner error",
            rounds_axis,
            result["weighted_errors"],
            "Weighted error",
        )

        save_plot(
            figures_dir / f"{name}_entropy.png",
            f"{name}: entropy of sample weights",
            state_axis,
            result["entropy_history"],
            "Entropy",
        )

        save_plot(
            figures_dir / f"{name}_hard_example_mass.png",
            f"{name}: hard-example mass",
            rounds_axis,
            result["hard_example_mass_history"],
            "Hard-example mass",
        )

        save_plot(
            figures_dir / f"{name}_minimum_margin.png",
            f"{name}: minimum sample margin",
            state_axis,
            np.min(result["margin_history"], axis=1),
            "Minimum margin",
        )


# This creates the output directories, runs the default AdaBoost experiments,
# writes the text report, saves the diagnostic figures, and prints a short summary.
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    results = run_default_adaboost_experiments(rounds=args.rounds)
    report_path = write_adaboost_report(results, output_dir / "adaboost_results.txt")
    save_all_adaboost_figures(results, figures_dir)

    print("Working directory:", project_dir)
    print("Import root:", sys.path[0])
    print("Rounds:", args.rounds)
    print("Report written to:", report_path)
    print("Figures saved under:", figures_dir)
    print()
    print(adaboost_report(results)[:20000])


if __name__ == "__main__":
    main()
