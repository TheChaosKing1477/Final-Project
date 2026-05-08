from pathlib import Path
import sys
import os
import argparse

# Record the directory the script is launched in.
# This provides the starting point for the program's detection of Hedge project files.
ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# This locates the project directory that contains both hedge_core.py and hedge_plots.py.
# The search first checks the current directory and its parent directories, then searches
# downward as needed. 
def locate_project_dir():
    candidates = [ROOT, *ROOT.parents]

    for candidate in candidates:
        if (candidate / "hedge_core.py").exists() and (candidate / "hedge_plots.py").exists():
            return candidate

    for candidate in candidates:
        for match in candidate.rglob("hedge_core.py"):
            parent = match.parent
            if (parent / "hedge_plots.py").exists():
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                return parent

    raise FileNotFoundError(
        "Could not locate the project folder. Place hedge_run.py, hedge_core.py, and hedge_plots.py in the same folder before running."
    )


# After identifying the project directory, this script adds it to the Python import
# path and switches the working directory to it so that all outputs are written there.
project_dir = locate_project_dir()
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))
os.chdir(project_dir)

# This imports the core simulation, report-writing, and plotting functions.
from hedge_core import DEFAULT_HORIZON, run_all_cases, full_report, write_report
from hedge_plots import save_all_figures

# This parse command enables the script to accept:
# - An optional simulation horizon T (default: 50)
# - a report-only mode that runs the hedge pipeline and provides a text print.
# - a figures-only mode that runs the hedge pipeline and saves figures, but does not write a text report.
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Hedge simulations, write the text report, and save figures. "
            f"If no horizon T is supplied, the script uses T={DEFAULT_HORIZON}."
        )
    )
    parser.add_argument(
        "T",
        nargs="?",
        type=int,
        default=DEFAULT_HORIZON,
        help="Optional simulation horizon (number of time steps). Default: %(default)s.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--report-only",
        action="store_true",
        help="Run the simulations and write the text report, but do not save figures.",
    )
    mode_group.add_argument(
        "--figures-only",
        action="store_true",
        help="Run the simulations and save figures, but do not write the text report.",
    )
    return parser.parse_args()


# This creates the output folders, runs all cases, and prints a short summary to the terminal.
# In the default case it writes the report and saves figures, which can be toggled off with the
# parsed input.
def main():
    args = parse_args()

    results_dir = Path("results")
    figures_dir = results_dir / "figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    all_results = run_all_cases(use_shared_eta=True, T=args.T)

    report_path = None
    figures_written = False

    if not args.figures_only:
        report_path = write_report(all_results, path=results_dir / "hedge_results.txt")

    if not args.report_only:
        save_all_figures(all_results, output_dir=figures_dir)
        figures_written = True

    print("Working directory:", project_dir)
    print("Import root:", sys.path[0])
    print("Simulation horizon T:", args.T)

    if args.report_only:
        print("Run mode: report only")
    elif args.figures_only:
        print("Run mode: figures only")
    else:
        print("Run mode: full")

    if report_path is not None:
        print("Report written to:", report_path)

    if figures_written:
        print("Figures saved under:", figures_dir)

    print()
    print(full_report(all_results)[:20000])


if __name__ == "__main__":
    main()
