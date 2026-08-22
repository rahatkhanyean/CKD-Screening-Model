"""Run the complete analysis pipeline in order, stopping at the first failure.

    python scripts/run_all.py                 # everything
    python scripts/run_all.py --skip-cv       # reuse existing CV predictions
    python scripts/run_all.py --repeats 1     # fast end-to-end smoke run
    python scripts/run_all.py --no-tests      # skip the test suite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], label: str) -> float:
    print("\n" + "=" * 78)
    print(f"  {label}")
    print("=" * 78)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"\nFAILED: {label} (exit code {proc.returncode}) after {elapsed:.1f}s")
        raise SystemExit(proc.returncode)
    print(f"\nOK: {label} in {elapsed:.1f}s")
    return elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-cv", action="store_true",
                   help="reuse the existing cv_predictions.csv.gz")
    p.add_argument("--no-tests", action="store_true")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    py = sys.executable
    timings: dict[str, float] = {}

    timings["01 prepare data"] = run([py, "scripts/01_prepare_data.py"], "Stage 1: data preparation")
    timings["02 EDA"] = run([py, "scripts/02_eda.py"], "Stage 2: exploratory analysis")

    if not args.no_tests:
        timings["tests"] = run([py, "-m", "pytest", "tests/", "-q"], "Automated tests")

    if not args.skip_cv:
        cmd = [py, "scripts/03_nested_cv.py", "--n-jobs", str(args.n_jobs)]
        if args.repeats:
            cmd += ["--repeats", str(args.repeats)]
        timings["03 nested CV"] = run(cmd, "Stage 3: repeated nested cross-validation")
    else:
        print("\nSkipping stage 3; reusing existing predictions.")

    timings["04 evaluate"] = run([py, "scripts/04_evaluate.py"], "Stage 4: evaluation and calibration")
    timings["05 importance"] = run([py, "scripts/05_importance_stability.py"],
                                   "Stage 5: explainability and stability")
    timings["06 report"] = run([py, "scripts/06_write_report.py"],
                               "Stage 6: assemble the research report")
    timings["notebook"] = run([py, "scripts/make_notebook.py"],
                              "Regenerate the walkthrough notebook")

    if not args.no_tests:
        # Re-run the suite now that artefacts exist: the artefact-dependent
        # tests were skipped the first time round.
        timings["tests (final)"] = run([py, "-m", "pytest", "tests/", "-q"],
                                       "Automated tests (with artefacts present)")

    print("\n" + "=" * 78)
    print("  Pipeline complete")
    print("=" * 78)
    for k, v in timings.items():
        print(f"  {k:<22} {v/60:6.2f} min")
    print(f"  {'TOTAL':<22} {sum(timings.values())/60:6.2f} min")
    print("\nOutputs:")
    print("  reports/research_report.md")
    print("  reports/data_quality_report.md")
    print("  reports/figures/")
    print("  reports/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
