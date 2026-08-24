"""One-command reproduction driver.

    python reproduce.py            # cheap path: reuse committed CV predictions
    python reproduce.py --full     # also re-run the expensive refitting stages
    python reproduce.py --list     # show the pipeline without running it

Performs, in order: input-hash validation, preprocessing, the analysis
stages, the test suite, table/figure regeneration, the manuscript and
supplement builds, and a final check that manuscript numbers match generated
results.

A Makefile with the same targets is provided for POSIX users; this script is
the portable path and is the one verified on the reference environment
(Windows), where ``make`` is not present.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).is_file():                      # POSIX venv layout
    alt = ROOT / ".venv" / "bin" / "python"
    PY = str(alt) if alt.is_file() else sys.executable

EXPECTED_HASHES = {
    "data/raw/ckd-dataset-v2.csv":
        "f24075f420b0f271bfddf3844a40061dbe9e2cb1c2336daa64463122344ea84a",
    "data/external/uci2015/raw/data.csv":
        "0eeea8d17f5ad8792d854999d6ddf1602ec4e116616f6ccdfdaef0ba8109e694",
}

#: (label, script, expensive?)
STAGES: list[tuple[str, str, bool]] = [
    ("provenance gate",            "scripts/00_provenance.py",            False),
    ("data preparation",           "scripts/01_prepare_data.py",          False),
    ("exploratory analysis",       "scripts/02_eda.py",                   False),
    ("nested cross-validation",    "scripts/03_nested_cv.py",             True),
    ("evaluation",                 "scripts/04_evaluate.py",              False),
    ("importance and stability",   "scripts/05_importance_stability.py",  True),
    ("literature tables",          "scripts/07_literature.py",            False),
    ("label robustness",           "scripts/08_sensitivity_labels.py",    False),
    ("optimism",                   "scripts/09_optimism.py",              False),
    ("EBM shape functions",        "scripts/10_ebm_shapes.py",            True),
    ("continuous recovery",        "scripts/11_continuous_recovery.py",   False),
    ("binned vs continuous",       "scripts/12_binned_vs_continuous.py",  True),
    ("TRIPOD checklist",           "scripts/13_tripod_checklist.py",      False),
    ("encoding robustness",        "scripts/14_encoding_robustness.py",   True),
    ("whole-procedure bootstrap",  "scripts/15_procedure_bootstrap.py",   True),
    ("provenance figure",          "scripts/17_provenance_figure.py",     False),
    ("MIMIC extraction",           "scripts/18_mimic_etl.py",             False),
    ("transportability probe",     "scripts/19_external_validation.py",   True),
    ("benchmark diagnostics",      "scripts/20_benchmark_diagnostics.py", False),
    ("provenance sensitivity",     "scripts/21_provenance_sensitivity.py", False),
    ("feature registry",           "scripts/22_build_feature_registry.py", False),
    ("case-mix reanalysis",        "scripts/23_casemix_reanalysis.py",    True),
    ("LaTeX table fragments",      "scripts/24_make_latex_tables.py",     False),
    ("bootstrap convergence",      "scripts/25_bootstrap_convergence.py", False),
]


def check_hashes() -> bool:
    ok = True
    for relative, expected in sorted(EXPECTED_HASHES.items()):
        path = ROOT / relative
        if not path.is_file():
            print(f"  MISSING  {relative}")
            ok = False
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == expected:
            print(f"  ok       {relative}")
        else:
            print(f"  MISMATCH {relative}\n           expected {expected[:16]}…"
                  f" found {actual[:16]}…")
            ok = False
    return ok


def run(label: str, command: list[str]) -> bool:
    print(f"\n>>> {label}")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - started
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"<<< {label}: {status} in {elapsed:.1f}s")
    return result.returncode == 0


def record_test_inventory(junit: Path) -> bool:
    """Turn the pytest run into a result file the manuscript can quote.

    The Reproducibility statement names a test count. Typed by hand it goes
    stale the moment a test is added --- it said 416 while the suite held
    457. Writing it as a result file means the same rule applies to it as to
    every other number in the paper: generated, then checked.
    """
    if not junit.is_file():
        print("  no junit report; test inventory not written")
        return False
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    frame = pd.DataFrame([{
        "tests": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
    }])
    out = ROOT / "reports" / "tables" / "table_52_test_inventory.csv"
    frame.to_csv(out, index=False, lineterminator="\n")
    print(f"  test inventory: {frame.iloc[0]['tests']} tests")
    return True


def sync_figures() -> bool:
    r"""Copy the figures the manuscript includes into the build directory.

    ``paper/ieee_paper.tex`` sets ``\graphicspath{{figures/}}``, so the build
    reads ``paper/figures/`` while the analysis stages write
    ``reports/figures/``. Without this step a re-run regenerates a figure and
    then builds the manuscript from the previous copy of it --- the kind of
    silent staleness this pipeline exists to prevent.
    """
    tex = (ROOT / "paper" / "ieee_paper.tex").read_text(encoding="utf-8")
    tex += (ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
    wanted = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
    source_dir = ROOT / "reports" / "figures"
    target_dir = ROOT / "paper" / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in sorted(wanted):
        source = source_dir / name
        if not source.is_file():
            missing.append(name)
            continue
        shutil.copy2(source, target_dir / name)
    print(f"  synced {len(wanted) - len(missing)} figure(s) to paper/figures/")
    for name in missing:
        print(f"  MISSING {name} (not produced by any stage that ran)")
    return not missing


def build_latex(name: str) -> bool:
    if shutil.which("pdflatex") is None:
        print(f"  pdflatex not found; skipping {name}")
        return True
    for _ in range(2):                       # twice, for cross-references
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name],
            cwd=ROOT / "paper", stdout=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            print(f"  {name}: FAILED")
            return False
    print(f"  built paper/{Path(name).stem}.pdf")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also run the expensive refitting stages")
    parser.add_argument("--list", action="store_true",
                        help="print the pipeline and exit")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    if args.list:
        for label, script, expensive in STAGES:
            print(f"  [{'expensive' if expensive else 'cheap    '}] "
                  f"{label:28s} {script}")
        return 0

    failures: list[str] = []

    print("=" * 70)
    print("1. input hashes")
    print("=" * 70)
    if not check_hashes():
        print("\nAborting: inputs differ from those the results were computed "
              "from. Diagnose before continuing.")
        return 1

    print("\n" + "=" * 70)
    print(f"2. analysis stages ({'full' if args.full else 'cheap path'})")
    print("=" * 70)
    for label, script, expensive in STAGES:
        if expensive and not args.full:
            print(f"  skipped (expensive): {label}")
            continue
        if not run(label, [PY, script]):
            failures.append(label)

    if not args.skip_tests:
        print("\n" + "=" * 70)
        print("3. test suite")
        print("=" * 70)
        junit = ROOT / "reports" / "tests" / "junit.xml"
        junit.parent.mkdir(parents=True, exist_ok=True)
        if not run("pytest", [PY, "-m", "pytest", "tests/", "-q",
                              f"--junitxml={junit}"]):
            failures.append("tests")

    print("\n" + "=" * 70)
    print("4. manuscript and supplement")
    print("=" * 70)
    record_test_inventory(ROOT / "reports" / "tests" / "junit.xml")
    run("regenerate LaTeX fragments",
        [PY, "scripts/24_make_latex_tables.py"])
    if not sync_figures():
        failures.append("figure sync")
    for document in ("ieee_paper.tex", "supplement.tex"):
        if not build_latex(document):
            failures.append(document)

    print("\n" + "=" * 70)
    print("5. manuscript-consistency verification")
    print("=" * 70)
    if not run("verify", [PY, "-m", "pytest",
                          "tests/test_manuscript_consistency.py", "-q"]):
        failures.append("verification")

    print("\n" + "=" * 70)
    if failures:
        print(f"REPRODUCTION INCOMPLETE — {len(failures)} step(s) failed:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("REPRODUCTION COMPLETE")
    if not args.full:
        print("Cheap path: reused the committed cross-validation predictions.")
        print("Run with --full to regenerate them (hours).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
