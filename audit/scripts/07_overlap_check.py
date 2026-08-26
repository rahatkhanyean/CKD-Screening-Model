"""Task 3: record-level overlap test between UCI id=857 and id=336.

Method
------
id=857 stores every continuous variable as an interval string. id=336 stores
raw values. So each id=857 patient is a box in 13-dimensional space, and an
id=336 record is a point. A record is *compatible* with a patient when the
outcome label matches and every shared value falls inside the corresponding
interval. Missing values on either side are treated as compatible, which can
only ever *raise* the match fraction --- the conservative direction for a
sceptic, and the reason uniqueness rather than match fraction is the
statistic that carries the argument.

Maximum bipartite matching then asks whether all 200 patients can be
simultaneously assigned to distinct records.

Two encodings must be undone first
----------------------------------
1. Excel date-mangling. Bin labels like "1-Jan" and "20-Dec" are Excel's
   rendering of "1-1" and "12-20". The rule is "D-Mon" <- "M - D", where Mon
   is the month name for M. Decoded here explicitly rather than dropped.
2. Open bins. "< X" and ">= X" (rendered with a Unicode >=) become
   half-infinite intervals.

Nulls
-----
Each candidate column is permuted independently, preserving every marginal
while destroying cross-variable structure, and the match fraction is
recomputed. Row-order permutation is NOT used: maximum bipartite matching
depends only on the multiset of candidate records, so permuting rows leaves
the statistic unchanged by construction and would be a degenerate null.

Cross-check
-----------
Ten categorical variables take no part in the matching. For patients pinned
to exactly one record, their values are compared. Agreement on variables the
matcher never saw is evidence the correspondence is real rather than an
artefact of the containment relation.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "outputs"

#: Shared variables stored as intervals in id=857 and as values in id=336.
CONTAINMENT = ("age", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
               "hemo", "pcv", "wbcc", "rbcc")

#: Shared categoricals deliberately excluded from matching, used only to
#: audit the pins afterwards.
HELD_OUT = ("rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe",
            "ane")

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def undo_excel(label: str) -> str:
    """'1-Jan' -> '1 - 1'; '20-Dec' -> '12 - 20'."""
    m = re.fullmatch(r"\s*(\d+)\s*-\s*([A-Za-z]{3})\s*", label)
    if not m:
        return label
    day, mon = int(m.group(1)), m.group(2).lower()
    if mon not in MONTHS:
        return label
    return f"{MONTHS[mon]} - {day}"


def parse_interval(raw) -> tuple[float, float] | None:
    """Interval string -> (low, high) inclusive bounds, or None if missing."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = undo_excel(str(raw).strip())
    if not text or text.lower() in {"nan", "?", ""}:
        return None
    text = text.replace("≥", ">=").replace("≤", "<=")
    if text.startswith("<"):
        return (-np.inf, float(text.lstrip("<= ").strip()))
    if text.startswith(">="):
        return (float(text.lstrip(">= ").strip()), np.inf)
    if text.startswith(">"):
        return (float(text.lstrip("> ").strip()), np.inf)
    m = re.fullmatch(r"(-?[\d.]+)\s*-\s*(-?[\d.]+)", text)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    try:
        v = float(text)
        return (v, v)
    except ValueError:
        return None


def normalise_categorical(series: pd.Series) -> pd.Series:
    """Map both releases' categorical spellings onto a common 0/1 scale."""
    mapping = {
        "normal": 0, "abnormal": 1, "notpresent": 0, "present": 1,
        "no": 0, "yes": 1, "good": 0, "poor": 1, "": np.nan, "?": np.nan,
    }
    def one(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower().rstrip("\t")
        if s in mapping:
            return mapping[s]
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.map(one)


def build_compatibility(boxes, points, y857, y336, tol=0.0):
    """Boolean matrix: patient i compatible with record j."""
    n, m = len(y857), len(y336)
    compat = np.zeros((n, m), dtype=bool)
    for i in range(n):
        if y857[i] is np.nan:
            continue
        ok = (y336 == y857[i])
        for var_index, bounds in enumerate(boxes[i]):
            if bounds is None:
                continue          # missing interval: compatible with anything
            lo, hi = bounds
            col = points[:, var_index]
            inside = np.isnan(col) | ((col >= lo - tol) & (col <= hi + tol))
            ok = ok & inside
            if not ok.any():
                break
        compat[i] = ok
    return compat


def match_fraction(compat: np.ndarray) -> tuple[float, np.ndarray]:
    sparse = csr_matrix(compat.astype(np.int8))
    pairing = maximum_bipartite_matching(sparse, perm_type="column")
    return float((pairing >= 0).mean()), pairing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(RAW / "uci_336_X.csv")
    b = pd.read_csv(RAW / "uci_857_X.csv")
    ya = pd.read_csv(RAW / "uci_336_y.csv").iloc[:, 0]
    yb = pd.read_csv(RAW / "uci_857_y.csv").iloc[:, 0]

    # Outcome to a common 0/1. id=336 has stray tab characters.
    y336 = ya.astype(str).str.strip().str.lower().map(
        {"ckd": 1, "notckd": 0}).to_numpy(dtype=float)
    y857 = yb.astype(str).str.strip().str.lower().map(
        {"ckd": 1, "notckd": 0}).to_numpy(dtype=float)
    print(f"id=336: {len(y336)} records ({int(np.nansum(y336))} ckd)")
    print(f"id=857: {len(y857)} patients ({int(np.nansum(y857))} ckd)")

    shared = [v for v in CONTAINMENT if v in a.columns and v in b.columns]
    print(f"shared containment variables ({len(shared)}): {shared}")

    boxes = [[parse_interval(b.iloc[i][v]) for v in shared]
             for i in range(len(b))]
    unparsed = sum(1 for row in boxes for cell in row if cell is None)
    print(f"intervals parsed; {unparsed} missing/unparsed cells "
          f"(treated as compatible)")

    points = np.column_stack([
        pd.to_numeric(a[v], errors="coerce").to_numpy(dtype=float)
        for v in shared])

    # ---- observed ----------------------------------------------------
    compat = build_compatibility(boxes, points, y857, y336)
    frac, pairing = match_fraction(compat)
    partners = compat.sum(axis=1)
    unique_pins = int((partners == 1).sum())
    print(f"\nOBSERVED  match fraction = {frac:.4f}")
    print(f"          unique pins     = {unique_pins} / {len(y857)}")
    print(f"          max partners    = {int(partners.max())}, "
          f"mean {partners.mean():.2f}, zero-partner rows "
          f"{int((partners == 0).sum())}")

    # ---- null: independent column permutation -------------------------
    rng = np.random.default_rng(args.seed)
    draws = np.empty(args.permutations)
    for k in range(args.permutations):
        shuffled = points.copy()
        for c in range(shuffled.shape[1]):
            shuffled[:, c] = rng.permutation(shuffled[:, c])
        null_compat = build_compatibility(boxes, shuffled, y857, y336)
        draws[k], _ = match_fraction(null_compat)
    print(f"\nNULL      column permutation over {args.permutations} draws")
    print(f"          mean {draws.mean():.4f}  sd {draws.std(ddof=1):.4f}  "
          f"max {draws.max():.4f}")
    exceed = int((draws >= frac).sum())
    print(f"          draws >= observed: {exceed} "
          f"(p <= {(exceed + 1) / (args.permutations + 1):.4f})")

    # ---- cross-check on variables never used --------------------------
    held = [v for v in HELD_OUT if v in a.columns and v in b.columns]
    a_held = {v: normalise_categorical(a[v]) for v in held}
    b_held = {v: normalise_categorical(b[v]) for v in held}
    agree = disagree = 0
    per_var = {}
    for i in range(len(y857)):
        if partners[i] != 1:
            continue
        j = int(np.flatnonzero(compat[i])[0])
        for v in held:
            x, y = b_held[v].iloc[i], a_held[v].iloc[j]
            if pd.isna(x) or pd.isna(y):
                continue
            hit = float(x) == float(y)
            agree += hit
            disagree += (not hit)
            d = per_var.setdefault(v, [0, 0])
            d[0] += hit
            d[1] += (not hit)
    total = agree + disagree
    print(f"\nCROSS-CHECK on {len(held)} variables never used in matching")
    print(f"          {agree}/{total} comparisons agree "
          f"({agree / max(total, 1):.4f}); {disagree} contradictions")
    for v, (ok, bad) in sorted(per_var.items()):
        print(f"            {v:6s} {ok:4d} agree, {bad:3d} disagree")

    summary = {
        "n_857": int(len(y857)), "n_336": int(len(y336)),
        "containment_variables": shared,
        "observed_match_fraction": frac,
        "unique_pins": unique_pins,
        "max_partners": int(partners.max()),
        "mean_partners": float(partners.mean()),
        "rows_with_no_partner": int((partners == 0).sum()),
        "null_draws": int(args.permutations),
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)),
        "null_max": float(draws.max()),
        "null_draws_at_or_above_observed": exceed,
        "null_p_upper_bound": (exceed + 1) / (args.permutations + 1),
        "held_out_variables": held,
        "held_out_agreements": int(agree),
        "held_out_contradictions": int(disagree),
        "held_out_comparisons": int(total),
        "seed": args.seed,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "overlap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"draw": np.arange(args.permutations),
                  "column_permutation_match_fraction": draws}).to_csv(
        OUT / "overlap_null_draws.csv", index=False, lineterminator="\n")
    pd.DataFrame({"patient_index_857": np.arange(len(y857)),
                  "n_compatible_336_records": partners,
                  "matched_336_index": pairing}).to_csv(
        OUT / "overlap_per_patient.csv", index=False, lineterminator="\n")
    print(f"\nwrote overlap_summary.json, overlap_null_draws.csv, "
          f"overlap_per_patient.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
