"""Task 1: fetch both UCI CKD releases and capture their metadata verbatim.

Downloads id=336 and id=857 through ``ucimlrepo``, checksums every file it
writes, and dumps the complete metadata object as JSON so that later
extraction works from a saved artefact rather than a live request.

Nothing here interprets the metadata. Extraction into provenance.csv is a
separate step (02_extract_provenance.py) so that the raw capture can be
re-read and re-parsed without re-downloading.

Outputs
-------
audit/raw/uci_<id>_metadata.json     full metadata object as returned
audit/raw/uci_<id>_features.csv      variable table as returned
audit/raw/uci_<id>_X.csv             feature matrix
audit/raw/uci_<id>_y.csv             target
audit/raw/checksums.csv              sha256 of everything written
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
DATASETS = {336: "Chronic Kidney Disease (Rubini et al. 2015)",
            857: "Risk Factor Prediction of CKD (Islam & Akter 2020)"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []

    for uci_id, label in DATASETS.items():
        print(f"\n=== id={uci_id}: {label} ===")
        try:
            repo = fetch_ucirepo(id=uci_id)
        except Exception as exc:
            print(f"  FETCH FAILED: {type(exc).__name__}: {exc}")
            written.append({"file": f"uci_{uci_id}", "sha256": "",
                            "status": f"fetch failed: {type(exc).__name__}"})
            continue

        # The metadata object is a dict-like; dump it whole and unmodified.
        meta = dict(repo.metadata)
        meta_path = RAW / f"uci_{uci_id}_metadata.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8")
        print(f"  wrote {meta_path.name} ({len(json.dumps(meta, default=str))} bytes)")

        variables = repo.variables
        if variables is not None:
            vpath = RAW / f"uci_{uci_id}_variables.csv"
            variables.to_csv(vpath, index=False, lineterminator="\n")
            print(f"  wrote {vpath.name} ({len(variables)} variables)")

        X, y = repo.data.features, repo.data.targets
        if X is not None:
            xpath = RAW / f"uci_{uci_id}_X.csv"
            X.to_csv(xpath, index=False, lineterminator="\n")
            print(f"  wrote {xpath.name} shape={X.shape}")
        if y is not None:
            ypath = RAW / f"uci_{uci_id}_y.csv"
            y.to_csv(ypath, index=False, lineterminator="\n")
            print(f"  wrote {ypath.name} shape={y.shape}")

        # Print the fields most likely to carry provenance, so the console
        # log itself is a record of what was there on this date.
        for key in ("name", "abstract", "area", "creators", "intro_paper",
                    "additional_info", "year_of_dataset_creation",
                    "dataset_doi", "external_url", "repository_url"):
            if key in meta and meta[key]:
                value = meta[key]
                shown = json.dumps(value, default=str, ensure_ascii=False)
                print(f"  [{key}] {shown[:400]}"
                      f"{'...' if len(shown) > 400 else ''}")

    for path in sorted(RAW.glob("*")):
        if path.is_file() and path.name != "checksums.csv":
            written.append({"file": path.name, "sha256": sha256(path),
                            "bytes": path.stat().st_size, "status": "ok"})

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame = pd.DataFrame(written)
    frame["downloaded_utc"] = stamp
    frame.to_csv(RAW / "checksums.csv", index=False, lineterminator="\n")
    print(f"\nwrote checksums.csv ({len(frame)} files) at {stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
