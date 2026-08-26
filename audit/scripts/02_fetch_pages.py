"""Task 1 (cont.): fetch the dataset pages and legacy documentation files.

``ucimlrepo`` returns the curated metadata object. The legacy
``machine-learning-databases`` tree additionally carries the original
``.names`` files, which for older donations often hold the source statement
that the modern dataset card omits. Both are fetched and saved raw so that
extraction works from disk.

Every response is written verbatim, with its HTTP status and checksum
recorded. A 404 is itself a finding and is recorded rather than silently
skipped.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"

TARGETS = {
    "uci_336_page.html":
        "https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease",
    "uci_857_page.html":
        "https://archive.ics.uci.edu/dataset/857/"
        "risk+factor+prediction+of+chronic+kidney+disease",
    "uci_336_legacy_index.html":
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/",
    "uci_336_names.txt":
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/"
        "Chronic_Kidney_Disease.rar",
    "uci_336_static_index.html":
        "https://archive.ics.uci.edu/static/public/336/",
    "uci_857_static_index.html":
        "https://archive.ics.uci.edu/static/public/857/",
}

HEADERS = {"User-Agent": "provenance-audit/1.0 (research; contact via repo)"}


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in TARGETS.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            status = response.status_code
            body = response.content
        except Exception as exc:
            print(f"  {name}: REQUEST FAILED {type(exc).__name__}")
            rows.append({"file": name, "url": url, "status": "request failed",
                         "bytes": 0, "sha256": ""})
            continue

        path = RAW / name
        if status == 200 and body:
            path.write_bytes(body)
            digest = hashlib.sha256(body).hexdigest()
            print(f"  {name}: {status}, {len(body)} bytes")
        else:
            digest = ""
            print(f"  {name}: {status} (not saved)")
        rows.append({"file": name, "url": url, "status": status,
                     "bytes": len(body), "sha256": digest})

    frame = pd.DataFrame(rows)
    frame["fetched_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    frame.to_csv(RAW / "page_fetches.csv", index=False, lineterminator="\n")
    print(f"\nwrote page_fetches.csv ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
