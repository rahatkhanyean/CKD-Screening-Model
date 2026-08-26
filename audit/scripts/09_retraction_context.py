"""Task 2 (addendum): is the retraction paper-specific or venue-wide?

A retraction flag on one paper reads very differently depending on whether
it stands alone or belongs to a batch. IEEE periodically retracts groups of
conference papers, in which case the flag says something about the venue's
process rather than about the individual paper or its data.

This script counts, from Crossref, how many proceedings articles of the same
conference carry a "Retracted:" title prefix. It draws no conclusion about
*why* any paper was retracted --- that requires the retraction notice, which
was not retrievable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "openalex"
OUT = ROOT / "outputs"

QUERY = ("https://api.crossref.org/works?query.container-title="
         "Intelligent+Sustainable+Systems&filter=from-pub-date:2020-01-01,"
         "until-pub-date:2021-06-30,type:proceedings-article&rows=200")
TARGET_DOI = "10.1109/iciss49785.2020.9315878"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    response = requests.get(QUERY, timeout=90,
                            headers={"User-Agent": "provenance-audit/1.0"})
    print(f"crossref HTTP {response.status_code}")
    if response.status_code != 200:
        return 1
    payload = response.json()["message"]
    (RAW / "iciss_proceedings.json").write_text(
        json.dumps(payload, indent=2)[:4_000_000], encoding="utf-8")

    items = payload.get("items", [])
    rows = []
    for item in items:
        title = (item.get("title") or [""])[0]
        container = "; ".join(item.get("container-title") or [])
        if "Intelligent Sustainable Systems" not in container:
            continue
        rows.append({
            "doi": item.get("DOI", ""),
            "title": title,
            "retracted_prefix": title.lower().startswith("retracted"),
            "is_target_paper": item.get("DOI", "").lower() == TARGET_DOI,
            "container": container,
        })

    frame = pd.DataFrame(rows)
    frame["retrieved_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    frame.to_csv(OUT / "retraction_context.csv", index=False,
                 lineterminator="\n")

    n = len(frame)
    r = int(frame["retracted_prefix"].sum())
    print(f"ICISS proceedings articles sampled: {n}")
    print(f"carrying a 'Retracted:' title prefix: {r} "
          f"({r / max(n, 1):.1%})")
    print(f"target paper present in sample: "
          f"{bool(frame['is_target_paper'].any())}")
    print()
    print("Interpretation limit: this counts retraction PREFIXES in a "
          "Crossref sample of up to 200 records. It shows the target paper "
          "is not the only retracted article in this proceedings. It does "
          "NOT establish why any of them were retracted, and it is not a "
          "complete census of the proceedings.")
    print(f"\nwrote {OUT / 'retraction_context.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
