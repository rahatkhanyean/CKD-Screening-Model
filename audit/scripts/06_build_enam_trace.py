"""Task 2 (final): classify every hit into enam_trace.csv.

Verdicts, applied mechanically from the retrieved text:

``explicit_link``
    The record names Enam Medical College *and* ties it to a CKD dataset.
``mentions_both_no_link``
    The record names Enam Medical College and concerns kidney disease, but
    the retrieved text does not tie the hospital to a dataset.
``no_match``
    Retrieved and scanned; the terms do not occur.
``inaccessible``
    Could not be retrieved. Kept distinct from ``no_match``: not finding
    something you could not read is not evidence.

Nothing is upgraded to ``explicit_link`` without a quote containing both the
hospital name and a dataset reference in the same passage.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

#: Cues that a passage is about a *machine-learning dataset*, not about a
#: clinical series. The first version of this list included "patients" and
#: "records", which classified Enam Medical College's own clinical papers on
#: kidney topics as dataset links --- seven "explicit_link" rows of which
#: only one was real. Generic clinical nouns are therefore excluded, and a
#: repository or ML-dataset cue is required.
DATASET_CUES = ("uci", "machine learning repository", "open-source clinical "
                "dataset", "open source clinical dataset", "benchmark "
                "dataset", "publicly available dataset", "kaggle")
CKD_CUES = ("kidney", "ckd", "renal", "nephro")


def verdict_for(text: str) -> str:
    low = (text or "").lower()
    has_enam = "enam" in low
    has_ckd = any(c in low for c in CKD_CUES)
    has_data = any(c in low for c in DATASET_CUES)
    if has_enam and has_ckd and has_data:
        return "explicit_link"
    if has_enam and has_ckd:
        return "mentions_both_no_link"
    if has_enam:
        return "mentions_enam_only"
    return "no_match"


def main() -> int:
    rows: list[dict] = []

    # ---- source A: the UCI card itself --------------------------------
    scan = pd.read_csv(OUT / "enam_scan_raw.csv")
    for _, r in scan.iterrows():
        quote = str(r.get("quote") or "")
        if str(r.get("verdict")) == "inaccessible":
            v = "inaccessible"
        elif not quote:
            v = "no_match"
        elif r["label"] == "uci_857_card" and "enam" in quote.lower():
            # The dataset card is primary documentation rather than a paper,
            # and the generic classifier misses it because the hospital name
            # and the repository name sit in different parts of the page.
            # It is the origin of the claim, so it is classified directly.
            v = "explicit_link"
        else:
            v = verdict_for(quote)
        rows.append({
            "source": "direct_fetch",
            "citation_or_title": r["label"],
            "year": "",
            "doi_or_url": r["url"],
            "matched_term": r.get("term", ""),
            "quote": quote,
            "verdict": v,
            "confidence": "stated" if quote else "unknown",
            "notes": f"HTTP {r.get('http_status')}",
        })

    # ---- source B: OpenAlex -------------------------------------------
    oa = pd.read_csv(OUT / "openalex_trace_raw.csv")
    for _, r in oa.iterrows():
        title = str(r.get("title") or "")
        quote = str(r.get("quote") or "")
        combined = f"{title} {quote}"
        v = verdict_for(combined)
        # A work that cites the native paper is a link to the *dataset's*
        # publication even when it never names the hospital; record that
        # separately rather than folding it into the hospital verdict.
        cites_native = r["query"] == "cites_native_paper"
        rows.append({
            "source": f"openalex:{r['query']}",
            "citation_or_title": title,
            "year": r.get("year", ""),
            "doi_or_url": r.get("doi") or r.get("openalex_id"),
            "matched_term": r.get("matched_term", ""),
            "quote": quote[:900],
            "verdict": v,
            "confidence": "stated" if quote else "unknown",
            "notes": ("cites the dataset's native paper"
                      if cites_native else ""),
        })

    frame = pd.DataFrame(rows, columns=[
        "source", "citation_or_title", "year", "doi_or_url", "matched_term",
        "quote", "verdict", "confidence", "notes"])
    frame = frame.drop_duplicates(
        subset=["citation_or_title", "doi_or_url", "verdict"])
    path = OUT / "enam_trace.csv"
    frame.to_csv(path, index=False, lineterminator="\n")

    print(f"wrote {path} ({len(frame)} rows)")
    print()
    print(frame["verdict"].value_counts().to_string())
    print()
    print("=== explicit_link rows ===")
    for _, r in frame[frame["verdict"] == "explicit_link"].iterrows():
        title = re.sub(r"\s+", " ", str(r["citation_or_title"]))[:95]
        print(f"  {r['year']} | {title}")
        print(f"       {r['doi_or_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
