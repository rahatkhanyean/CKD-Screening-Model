"""Task 2 (cont.): systematic trace via the OpenAlex API.

Three queries, each reproducible from the URL recorded in the output:

1. Resolve the native paper by its DOI and record its citation count.
2. List every work citing the native paper.
3. Full-text search for "Enam Medical College", then intersect with CKD
   terms, to find third-party papers naming the hospital.

OpenAlex is used because it is open, has no key requirement, indexes
abstracts and (for many records) full text, and returns a stable work id
per record so the query can be re-run.

Every hit is written with the exact matched text where the API returns it.
Where OpenAlex gives only an inverted index of the abstract, it is
reconstructed before quoting, and the output says so.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "openalex"
OUT = ROOT / "outputs"

BASE = "https://api.openalex.org"
# OpenAlex asks for a contact address in the polite pool; a generic mailto is
# used rather than the user's, which must not be sent to third parties.
MAILTO = "mailto=openalex-polite-pool@example.org"

NATIVE_DOI = "10.1109/ICISS49785.2020.9315878"


def get(url: str, label: str) -> dict | None:
    RAW.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=90)
        except Exception as exc:
            print(f"    {label}: request failed {type(exc).__name__}")
            return None
        if response.status_code == 200:
            (RAW / f"{label}.json").write_text(response.text, encoding="utf-8")
            return response.json()
        if response.status_code in (429, 503):
            wait = 5 * (attempt + 1)
            print(f"    {label}: HTTP {response.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        print(f"    {label}: HTTP {response.status_code}")
        return None
    return None


def abstract_text(work: dict) -> str:
    """Reconstruct the abstract from OpenAlex's inverted index."""
    index = work.get("abstract_inverted_index")
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in index.items():
        positions.extend((spot, word) for spot in spots)
    return " ".join(word for _, word in sorted(positions))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # ---- 1. the native paper -----------------------------------------
    print("1. native paper by DOI")
    url = f"{BASE}/works/doi:{NATIVE_DOI}?{MAILTO}"
    native = get(url, "native_paper")
    native_id = None
    if native:
        native_id = native.get("id")
        print(f"    {native.get('title')}")
        print(f"    year={native.get('publication_year')} "
              f"cited_by={native.get('cited_by_count')}")
        rows.append({
            "query": "native_paper_by_doi", "openalex_id": native_id,
            "title": native.get("title"),
            "year": native.get("publication_year"),
            "doi": native.get("doi"),
            "cited_by_count": native.get("cited_by_count"),
            "matched_term": "N/A - resolved by DOI",
            "quote": abstract_text(native)[:600],
            "quote_source": "OpenAlex abstract (reconstructed from inverted index)",
            "url_or_doi": url,
        })

    # ---- 2. works citing the native paper -----------------------------
    if native_id:
        print("2. works citing the native paper")
        short = native_id.rsplit("/", 1)[-1]
        url = (f"{BASE}/works?filter=cites:{short}"
               f"&per-page=200&{MAILTO}")
        citing = get(url, "citing_native")
        results = (citing or {}).get("results", [])
        print(f"    {len(results)} citing works retrieved")
        for work in results:
            text = f"{work.get('title') or ''} {abstract_text(work)}".lower()
            names_enam = "enam" in text
            rows.append({
                "query": "cites_native_paper",
                "openalex_id": work.get("id"),
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "doi": work.get("doi"),
                "cited_by_count": work.get("cited_by_count"),
                "matched_term": "enam" if names_enam else "",
                "quote": (abstract_text(work)[:600] if names_enam else ""),
                "quote_source": ("OpenAlex abstract (reconstructed)"
                                 if names_enam else "title/abstract scanned, no hit"),
                "url_or_doi": work.get("doi") or work.get("id"),
            })

    # ---- 3. full-text search for the hospital -------------------------
    print("3. full-text search: Enam Medical College")
    for phrase, label in (('"Enam Medical College"', "search_enam_exact"),
                          ('"Enam Medical College" kidney', "search_enam_kidney")):
        url = (f"{BASE}/works?search={quote(phrase)}"
               f"&per-page=100&{MAILTO}")
        found = get(url, label)
        results = (found or {}).get("results", [])
        print(f"    {label}: {len(results)} results")
        for work in results:
            abstract = abstract_text(work)
            text = f"{work.get('title') or ''} {abstract}".lower()
            ckd = any(t in text for t in
                      ("kidney", "ckd", "renal", "nephro"))
            enam = "enam" in text
            rows.append({
                "query": label,
                "openalex_id": work.get("id"),
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "doi": work.get("doi"),
                "cited_by_count": work.get("cited_by_count"),
                "matched_term": ("enam+kidney" if (enam and ckd)
                                 else "enam" if enam
                                 else "kidney" if ckd else ""),
                "quote": abstract[:600],
                "quote_source": "OpenAlex abstract (reconstructed)",
                "url_or_doi": work.get("doi") or work.get("id"),
            })

    frame = pd.DataFrame(rows)
    frame["retrieved_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    path = OUT / "openalex_trace_raw.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"\nwrote {path} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
