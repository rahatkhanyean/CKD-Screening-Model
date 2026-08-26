"""Task 2: fetch candidate papers and scan them for the Enam link.

Candidates come from targeted searches. Each is fetched, saved raw, and
scanned for the terms that would establish a link. The scan is mechanical:
it records which terms occur and extracts the surrounding sentence verbatim,
so the verdict in ``enam_trace.csv`` rests on quoted text rather than on
recollection.

A fetch that fails is recorded with its HTTP status. Inaccessible is a
finding; it is not the same as "no match", and the two are kept distinct in
the verdict field.
"""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "papers"
OUT = ROOT / "outputs"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; provenance-audit/1.0)"}

#: label -> URL. Chosen because each either (a) is Bangladeshi-authored CKD
#: ML work that might reuse the Enam data, or (b) surfaced in searches
#: pairing "Enam Medical College" with a CKD dataset.
CANDIDATES = {
    "arxiv_2406.06728_ensemble_xai":
        "https://arxiv.org/abs/2406.06728",
    "arxiv_2012.12089_deep_neural_network":
        "https://arxiv.org/abs/2012.12089",
    "uci_857_card":
        "https://archive.ics.uci.edu/dataset/857/"
        "risk+factor+prediction+of+chronic+kidney+disease",
    "opendatabay_renal_bangladesh":
        "https://www.opendatabay.com/data/healthcare/"
        "cebbc2bc-8b56-4866-8e09-85a810a31f98",
    "semanticscholar_islam2020_native":
        "https://api.semanticscholar.org/graph/v1/paper/"
        "a4b90dd7b9dfaffa9eff1efc6555bbce62a963d7"
        "?fields=title,abstract,year,venue,externalIds,citationCount,authors",
}

#: Terms whose presence would matter, and what each would establish.
TERMS = {
    "enam medical college": "names the hospital",
    "enam medical": "names the hospital (partial)",
    "savar": "names the town",
    "10.24432/c5wp64": "cites the id=857 DOI",
    "10.24432/c5g020": "cites the id=336 DOI",
    "risk factor prediction of chronic kidney disease": "names the id=857 title",
    "bangladesh": "names the country",
}


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def sentence_around(text: str, index: int, width: int = 320) -> str:
    lo = max(0, index - width // 2)
    hi = min(len(text), index + width // 2)
    return text[lo:hi].strip()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for label, url in CANDIDATES.items():
        print(f"\n--- {label}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=90)
            status = response.status_code
            body = response.text
        except Exception as exc:
            print(f"    REQUEST FAILED: {type(exc).__name__}")
            rows.append({"label": label, "url": url, "http_status":
                         f"failed:{type(exc).__name__}", "term": "",
                         "quote": "", "verdict": "inaccessible"})
            continue

        print(f"    HTTP {status}, {len(body)} chars")
        if status != 200 or not body:
            rows.append({"label": label, "url": url, "http_status": status,
                         "term": "", "quote": "", "verdict": "inaccessible"})
            continue

        path = RAW / f"{label}.html"
        path.write_text(body, encoding="utf-8", errors="replace")
        flat = strip_html(body)
        low = flat.lower()

        hits = 0
        for term, meaning in TERMS.items():
            idx = low.find(term)
            if idx == -1:
                continue
            hits += 1
            quote = sentence_around(flat, idx)
            print(f"    HIT [{term}] -> {quote[:120]}...")
            rows.append({
                "label": label, "url": url, "http_status": status,
                "term": term, "meaning": meaning, "quote": quote,
                "verdict": "",  # assigned below
            })
        if hits == 0:
            print("    no terms found")
            rows.append({"label": label, "url": url, "http_status": status,
                         "term": "", "meaning": "", "quote": "",
                         "verdict": "no_match"})

    frame = pd.DataFrame(rows)
    frame["scanned_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    path = OUT / "enam_scan_raw.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"\nwrote {path} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
