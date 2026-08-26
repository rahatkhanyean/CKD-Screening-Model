"""Task 4: build literature.csv from OpenAlex records plus targeted checks.

Fields are populated only where the retrieved text supports them. Where
OpenAlex gives a title and abstract but not full text --- which is the
common case --- fields that can only be settled from the Methods section
(exact release used, whether creatinine was a predictor, cross-release
validation) are recorded as "unclear" rather than guessed.

That means most rows will be mostly "unclear". That is the honest result of
abstract-level screening, and it is preferable to a table that looks
complete because its gaps were filled by inference.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "openalex"
OUT = ROOT / "outputs"

#: Cues in title/abstract. Each maps to the field it can settle.
RELEASE_857 = ("risk factor prediction of chronic kidney disease",
               "enam", "200 patients", "28 clinical features",
               "c5wp64")
RELEASE_336 = ("400 instances", "400 records", "400 patients", "c5g020",
               "apollo")
MERGE_CUES = ("merged", "merging", "combined dataset", "combining both")
CROSS_CUES = ("cross-dataset", "cross dataset", "external validation",
              "validated on the other", "second dataset", "another dataset")
SC_CUES = ("serum creatinine", "creatinine")
EGFR_CUES = ("egfr", "estimated glomerular", "glomerular filtration")


def abstract_text(work: dict) -> str:
    index = work.get("abstract_inverted_index")
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in index.items():
        positions.extend((spot, word) for spot in spots)
    return " ".join(word for _, word in sorted(positions))


def any_of(text: str, cues) -> bool:
    return any(c in text for c in cues)


def accuracy_from(text: str) -> str:
    """Highest reported percentage-like figure in the abstract, if any."""
    hits = re.findall(r"(\d{2}\.?\d*)\s*%", text)
    values = [float(h) for h in hits if 50.0 <= float(h) <= 100.0]
    if not values:
        return "not stated in abstract"
    return f"{max(values):.2f}% (max figure in abstract; metric unverified)"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = RAW / "citing_native.json"
    if not path.is_file():
        print("citing_native.json missing; run 05_openalex_trace.py first")
        return 1
    works = json.loads(path.read_text("utf-8")).get("results", [])
    print(f"screening {len(works)} works citing the id=857 native paper")

    rows = []
    for work in works:
        title = work.get("title") or ""
        abstract = abstract_text(work)
        text = f"{title} {abstract}".lower()

        uses_857 = any_of(text, RELEASE_857)
        uses_336 = any_of(text, RELEASE_336)
        merged = any_of(text, MERGE_CUES)
        # "merged" only counts when the abstract also references one of the
        # two releases; otherwise the word is about something else.
        if merged and (uses_857 or uses_336):
            release = "merged (abstract says so)"
        elif uses_857 and uses_336:
            release = "both (abstract mentions cues for each)"
        elif uses_857:
            release = "857"
        elif uses_336:
            release = "336"
        else:
            release = "unclear"

        cross = ("claimed" if any_of(text, CROSS_CUES)
                 else "not stated in abstract")
        creat = ("yes" if any_of(text, SC_CUES)
                 else "not stated in abstract")
        egfr = ("yes" if any_of(text, EGFR_CUES)
                else "not stated in abstract")

        # A paper is flagged only when the abstract itself indicates a
        # cross-release or merged design. Per Task 3, such a design is
        # non-independent.
        # The flag requires the abstract to reference one of the two CKD
        # releases AND to indicate a merged or cross-release design. An
        # earlier version flagged on the merge cue alone, which caught three
        # papers whose abstracts used "merged" about something else
        # entirely --- a breast-cancer study and a kidney-cancer survey. A
        # word is not a design.
        flag = ""
        if (uses_857 or uses_336) and (merged or cross == "claimed"):
            flag = ("NON-INDEPENDENT per Task 3: cross-release or merged "
                    "design indicated in the abstract")

        rows.append({
            "citation": title,
            "year": work.get("publication_year"),
            "doi_or_id": work.get("doi") or work.get("id"),
            "release_used": release,
            "headline_accuracy": accuracy_from(text),
            "uses_serum_creatinine": creat,
            "uses_egfr": egfr,
            "cross_or_external_validation_claimed": cross,
            "flag": flag,
            "evidence_level": ("title+abstract only (OpenAlex); Methods not "
                               "retrieved"),
            "quote": abstract[:500],
        })

    frame = pd.DataFrame(rows).sort_values(
        ["year", "citation"], ascending=[False, True])
    frame["screened_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    out = OUT / "literature.csv"
    frame.to_csv(out, index=False, lineterminator="\n")

    print(f"wrote {out} ({len(frame)} rows)")
    print()
    print("release_used:")
    print(frame["release_used"].value_counts().to_string())
    print()
    print("cross/external validation claimed in abstract:")
    print(frame["cross_or_external_validation_claimed"]
          .value_counts().to_string())
    print()
    flagged = frame[frame["flag"] != ""]
    print(f"flagged as potentially non-independent: {len(flagged)}")
    for _, r in flagged.iterrows():
        print(f"  {r['year']} | {str(r['citation'])[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
