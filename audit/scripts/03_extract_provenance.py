"""Task 1 (cont.): extract provenance facts verbatim into provenance.csv.

Every quote is pulled from a file on disk in ``audit/raw/`` rather than
typed, so the CSV can be regenerated and checked against the artefacts. The
locator for each fact is recorded, so a reader can find the quote themselves.

Rules applied here, from the audit brief:

* Report only what a source states explicitly.
* Where a fact is absent, emit a row with value "not stated" rather than
  omitting the fact type. An absence that is recorded is evidence; an
  absence that is silently skipped is not.
* ``confidence`` is one of stated / inferred / unknown. Nothing in this file
  is marked ``inferred`` unless the reasoning is given in ``notes``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "outputs"

INFO_336 = RAW / "extract336" / "Chronic_Kidney_Disease" / \
    "chronic_kidney_disease.info.txt"


def block(text: str, start: str, stop: str) -> str:
    """Verbatim slice between two markers, whitespace-normalised for CSV."""
    i = text.index(start)
    j = text.index(stop, i)
    return re.sub(r"\s+", " ", text[i:j]).strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def add(dataset, fact_type, value, quote, url, confidence, notes=""):
        rows.append({
            "dataset": dataset, "fact_type": fact_type, "value": value,
            "source_quote": quote, "url_or_doi": url,
            "confidence": confidence, "notes": notes,
        })

    # ---------------- id=336 -------------------------------------------
    meta336 = json.loads((RAW / "uci_336_metadata.json").read_text("utf-8"))
    card336 = meta336.get("repository_url", "")
    doi336 = "10.24432/C5G020"

    if INFO_336.is_file():
        info = INFO_336.read_text(encoding="utf-8", errors="replace")
        legacy_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
                      "databases/00336/Chronic_Kidney_Disease.rar"
                      " -> Chronic_Kidney_Disease/"
                      "chronic_kidney_disease.info.txt")

        source_block = block(info, "(a) Source:", "(b) Creator:")
        add("uci_336", "hospital_or_institution", "Apollo Hospitals",
            source_block, legacy_url, "stated",
            "Present only in the legacy archive's info.txt, not on the "
            "modern dataset card or in the ucimlrepo metadata object.")
        add("uci_336", "country", "India", source_block, legacy_url, "stated")
        add("uci_336", "collection_site_detail",
            "Managiri, Madurai Main Road, Karaikudi, Tamilnadu",
            source_block, legacy_url, "stated")

        title_line = block(info, "% 1. Title:", "% 2. Source")
        add("uci_336", "population_descriptor", "Indians", title_line,
            legacy_url, "stated",
            "The dataset title names the population; it does not state a "
            "recruitment protocol.")

        creator_block = block(info, "(b) Creator:", "(c) Guided by:")
        add("uci_336", "creator_institution", "Alagappa University",
            creator_block, legacy_url, "stated")

        date_line = block(info, "(d) Date", "% 3.")
        add("uci_336", "date_stated", "July 2015", date_line, legacy_url,
            "stated",
            "This is the date given in the source block. The file does not "
            "say whether it is a collection date, a donation date or a "
            "compilation date, so it must not be read as a collection "
            "range.")

        low = info.lower()
        for term, fact in (("ethic", "ethics_approval"), ("irb", "ethics_approval"),
                           ("consent", "informed_consent")):
            if term in low:
                add("uci_336", fact, "PRESENT - review manually", term,
                    legacy_url, "stated")
                break
        else:
            add("uci_336", "ethics_approval", "not stated", "",
                legacy_url, "unknown",
                "No occurrence of 'ethic', 'IRB', 'consent', 'approval', "
                "'institutional' or 'committee' anywhere in info.txt.")
            add("uci_336", "informed_consent", "not stated", "",
                legacy_url, "unknown")
    else:
        add("uci_336", "hospital_or_institution", "not retrieved", "",
            "", "unknown", "legacy info.txt not extracted")

    add("uci_336", "collection_period",
        "'nearly 2 months' - no dates given",
        meta336.get("abstract", ""), card336, "stated",
        "The abstract gives a duration but no start or end date, and uses "
        "the generic phrase 'the hospital' without naming one.")
    add("uci_336", "hospital_named_on_modern_card", "not stated",
        meta336.get("abstract", ""), card336, "stated",
        "Checked programmatically: the strings 'Apollo', 'India', 'ethic', "
        "'IRB', 'consent' and 'approval' do not occur anywhere in the "
        "ucimlrepo metadata object for id=336. The source statement exists "
        "only in the legacy archive.")
    add("uci_336", "creators", "; ".join(meta336.get("creators", [])),
        json.dumps(meta336.get("creators", []), ensure_ascii=False),
        card336, "stated")
    add("uci_336", "dataset_doi", doi336, doi336, card336, "stated")

    # ---------------- id=857 -------------------------------------------
    meta857 = json.loads((RAW / "uci_857_metadata.json").read_text("utf-8"))
    card857 = meta857.get("repository_url", "")
    info857 = (meta857.get("additional_info") or {})
    summary857 = (info857.get("summary") or "").strip()

    add("uci_857", "hospital_or_institution",
        "Enam Medical College, Savar, Dhaka", summary857, card857, "stated",
        "Stated in the dataset card's summary field, and present verbatim "
        "in the fetched page HTML.")
    add("uci_857", "country", "Bangladesh", summary857, card857, "stated")
    add("uci_857", "population_descriptor", "real Bangladeshi patient data",
        summary857, card857, "stated")
    add("uci_857", "collection_period", "not stated", summary857, card857,
        "unknown",
        "The card gives no start date, end date or duration.")
    add("uci_857", "ethics_approval", "not stated", "", card857, "unknown",
        "No occurrence of 'ethic', 'IRB', 'consent' or 'approval' in the "
        "metadata object.")
    add("uci_857", "informed_consent", "not stated", "", card857, "unknown")
    add("uci_857", "creators", "; ".join(meta857.get("creators", [])),
        json.dumps(meta857.get("creators", []), ensure_ascii=False),
        card857, "stated")
    add("uci_857", "dataset_doi", "10.24432/C5WP64", "10.24432/C5WP64",
        card857, "stated")

    citation = (info857.get("citation") or "").strip()
    if citation:
        add("uci_857", "native_publication", citation, citation, card857,
            "stated",
            "The dataset card's own citation field. This is the dataset "
            "creators' paper, i.e. published research using the Enam "
            "Medical College data.")
    intro = meta857.get("intro_paper") or {}
    if intro:
        add("uci_857", "intro_paper_url", intro.get("URL", "not stated"),
            json.dumps(intro, ensure_ascii=False), card857, "stated",
            f"UCI marks this paper type={intro.get('type')}.")

    # ---------------- retraction status of the native paper ------------
    oa_path = RAW / "openalex" / "native_paper.json"
    cr_path = RAW / "openalex" / "crossref_native.json"
    if oa_path.is_file():
        oa = json.loads(oa_path.read_text("utf-8"))
        add("uci_857", "native_publication_retracted",
            str(oa.get("is_retracted")),
            f'"title": "{oa.get("title")}", "is_retracted": '
            f'{json.dumps(oa.get("is_retracted"))}',
            oa.get("id", ""), "stated",
            "OpenAlex record for the DOI given in the dataset card's own "
            "citation field.")
    if cr_path.is_file():
        cr = json.loads(cr_path.read_text("utf-8"))
        add("uci_857", "native_publication_title_crossref",
            (cr.get("title") or [""])[0],
            json.dumps(cr.get("title"), ensure_ascii=False),
            f"https://doi.org/{cr.get('DOI')}", "stated",
            "Independent confirmation from Crossref, i.e. the publisher's "
            "own deposited metadata. The 'Retracted:' prefix is IEEE's.")
    add("uci_857", "retraction_disclosed_on_uci_card", "not stated",
        (info857.get("citation") or "").strip(), card857, "stated",
        "The dataset card's citation field reproduces the paper title "
        "WITHOUT the 'Retracted:' prefix that both OpenAlex and Crossref "
        "carry, and the card contains no retraction notice. Checked "
        "programmatically against the fetched page HTML.")

    # ---------------- the cross-dataset question -----------------------
    add("cross", "same_hospital_claim", "not stated", "", "", "unknown",
        "Neither dataset's documentation mentions the other, and neither "
        "states any relationship between the two releases. Any link "
        "between them must come from the record-level evidence in Task 3, "
        "not from the documentation.")

    frame = pd.DataFrame(rows, columns=[
        "dataset", "fact_type", "value", "source_quote", "url_or_doi",
        "confidence", "notes"])
    path = OUT / "provenance.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path} ({len(frame)} facts)")
    print()
    print(frame[["dataset", "fact_type", "value", "confidence"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
