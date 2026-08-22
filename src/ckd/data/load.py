"""Loading the raw CSV and removing the two non-patient metadata rows.

The released file has this layout::

    line 1  : column header
    line 2  : "discrete,discrete,...,discrete"        <- metadata, not a patient
    line 3  : ",,,...,class,meta"                     <- metadata, not a patient
    line 4+ : 200 patient records

Both metadata rows are removed here and nowhere else, so there is exactly one
place in the codebase that decides what counts as a patient record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve


@dataclass(frozen=True)
class RawLoadResult:
    """Raw file contents plus provenance information."""

    raw: pd.DataFrame          # everything below the header, including metadata rows
    metadata: pd.DataFrame     # the removed metadata rows
    patients: pd.DataFrame     # patient records only
    source_path: Path
    sha256: str

    @property
    def shape_before(self) -> tuple[int, int]:
        return self.raw.shape

    @property
    def shape_after(self) -> tuple[int, int]:
        return self.patients.shape


def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file, used to pin the exact input in the reports."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str | Path | None = None) -> RawLoadResult:
    """Read the raw CSV and split metadata rows from patient rows.

    Everything is read as ``str`` with ``keep_default_na=False`` so that pandas
    never silently reinterprets a cell (e.g. turning the string ``"na"`` into a
    float ``NaN``, or coercing ``"1 - 1"`` in one column but not another).
    Missing-value decisions are made explicitly in :mod:`ckd.data.clean`.
    """
    cfg = load_config()
    src = Path(path) if path is not None else resolve(cfg["paths"]["raw_csv"])
    n_meta = int(cfg["data"]["n_metadata_rows"])

    raw = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8")
    metadata = raw.iloc[:n_meta].copy()
    patients = raw.iloc[n_meta:].copy().reset_index(drop=True)

    # Provenance: keep the original 1-based CSV line number of every patient so
    # that any finding can be traced back to a physical line in the source file.
    patients.insert(0, "source_csv_line", range(n_meta + 2, n_meta + 2 + len(patients)))

    return RawLoadResult(
        raw=raw,
        metadata=metadata,
        patients=patients,
        source_path=src,
        sha256=sha256_of(src),
    )


def verify_metadata_rows(metadata: pd.DataFrame) -> dict[str, object]:
    """Sanity-check that the rows we dropped really are the metadata rows.

    Returns a dictionary of evidence rather than raising, so the data-quality
    report can show *why* the rows were classified as metadata.
    """
    row0 = set(str(v).strip() for v in metadata.iloc[0].tolist()) if len(metadata) > 0 else set()
    row1 = set(str(v).strip() for v in metadata.iloc[1].tolist()) if len(metadata) > 1 else set()
    return {
        "n_metadata_rows": int(len(metadata)),
        "row0_unique_values": sorted(row0),
        "row1_unique_values": sorted(row1),
        "row0_is_type_declaration": row0 == {"discrete"},
        "row1_declares_target": "class" in row1,
        "row1_declares_meta": "meta" in row1,
    }
