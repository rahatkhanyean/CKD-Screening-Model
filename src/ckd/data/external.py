"""External-dataset registry, verification and loading (Phase 0.2).

This module is the *gatekeeper* for every dataset that is not the internal
``ckd-dataset-v2.csv``. It mirrors the design philosophy of the internal
pipeline:

* **Declaration first.** Every candidate dataset is declared in
  ``config/external_datasets.yaml`` with its provenance, license, checksums,
  cohort arithmetic, and — critically — its *own* prohibited-column sets
  (the source's equivalents of our ``class``/``affected`` and
  ``stage``/``grf``). Nothing is inferred silently.
* **Fail loudly.** Checksums are verified on load, exactly as the internal
  raw file is; a mismatch raises rather than warns.
* **Provenance before use.** Whether a dataset may support *external*
  claims is not decided here: that verdict is computed by
  ``scripts/00_provenance.py`` and written to
  ``reports/tables/table_24_provenance.csv``. The gate test in
  ``tests/test_external.py`` enforces that no dataset lacking an
  INDEPENDENT verdict ever appears in an external-validation table.

Known data defects are handled the way the internal cleaning code handles
them: corrected where the correction is unambiguous, and *recorded* in the
returned result rather than silently absorbed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import project_root

#: Allowed values for DatasetRecord.status.
STATUSES = frozenset({"OBTAINED", "RESTRICTED"})

#: Allowed values for DatasetRecord.design.
DESIGNS = frozenset(
    {
        "cross-sectional-diagnostic",
        "prospective-incident",
        "longitudinal-repeated",
        "ehr-extract",
    }
)

#: Provenance verdicts the gate recognises. Only INDEPENDENT datasets may
#: ever contribute to an external-validation table.
VERDICTS = frozenset({"INDEPENDENT", "OVERLAPPING", "SAME-SOURCE"})


class ExternalDataError(RuntimeError):
    """Raised when an external dataset fails verification or loading."""


@dataclass(frozen=True)
class RawFile:
    """One checksummed raw file belonging to a dataset."""

    path: str
    sha256: str


@dataclass(frozen=True)
class DatasetRecord:
    """One registered external dataset, as declared in the registry YAML."""

    dataset_id: str
    citation: str
    doi: str
    url: str
    license: str
    country: str
    design: str
    status: str
    raw_files: tuple[RawFile, ...]
    outcome_columns: tuple[str, ...]
    post_diagnosis_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    grouped_split_by: str | None
    expected: dict[str, Any]
    notes: str
    acquisition_note: str = ""

    @property
    def prohibited_columns(self) -> frozenset[str]:
        """This source's full prohibited set: outcome + post-diagnosis.

        This is what must be handed to :class:`ckd.features.encoders.LeakageGuard`
        (as ``forbidden``) for any pipeline built on this source.
        """
        return frozenset(self.outcome_columns) | frozenset(self.post_diagnosis_columns)


@lru_cache(maxsize=None)
def load_registry(path: str | None = None) -> dict[str, DatasetRecord]:
    """Load and validate the registry from ``config/external_datasets.yaml``."""
    cfg_path = Path(path) if path is not None else project_root() / "config" / "external_datasets.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    registry: dict[str, DatasetRecord] = {}
    for dataset_id, entry in raw["datasets"].items():
        record = DatasetRecord(
            dataset_id=dataset_id,
            citation=str(entry["citation"]).strip(),
            doi=str(entry["doi"]),
            url=str(entry["url"]),
            license=str(entry["license"]),
            country=str(entry["country"]),
            design=str(entry["design"]),
            status=str(entry["status"]),
            raw_files=tuple(
                RawFile(path=f["path"], sha256=f["sha256"])
                for f in (entry.get("raw_files") or ())
            ),
            outcome_columns=tuple(entry.get("outcome_columns") or ()),
            post_diagnosis_columns=tuple(entry.get("post_diagnosis_columns") or ()),
            identifier_columns=tuple(entry.get("identifier_columns") or ()),
            grouped_split_by=entry.get("grouped_split_by"),
            expected=dict(entry.get("expected") or {}),
            notes=str(entry.get("notes", "")).strip(),
            acquisition_note=str(entry.get("acquisition_note", "")).strip(),
        )
        if record.status not in STATUSES:
            raise ExternalDataError(
                f"{dataset_id}: status {record.status!r} not in {sorted(STATUSES)}"
            )
        if record.design not in DESIGNS:
            raise ExternalDataError(
                f"{dataset_id}: design {record.design!r} not in {sorted(DESIGNS)}"
            )
        if record.status == "OBTAINED" and not record.raw_files:
            raise ExternalDataError(f"{dataset_id}: OBTAINED but no raw_files declared")
        if record.status == "RESTRICTED" and not record.acquisition_note:
            raise ExternalDataError(f"{dataset_id}: RESTRICTED but no acquisition_note")
        if not record.outcome_columns:
            raise ExternalDataError(f"{dataset_id}: no outcome_columns declared")
        registry[dataset_id] = record
    return registry


def get_dataset(dataset_id: str) -> DatasetRecord:
    """Return one registered dataset or raise with the known ids."""
    registry = load_registry()
    if dataset_id not in registry:
        raise ExternalDataError(
            f"Unknown dataset {dataset_id!r}. Registered: {sorted(registry)}"
        )
    return registry[dataset_id]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_files(record: DatasetRecord) -> list[Path]:
    """Verify presence and SHA-256 of every declared raw file.

    Returns the resolved paths; raises :class:`ExternalDataError` on the
    first missing file or checksum mismatch. RESTRICTED datasets (no files)
    verify trivially.
    """
    root = project_root()
    resolved: list[Path] = []
    for raw in record.raw_files:
        path = root / raw.path
        if not path.is_file():
            raise ExternalDataError(f"{record.dataset_id}: missing raw file {raw.path}")
        actual = _sha256(path)
        if actual != raw.sha256:
            raise ExternalDataError(
                f"{record.dataset_id}: checksum mismatch for {raw.path}: "
                f"expected {raw.sha256}, found {actual}. The raw file must "
                "never be modified; restore it from the original source."
            )
        resolved.append(path)
    return resolved


@dataclass(frozen=True)
class ExternalLoadResult:
    """A loaded external dataset plus everything worth reporting about it."""

    dataset_id: str
    frame: pd.DataFrame
    target: pd.Series  # 1 = CKD / event, 0 = not
    defects: tuple[str, ...] = field(default=())


def load_uci2015() -> ExternalLoadResult:
    """Load the UCI-2015 (id 336) table with its known label defect handled.

    Two records carry the label ``'ckd\\t'`` (trailing tab). The tab is
    stripped — the correction is unambiguous — and the defect is recorded in
    the result rather than silently absorbed. All other columns are returned
    as read; downstream encoding decisions belong to the harmonization
    layer, not the loader.
    """
    record = get_dataset("uci2015")
    (csv_path, _zip) = verify_raw_files(record)
    frame = pd.read_csv(csv_path)

    defects: list[str] = []
    raw_labels = frame["class"].astype(str)
    dirty = raw_labels[raw_labels != raw_labels.str.strip()]
    if len(dirty):
        defects.append(
            f"{len(dirty)} class label(s) carried surrounding whitespace "
            f"({sorted(set(map(repr, dirty)))}); stripped."
        )
    labels = raw_labels.str.strip()
    bad = sorted(set(labels) - {"ckd", "notckd"})
    if bad:
        raise ExternalDataError(f"uci2015: unexpected class labels {bad}")
    target = (labels == "ckd").astype(int)
    return ExternalLoadResult("uci2015", frame, target, tuple(defects))


def load_th_uae() -> ExternalLoadResult:
    """Load the TH (Tawam Hospital, UAE) prospective incident-CKD table."""
    record = get_dataset("th_uae")
    (xlsx_path,) = verify_raw_files(record)
    frame = pd.read_excel(xlsx_path)
    # One column ships with a trailing space in its header; normalise and say so.
    defects: list[str] = []
    stripped = {c: c.strip() for c in frame.columns if c != c.strip()}
    if stripped:
        defects.append(f"column header(s) with stray whitespace normalised: {stripped}")
        frame = frame.rename(columns=stripped)
    target = frame["EventCKD35"].astype(int)
    return ExternalLoadResult("th_uae", frame, target, tuple(defects))


def load_birdem() -> ExternalLoadResult:
    """Load the BIRDEM longitudinal table (4,000 rows, 400 patients).

    The returned target is ROW-level (the label as recorded in that yearly
    row). Patient-level ever-CKD must be derived explicitly by the caller —
    the two disagree (386 positive rows vs 185 ever-CKD patients) and which
    one is correct depends on the question being asked. Any resampling MUST
    group by ``Patient ID`` (see the registry's ``grouped_split_by``).
    """
    record = get_dataset("birdem")
    (csv_path,) = verify_raw_files(record)
    frame = pd.read_csv(csv_path)
    target = frame["CKD (1-yes, 0-no)"].astype(int)
    return ExternalLoadResult("birdem", frame, target)


#: Loaders for every OBTAINED tabular dataset. mimic_iv_demo is deliberately
#: absent: it is a relational extract whose ETL (and only then a loader)
#: belongs to Phase 3.
LOADERS = {
    "uci2015": load_uci2015,
    "th_uae": load_th_uae,
    "birdem": load_birdem,
}


def dataset_seed(master_seed: int, dataset_id: str) -> int:
    """Derive a per-dataset seed from the master seed, deterministically.

    Platform-stable (sha256, not ``hash()``) so the same master seed gives
    the same partitions on any machine, and distinct datasets get distinct
    seeds. Used by the Phase-3 dataset-parameterized runner.
    """
    digest = hashlib.sha256(f"{master_seed}:{dataset_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)
