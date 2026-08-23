"""ETL from the MIMIC-IV Clinical Database Demo to this study's schema.

MIMIC is the only registered external source that carries the *same
measurements* as the analysed benchmark: a full blood chemistry panel plus a
urine dipstick. That makes it usable for genuine external validation rather
than the task-shifted comparison the other sources permit.

Three properties of the extraction are load-bearing and are therefore
implemented explicitly rather than left to a caller:

**The label is ICD-coded, not adjudicated.** A subject is CKD-positive if any
admission carries ICD-10 ``N18*`` or ICD-9 ``585*``. Coding is administrative;
it is neither a chart review nor a KDIGO determination, and no chronicity
criterion is verified. The derived column is named ``ckd_label`` because the
external registry declares that name prohibited, so the leakage guard covers
it before this module existed.

**Features must not postdate the diagnosis.** MIMIC codes diagnoses per
admission, so the finest available resolution is the admission in which CKD
was first coded. Laboratory values are taken from strictly before that
admission where one exists, and the number of subjects for whom no such value
exists is reported rather than silently imputed. For CKD-negative subjects
every admission is eligible.

**Urine dipstick results are semi-quantitative text.** ``NEG``/``TRACE``/
``30``/``100``/``300`` are mapped to the ordinal scale the analysed benchmark
uses (0-5), which is the same information the dipstick carries. The mapping is
a pure function of one cell's text, mirroring the interval encoding of the
main analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: MIMIC ``itemid`` for each variable of this study's schema. Blood analytes
#: first, then the urine dipstick tier. Chosen as the item with the widest
#: subject coverage where a label maps to several items.
LAB_ITEMS: dict[str, int] = {
    "sc": 50912,     # Creatinine, Blood
    "hemo": 51222,   # Hemoglobin, Blood
    "sod": 50983,    # Sodium, Blood
    "pot": 50971,    # Potassium, Blood
    "bu": 51006,     # Urea Nitrogen, Blood
    "bgr": 50931,    # Glucose, Blood
    "wbcc": 51301,   # White Blood Cells, Blood
    "rbcc": 51279,   # Red Blood Cells, Blood
    "pcv": 51221,    # Hematocrit, Blood
    "sg": 51498,     # Specific Gravity, Urine
}

#: Urine dipstick items whose results are semi-quantitative text.
DIPSTICK_ITEMS: dict[str, int] = {
    "al": 51492,     # Protein, Urine  -> albumin/protein tier
    "su": 51478,     # Glucose, Urine
    "rbc": 51493,    # RBC, Urine
}

#: Unit reconciliation. MIMIC reports WBC in K/uL and RBC in m/uL; the
#: analysed benchmark uses cells/uL and millions/uL respectively.
UNIT_SCALE: dict[str, float] = {"wbcc": 1000.0}

#: Semi-quantitative dipstick vocabulary -> the benchmark's ordinal scale.
DIPSTICK_SCALE: dict[str, float] = {
    "neg": 0.0, "negative": 0.0, "none": 0.0,
    "tr": 1.0, "trace": 1.0,
    "1+": 1.0, "2+": 2.0, "3+": 3.0, "4+": 4.0,
    "30": 2.0, "100": 3.0, "300": 4.0, "500": 5.0, ">300": 5.0, ">600": 5.0,
    "small": 1.0, "moderate": 3.0, "large": 4.0,
    "0-2": 1.0, "3-5": 2.0, "6-10": 3.0, "11-20": 4.0, ">50": 5.0, "21-50": 4.0,
}

CKD_ICD10_PREFIX = "N18"
CKD_ICD9_PREFIX = "585"

#: ICD-10 code -> the benchmark's stage vocabulary, for case-mix comparison.
STAGE_FROM_ICD10: dict[str, str] = {
    "N181": "s1", "N182": "s2", "N183": "s3",
    "N184": "s4", "N185": "s5", "N186": "s5",
}


@dataclass(frozen=True)
class MimicCohort:
    """The harmonised cohort plus everything worth reporting about it."""

    frame: pd.DataFrame           # one row per subject, benchmark variable names
    target: np.ndarray            # 1 = CKD coded, 0 = not
    stage: pd.Series              # s1..s5 or NA (unspecified / not CKD)
    notes: tuple[str, ...] = field(default=())


def _read(root: Path, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(root / name, **kwargs)


def map_dipstick(values: pd.Series) -> pd.Series:
    """Map semi-quantitative dipstick text to the benchmark's ordinal scale.

    A pure function of one cell's text, like the interval encoding of the
    main analysis. Unrecognised tokens become missing rather than being
    guessed at.
    """
    cleaned = values.astype(str).str.strip().str.lower()
    return cleaned.map(DIPSTICK_SCALE)


def build_cohort(root: Path) -> MimicCohort:
    """Extract one row per subject in the benchmark's schema."""
    notes: list[str] = []

    patients = _read(root, "hosp/patients.csv.gz")
    admissions = _read(root, "hosp/admissions.csv.gz",
                       usecols=["subject_id", "hadm_id", "admittime"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    diagnoses = _read(root, "hosp/diagnoses_icd.csv.gz")

    # ---- label -----------------------------------------------------------
    is_ckd_code = (
        (diagnoses["icd_version"] == 10)
        & diagnoses["icd_code"].str.startswith(CKD_ICD10_PREFIX, na=False)
    ) | (
        (diagnoses["icd_version"] == 9)
        & diagnoses["icd_code"].str.startswith(CKD_ICD9_PREFIX, na=False)
    )
    ckd_rows = diagnoses[is_ckd_code]
    ckd_subjects = set(ckd_rows["subject_id"])

    subjects = pd.Index(sorted(patients["subject_id"].unique()), name="subject_id")
    target = np.array([1 if s in ckd_subjects else 0 for s in subjects], dtype=int)
    notes.append(
        f"label from ICD-10 {CKD_ICD10_PREFIX}* / ICD-9 {CKD_ICD9_PREFIX}*: "
        f"{int(target.sum())} of {len(subjects)} subjects coded CKD"
    )

    # ---- stage, for case-mix comparison ----------------------------------
    stage_map: dict[int, str] = {}
    for subject, group in ckd_rows[ckd_rows["icd_version"] == 10].groupby("subject_id"):
        stages = [STAGE_FROM_ICD10.get(code) for code in group["icd_code"]]
        stages = [s for s in stages if s]
        if stages:
            stage_map[subject] = max(stages)  # most advanced coded stage
    stage = pd.Series([stage_map.get(s) for s in subjects], index=subjects, dtype="object")
    n_staged = int(stage.notna().sum())
    notes.append(
        f"stage recoverable for {n_staged} of {int(target.sum())} CKD subjects "
        f"(remainder coded N18.9 'unspecified')"
    )

    # ---- the temporal guard ----------------------------------------------
    # For each CKD subject, the admission time at which CKD was first coded.
    first_ckd_admit: dict[int, pd.Timestamp] = {}
    ckd_admits = ckd_rows.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    for subject, group in ckd_admits.groupby("subject_id"):
        times = group["admittime"].dropna()
        if len(times):
            first_ckd_admit[subject] = times.min()

    # ---- laboratory values -----------------------------------------------
    wanted = {**LAB_ITEMS, **DIPSTICK_ITEMS}
    labs = _read(
        root, "hosp/labevents.csv.gz",
        usecols=["subject_id", "itemid", "charttime", "valuenum", "value"],
    )
    labs = labs[labs["itemid"].isin(set(wanted.values()))].copy()
    labs["charttime"] = pd.to_datetime(labs["charttime"])

    # Drop values that do not strictly precede the first CKD-coded admission.
    before = labs["subject_id"].map(first_ckd_admit)
    postdates = before.notna() & (labs["charttime"] >= before)
    n_dropped = int(postdates.sum())
    labs = labs[~postdates]
    notes.append(
        f"temporal guard: dropped {n_dropped} laboratory values not strictly "
        f"preceding the admission at which CKD was first coded"
    )

    columns: dict[str, pd.Series] = {}
    for variable, itemid in LAB_ITEMS.items():
        subset = labs[labs["itemid"] == itemid].dropna(subset=["valuenum"])
        earliest = (
            subset.sort_values("charttime").groupby("subject_id")["valuenum"].first()
        )
        series = earliest.reindex(subjects).astype(float)
        if variable in UNIT_SCALE:
            series = series * UNIT_SCALE[variable]
        columns[variable] = series

    for variable, itemid in DIPSTICK_ITEMS.items():
        subset = labs[labs["itemid"] == itemid].copy()
        subset["ordinal"] = map_dipstick(subset["value"])
        subset = subset.dropna(subset=["ordinal"])
        earliest = (
            subset.sort_values("charttime").groupby("subject_id")["ordinal"].first()
        )
        columns[variable] = earliest.reindex(subjects).astype(float)

    frame = pd.DataFrame(columns, index=subjects)

    # A CKD subject with no pre-diagnosis measurement contributes nothing but
    # a label; report how many rather than imputing them into existence.
    empty = frame.isna().all(axis=1)
    notes.append(
        f"{int(empty.sum())} subject(s) have no eligible measurement after the "
        f"temporal guard and are reported, not imputed"
    )

    frame.insert(0, "subject_id", subjects)
    return MimicCohort(frame=frame, target=target, stage=stage, notes=tuple(notes))
