"""Feature-set configurations, with a written justification for every variable.

Cost tiering
------------
The released dataset ships **no data dictionary and no cost information**. The
tiering below is therefore an explicit, auditable judgement about how each
measurement is normally obtained in a primary-care / community screening
setting, not a fact extracted from the file. Every assignment carries a
one-line rationale and, where the evidence in the file is insufficient to
decide, an explicit ``uncertain`` flag instead of an invented interpretation.

Tiers
-----
``history``     Patient history or a question at the bedside. No consumables.
``exam``        Physical examination or a sphygmomanometer. No consumables.
``urine_dip``   Urine reagent strip. Non-invasive; strips are the cheapest
                laboratory-grade test in this dataset and need no instrument.
``urine_micro`` Urine microscopy. Non-invasive to the patient but requires a
                microscope, a centrifuge and a trained technician.
``blood_lab``   Venepuncture plus a laboratory assay. Invasive and the most
                expensive tier here.
``derived_dx``  Derived from, or defining, the diagnosis itself. Never valid
                as a screening predictor.
``target``      The outcome or an exact copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal[
    "history", "exam", "urine_dip", "urine_micro", "blood_lab", "derived_dx", "target"
]


@dataclass(frozen=True)
class VariableSpec:
    """One column: what we believe it is, how it is obtained, and how sure we are."""

    name: str
    tier: Tier
    description: str
    rationale: str
    uncertain: bool = False
    uncertainty_note: str = ""


# ---------------------------------------------------------------------------
# Variable-level specification. Descriptions are limited to what can be
# verified from the file itself (column name + observed value vocabulary) plus
# standard nephrology abbreviations. Where the abbreviation is ambiguous the
# `uncertain` flag is set and the ambiguity is described rather than resolved.
# ---------------------------------------------------------------------------
VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec(
        "age", "history",
        "Patient age, released as 10 ordered bins from '< 12' to '>= 74' (years).",
        "Age is recorded at registration at no marginal cost and is a "
        "well-established CKD risk factor.",
    ),
    VariableSpec(
        "bp (Diastolic)", "exam",
        "Diastolic blood pressure indicator, released as a binary 0/1 flag.",
        "Blood pressure is measured with a reusable sphygmomanometer; the "
        "marginal cost per patient is effectively zero.",
        uncertain=True,
        uncertainty_note=(
            "The column is named for diastolic pressure but ships as a binary "
            "0/1 flag, so the underlying mmHg value and the threshold that "
            "produced the flag are both unrecoverable. The direction of the "
            "coding (1 = elevated) is inferred from its association with "
            "'bp limit' and 'htn', not from documentation."
        ),
    ),
    VariableSpec(
        "bp limit", "exam",
        "Three-level blood-pressure indicator (0/1/2).",
        "Derived from the same sphygmomanometer reading as 'bp (Diastolic)', "
        "so it adds no marginal cost.",
        uncertain=True,
        uncertainty_note=(
            "Undocumented. Empirically it refines 'bp (Diastolic)': every "
            "patient with bp (Diastolic)=0 has bp limit=0, while bp "
            "(Diastolic)=1 splits across all three levels. It is therefore "
            "most likely a graded severity band, but the mmHg cut-points are "
            "unknown and are NOT reconstructed here."
        ),
    ),
    VariableSpec(
        "htn", "history",
        "Hypertension, binary 0/1.",
        "Elicited as history or read from an existing chart entry; no "
        "consumables required.",
    ),
    VariableSpec(
        "dm", "history",
        "Diabetes mellitus, binary 0/1.",
        "Treated as a history item: a previously established diagnosis is free "
        "to elicit at screening. Verified empirically NOT to be a "
        "deterministic function of the blood-glucose column, so it carries "
        "independent history information.",
        uncertain=True,
        uncertainty_note=(
            "A first-time diabetes diagnosis requires a glucose assay, so in a "
            "population with poor prior healthcare contact this variable would "
            "be less freely available than assumed here."
        ),
    ),
    VariableSpec(
        "cad", "history",
        "Coronary artery disease, binary 0/1.",
        "Recorded as a prior diagnosis during history taking; free to elicit "
        "at the point of screening.",
        uncertain=True,
        uncertainty_note=(
            "Establishing CAD de novo is expensive. Its value as a *screening* "
            "predictor depends on patients already knowing the diagnosis."
        ),
    ),
    VariableSpec(
        "appet", "history",
        "Appetite, binary 0/1.",
        "A single symptom question; no cost.",
        uncertain=True,
        uncertainty_note=(
            "Polarity (whether 1 denotes poor or good appetite) is not "
            "documented in the released file."
        ),
    ),
    VariableSpec(
        "pe", "exam",
        "Pedal oedema, binary 0/1.",
        "Established by inspection and palpation of the ankles; no consumables.",
    ),
    VariableSpec(
        "ane", "blood_lab",
        "Anaemia, binary 0/1.",
        "Conservatively tiered as laboratory-derived. Anaemia is defined by a "
        "haemoglobin concentration, and in a dataset that also ships 'hemo' "
        "the most likely provenance of this flag is the full blood count.",
        uncertain=True,
        uncertainty_note=(
            "IMPORTANT AND CONSEQUENTIAL. Anaemia can also be recorded "
            "clinically (conjunctival or palmar pallor) at no cost. We "
            "verified that 'ane' is NOT a deterministic function of the "
            "'hemo' bins (for example 25 patients with hemo 10-11.3 are ane=0 "
            "while 3 are ane=1), so it is not a pure re-coding of "
            "haemoglobin. Because provenance cannot be established from the "
            "file, it is EXCLUDED from the low-cost configuration. This is "
            "the conservative choice: it can only understate low-cost "
            "performance, never inflate it."
        ),
    ),
    VariableSpec(
        "sg", "urine_dip",
        "Urine specific gravity, 5 ordered bins from '< 1.007' to '>= 1.023'.",
        "Read directly from a urine reagent strip; non-invasive and among the "
        "cheapest measurements available in any clinic.",
    ),
    VariableSpec(
        "al", "urine_dip",
        "Urine albumin, 5 ordered bins ('< 0' to '>= 4'), i.e. a dipstick "
        "protein/albumin grade.",
        "Read from the same urine reagent strip as 'sg'; non-invasive and "
        "cheap. Albuminuria is one of the two axes on which CKD is defined, "
        "which makes it both the most clinically relevant and the most "
        "scrutinised low-cost variable in this study.",
    ),
    VariableSpec(
        "su", "urine_dip",
        "Urine sugar, 6 bins from '< 0' to '>= 4'.",
        "Read from the same urine reagent strip; non-invasive and cheap.",
        uncertain=True,
        uncertainty_note=(
            "The published bin edges for this column are internally "
            "inconsistent and overlapping ('1 - 2' vs '2 - 2'; '3 - 4' vs "
            "'4 - 4' vs '>= 4'), so its ordinal encoding contains ties. "
            "Reported as a data-quality defect; not silently repaired."
        ),
    ),
    VariableSpec(
        "rbc", "urine_micro",
        "Red blood cells in urine, binary 0/1.",
        "Requires urine microscopy: non-invasive to the patient, but needs a "
        "microscope and a trained operator, so it is not free at the point of "
        "community screening.",
        uncertain=True,
        uncertainty_note=(
            "Could in principle refer to a blood-count red-cell abnormality "
            "rather than urinary red cells, but its position among the "
            "urinalysis columns and the presence of a separate 'rbcc' "
            "(red blood cell count) column make urinary microscopy the far "
            "more likely reading."
        ),
    ),
    VariableSpec(
        "pc", "urine_micro",
        "Pus cells in urine, binary 0/1.",
        "Urine microscopy; same infrastructure requirement as 'rbc'.",
    ),
    VariableSpec(
        "pcc", "urine_micro",
        "Pus cell clumps in urine, binary 0/1.",
        "Urine microscopy; same infrastructure requirement as 'rbc'.",
    ),
    VariableSpec(
        "ba", "urine_micro",
        "Bacteria in urine, binary 0/1.",
        "Urine microscopy; same infrastructure requirement as 'rbc'.",
    ),
    VariableSpec(
        "bgr", "blood_lab",
        "Blood glucose (random), 10 bins from '< 112' to '>= 448' mg/dL.",
        "Venous or capillary blood assay.",
    ),
    VariableSpec(
        "bu", "blood_lab",
        "Blood urea, 8 bins from '< 48.1' to '>= 352.9' mg/dL.",
        "Serum chemistry panel; requires venepuncture and an analyser.",
    ),
    VariableSpec(
        "sc", "blood_lab",
        "Serum creatinine, 7 bins from '< 3.65' to '>= 28.85' mg/dL.",
        "Serum chemistry panel; requires venepuncture and an analyser.",
        uncertain=True,
        uncertainty_note=(
            "Serum creatinine is the input to the eGFR equation that defines "
            "'grf' and 'stage'. It is retained in the valid laboratory "
            "configuration because it is a genuine, routinely measured "
            "screening analyte rather than a diagnostic label, but it is the "
            "variable most likely to behave as a near-proxy for the outcome, "
            "and results are interpreted with that in mind."
        ),
    ),
    VariableSpec(
        "sod", "blood_lab",
        "Serum sodium, 9 bins from '< 118' to '>= 158' mEq/L.",
        "Serum electrolyte panel; requires venepuncture.",
    ),
    VariableSpec(
        "pot", "blood_lab",
        "Serum potassium, 4 bins from '< 7.31' to '>= 42.59' mEq/L.",
        "Serum electrolyte panel; requires venepuncture.",
        uncertain=True,
        uncertainty_note=(
            "Near-constant: 197 of 200 patients fall in the single bin "
            "'< 7.31'. The upper bins ('38.18 - 42.59', '>= 42.59') lie far "
            "outside any survivable serum potassium range and are almost "
            "certainly recording errors. Flagged, retained as-is, and carries "
            "essentially no usable information."
        ),
    ),
    VariableSpec(
        "hemo", "blood_lab",
        "Haemoglobin, 10 bins from '< 6.1' to '>= 16.5' g/dL.",
        "Full blood count; requires venepuncture and an analyser.",
    ),
    VariableSpec(
        "pcv", "blood_lab",
        "Packed cell volume (haematocrit), 10 bins from '< 17.9' to "
        "'>= 49.1' percent.",
        "Full blood count; requires venepuncture.",
    ),
    VariableSpec(
        "rbcc", "blood_lab",
        "Red blood cell count, 9 bins from '< 2.69' to '>= 7.41' (millions/uL).",
        "Full blood count; requires venepuncture.",
    ),
    VariableSpec(
        "wbcc", "blood_lab",
        "White blood cell count, 9 bins from '< 4980' to '>= 24020' (cells/uL).",
        "Full blood count; requires venepuncture.",
    ),
    VariableSpec(
        "grf", "derived_dx",
        "Glomerular filtration rate (eGFR), 10 bins plus one invalid entry.",
        "eGFR is computed from serum creatinine and is the quantity that "
        "*defines* CKD stage. Using it to predict CKD is close to using the "
        "diagnostic criterion itself as a predictor.",
    ),
    VariableSpec(
        "stage", "derived_dx",
        "CKD stage s1 to s5.",
        "A staging label assigned after diagnosis. Empirically it is a "
        "deterministic banding of 'grf' in this file, and stages s3 and s5 "
        "are 100 percent CKD, so it encodes the outcome almost directly.",
    ),
    VariableSpec(
        "affected", "target",
        "Binary 0/1 flag.",
        "An exact one-to-one copy of the outcome 'class' in all 200 rows "
        "(ckd maps to 1, notckd maps to 0). It is the target, renamed.",
    ),
    VariableSpec(
        "class", "target",
        "Outcome: 'ckd' or 'notckd'.",
        "The prediction target.",
    ),
)

SPEC_BY_NAME: dict[str, VariableSpec] = {v.name: v for v in VARIABLES}

# ---------------------------------------------------------------------------
# Hard prohibitions, enforced programmatically by ckd.features.encoders.
# ---------------------------------------------------------------------------
TARGET_COLUMN = "class"

#: Columns that must never enter ANY model as a predictor.
ALWAYS_FORBIDDEN: frozenset[str] = frozenset({"class", "affected"})

#: Columns that must never enter a *clinically valid* screening model.
POST_DIAGNOSIS: frozenset[str] = frozenset({"stage", "grf"})

#: Union: forbidden in every configuration except the deliberately invalid one.
FORBIDDEN_IN_VALID: frozenset[str] = ALWAYS_FORBIDDEN | POST_DIAGNOSIS


def _names(*tiers: str) -> list[str]:
    return [v.name for v in VARIABLES if v.tier in tiers]


@dataclass(frozen=True)
class FeatureConfig:
    """A named feature set together with its scientific purpose."""

    name: str
    features: tuple[str, ...]
    purpose: str
    valid_for_clinical_interpretation: bool
    contains_forbidden: tuple[str, ...] = ()


_HISTORY_EXAM = _names("history", "exam")
_URINE_DIP = _names("urine_dip")
_URINE_MICRO = _names("urine_micro")
_BLOOD = _names("blood_lab")

# `ane` is tiered blood_lab (see its uncertainty note) and therefore appears in
# the laboratory / full configurations but never in the low-cost ones.
_ALL_VALID = tuple(_HISTORY_EXAM + _URINE_DIP + _URINE_MICRO + _BLOOD)

FEATURE_CONFIGS: dict[str, FeatureConfig] = {
    "leaky_model": FeatureConfig(
        name="leaky_model",
        features=_ALL_VALID + ("grf", "stage", "affected"),
        purpose=(
            "DELIBERATELY INVALID. Includes an exact target copy ('affected'), "
            "a post-diagnosis staging label ('stage') and the diagnostic "
            "quantity itself ('grf'). Used ONLY to quantify how much target "
            "leakage inflates apparent performance. Its results must never be "
            "read as evidence of clinical usefulness."
        ),
        valid_for_clinical_interpretation=False,
        contains_forbidden=("grf", "stage", "affected"),
    ),
    "full_valid_model": FeatureConfig(
        name="full_valid_model",
        features=_ALL_VALID,
        purpose=(
            "Every predictor that is not the outcome, a copy of the outcome, "
            "or a post-diagnosis derivative. The upper reference for what is "
            "achievable from this file without leakage."
        ),
        valid_for_clinical_interpretation=True,
    ),
    "low_cost_model": FeatureConfig(
        name="low_cost_model",
        features=tuple(_HISTORY_EXAM + _URINE_DIP),
        purpose=(
            "The primary screening question: history, physical examination and "
            "a urine reagent strip only. No venepuncture, no microscope, no "
            "laboratory analyser."
        ),
        valid_for_clinical_interpretation=True,
    ),
    "laboratory_model": FeatureConfig(
        name="laboratory_model",
        features=tuple(_URINE_MICRO + _BLOOD),
        purpose=(
            "Laboratory measurements only (blood chemistry, full blood count "
            "and urine microscopy), excluding all target proxies and "
            "post-diagnosis variables. The comparator against which the "
            "low-cost model must be judged."
        ),
        valid_for_clinical_interpretation=True,
    ),
    "clinical_only_model": FeatureConfig(
        name="clinical_only_model",
        features=tuple(_HISTORY_EXAM),
        purpose=(
            "SENSITIVITY ANALYSIS. History and examination alone, with no "
            "urine testing at all. Isolates how much of the low-cost model's "
            "performance comes from the dipstick."
        ),
        valid_for_clinical_interpretation=True,
    ),
    "low_cost_plus_urine_micro_model": FeatureConfig(
        name="low_cost_plus_urine_micro_model",
        features=tuple(_HISTORY_EXAM + _URINE_DIP + _URINE_MICRO),
        purpose=(
            "SENSITIVITY ANALYSIS. Low-cost set plus urine microscopy. Tests "
            "whether the judgement call that placed microscopy outside the "
            "low-cost tier changes the study's conclusions."
        ),
        valid_for_clinical_interpretation=True,
    ),
}


def get_config(name: str) -> FeatureConfig:
    if name not in FEATURE_CONFIGS:
        raise KeyError(
            f"Unknown feature configuration {name!r}. Known: {sorted(FEATURE_CONFIGS)}"
        )
    return FEATURE_CONFIGS[name]


def valid_config_names() -> list[str]:
    """Names of configurations that are legitimate for clinical interpretation."""
    return [n for n, c in FEATURE_CONFIGS.items() if c.valid_for_clinical_interpretation]


def uncertain_variables() -> list[VariableSpec]:
    """Variables whose interpretation could not be established from the file."""
    return [v for v in VARIABLES if v.uncertain]
