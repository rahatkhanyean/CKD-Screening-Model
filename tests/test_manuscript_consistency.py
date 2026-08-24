"""Automated manuscript-consistency checks.

These tests treat the LaTeX source as an artefact that must agree with the
generated results, in the same way the generated report already does. They
fail the build when a manuscript claim drifts from its source table, when a
placeholder survives, or when a guarantee the paper asserts is not actually
enforced.

Checks implemented, per the revision brief:

* numbers in LaTeX differing from generated outputs
* missing figures or tables referenced by the manuscript
* non-independent datasets appearing in external-validation tables
* prohibited variables appearing in valid configurations
* placeholder author information
* unresolved TODO markers and internal labels
* broken citations and undefined LaTeX references
* changed dataset hashes
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest

from ckd.config import project_root
from ckd.features.registry import prohibited_variables

ROOT = project_root()
TEX = ROOT / "paper" / "ieee_paper.tex"
TABLES = ROOT / "reports" / "tables"

#: SHA-256 of every input the manuscript's numbers depend on.
EXPECTED_HASHES = {
    "data/raw/ckd-dataset-v2.csv":
        "f24075f420b0f271bfddf3844a40061dbe9e2cb1c2336daa64463122344ea84a",
    "data/external/uci2015/raw/data.csv":
        "0eeea8d17f5ad8792d854999d6ddf1602ec4e116616f6ccdfdaef0ba8109e694",
}


def _expand_inputs(source: str, base: Path, depth: int = 0) -> str:
    """Inline ``\\input`` directives, as the LaTeX build does.

    Labels, references and generated numbers may live in fragments emitted by
    ``scripts/24_make_latex_tables.py``. A consistency check that reads only
    the top-level file would report those as dangling, so the fixture
    resolves the same includes the compiler resolves.
    """
    if depth > 5:
        return source

    def replace(match: re.Match) -> str:
        name = match.group(1).strip()
        path = base / (name if name.endswith(".tex") else f"{name}.tex")
        if not path.is_file():
            return match.group(0)
        return _expand_inputs(path.read_text(encoding="utf-8"), base, depth + 1)

    return re.sub(r"\\input\{([^}]+)\}", replace, source)


@pytest.fixture(scope="module")
def tex() -> str:
    if not TEX.is_file():
        pytest.skip("manuscript source not present")
    return _expand_inputs(TEX.read_text(encoding="utf-8"), TEX.parent)


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


class TestDatasetIntegrity:
    @pytest.mark.parametrize("relative,expected", sorted(EXPECTED_HASHES.items()))
    def test_input_hashes_unchanged(self, relative, expected):
        path = ROOT / relative
        if not path.is_file():
            pytest.skip(f"{relative} not present")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{relative} changed: manuscript numbers were computed from "
            f"{expected[:16]}…, found {actual[:16]}…"
        )


class TestManuscriptNumbersMatchGeneratedResults:
    """Each headline value is re-derived from its table and must appear."""

    def test_external_transportability_numbers(self, tex):
        ext = table("table_42_external_validation.csv")
        frozen = ext[ext["arm"] == "external_frozen_transfer"]
        internal = ext[ext["arm"] == "internal_nested_cv"]
        for value in (f"{frozen['roc_auc'].min():.3f}",
                      f"{frozen['roc_auc'].max():.3f}",
                      f"{internal['roc_auc'].max():.3f}",
                      str(int(frozen["n_positive"].iloc[0]))):
            assert value in tex, f"{value} missing from manuscript"

    def test_provenance_numbers(self, tex):
        prov = table("table_24_provenance.csv")
        uci = prov[prov["dataset_id"] == "uci2015"].iloc[0]
        assert f"{uci['containment_match_fraction']:.3f}" in tex

    def test_provenance_sensitivity_numbers(self, tex):
        sens = table("table_44_provenance_sensitivity.csv")
        primary = sens[sens["analysis"] == "primary"].iloc[0]
        assert str(int(primary["n_unique_pins"])) in tex
        control = sens[sens["analysis"] == "negative_control"]
        if len(control):
            assert f"{control['match_fraction'].iloc[0]:.3f}" in tex

    def test_copula_null_reported(self, tex):
        nulls = table("table_46_provenance_nulls.csv")
        column = "copula_preserving_correlations"
        if column not in nulls.columns:
            pytest.skip("copula null not generated")
        assert f"{nulls[column].mean():.3f}" in tex

    def test_subgroup_intervals_reported(self, tex):
        subgroup = table("table_48_subgroup_metrics_ci.csv")
        early = subgroup[(subgroup["subgroup"] == "early_ckd_s1_s2")
                         & (subgroup["config"] == "low_cost_model")].iloc[0]
        assert str(int(early["n_cases"])) in tex
        assert f"{early['sensitivity_ci_low']:.3f}" in tex, (
            "early-stage sensitivity interval must appear in the manuscript"
        )

    def test_binning_and_imputation_numbers(self, tex):
        cost = table("table_31_binning_cost.csv").set_index("variable")
        assert f"{cost.loc['sc', 'binning_cost']:.3f}" in tex
        imp = table("table_32_imputation_audit.csv")
        assert str(int(imp["n_unobserved_in_source"].sum())) in tex


class TestNoPlaceholdersOrInternalLabels:
    def test_no_todo_markers(self, tex):
        """No unresolved markers, with one deliberate exception.

        The author block is the single placeholder the authors must supply
        and an agent must not invent, so ``TODO(author)`` is permitted --- and
        required, by ``test_author_placeholder_is_flagged_not_silent`` --- for
        as long as that block is unfilled. Every other marker is a defect.
        """
        without_author = tex.replace("TODO(author)", "")
        for marker in ("TODO", "FIXME", "XXX", "todo_hand_check"):
            assert marker not in without_author, (
                f"unresolved marker {marker!r} in manuscript"
            )

    def test_no_internal_figure_labels(self, tex):
        """Internal names such as 'Figure R12' must not reach the reader."""
        assert not re.search(r"\bFig(?:ure)?\.?~?\s*R\d+", tex)

    def test_author_placeholder_is_flagged_not_silent(self, tex):
        """A placeholder author block is permitted during revision but must
        carry the marker that makes it impossible to submit by accident."""
        if "Author Name" in tex or "email@institution" in tex:
            assert "TODO(author)" in tex or "placeholder" in tex.lower(), (
                "author block is a placeholder but is not marked as one"
            )


class TestGuaranteesTheManuscriptAsserts:
    def test_no_prohibited_variable_in_valid_configurations(self):
        from ckd.features.configs import FEATURE_CONFIGS, valid_config_names

        for name in valid_config_names():
            overlap = set(FEATURE_CONFIGS[name].features) & prohibited_variables()
            assert not overlap, f"{name} contains prohibited {sorted(overlap)}"

    def test_no_non_independent_dataset_in_external_tables(self):
        verdicts = table("table_24_provenance.csv").set_index(
            "dataset_id")["verdict"].to_dict()
        blocked = {d for d, v in verdicts.items() if v != "INDEPENDENT"}
        for path in TABLES.glob("*external*.csv"):
            content = pd.read_csv(path).astype(str)
            for dataset_id in blocked:
                hit = content.apply(
                    lambda col: col.str.contains(dataset_id, regex=False)
                ).any().any()
                assert not hit, f"{path.name} references blocked {dataset_id}"


class TestFiguresAndTablesExist:
    def test_every_included_graphic_is_present(self, tex):
        for name in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
            candidate = ROOT / "paper" / "figures" / name
            assert candidate.is_file(), f"missing figure file: {name}"

    def test_every_table_source_comment_points_at_a_real_file(self, tex):
        for name in re.findall(r"% Source: reports/tables/([\w./]+\.csv)", tex):
            assert (TABLES / Path(name).name).is_file(), f"missing table: {name}"


class TestLatexIntegrity:
    def test_no_dangling_references(self, tex):
        labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
        refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
        assert not (refs - labels), f"dangling refs: {sorted(refs - labels)}"

    def test_no_uncited_bibliography_entries(self, tex):
        defined = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
        body = tex.split("thebibliography")[0]
        cited = {k.strip() for m in re.findall(r"\\cite\{([^}]+)\}", body)
                 for k in m.split(",")}
        assert not (defined - cited), f"uncited: {sorted(defined - cited)}"
        assert not (cited - defined), f"missing keys: {sorted(cited - defined)}"

    def test_no_hardcoded_section_cross_references(self, tex):
        hardcoded = re.findall(r"Section~[IVX]+-[A-Z]\b", tex)
        assert not hardcoded, f"hardcoded section refs: {hardcoded}"
