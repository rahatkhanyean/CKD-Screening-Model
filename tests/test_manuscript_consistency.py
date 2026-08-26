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
import subprocess
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ckd.config import project_root
from ckd.features.registry import prohibited_variables

ROOT = project_root()
TEX = ROOT / "paper" / "ieee_paper.tex"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "paper" / "figures"

#: SHA-256 of every input the manuscript's numbers depend on.
EXPECTED_HASHES = {
    "data/raw/ckd-dataset-v2.csv":
        "f24075f420b0f271bfddf3844a40061dbe9e2cb1c2336daa64463122344ea84a",
    "data/external/uci2015/raw/data.csv":
        "0eeea8d17f5ad8792d854999d6ddf1602ec4e116616f6ccdfdaef0ba8109e694",
}


def table_or_skip(name: str):
    """Load a generated table, skipping if the stage has not been run."""
    path = TABLES / name
    if not path.is_file():
        pytest.skip(f"{name} not generated; run the pipeline")
    return pd.read_csv(path)


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

    def test_copula_null_matches_its_source_exactly(self):
        """The null statistics must equal what the CSV says, not merely
        appear somewhere in the file.

        The earlier version of this test asserted that the formatted mean
        was a substring of the manuscript. It passed while the manuscript
        quoted a stale 100-draw value, because an unrelated number in
        Section VI happened to render as the same three digits. Substring
        containment is not verification. These values are now generated
        into paper/generated/numbers.tex, and this test compares that
        fragment against the CSV it claims to come from.
        """
        nulls = table("table_46_provenance_nulls.csv")
        column = "copula_preserving_correlations"
        if column not in nulls.columns:
            pytest.skip("copula null not generated")

        fragment = (ROOT / "paper" / "generated" / "numbers.tex")
        assert fragment.is_file(), (
            "numbers.tex missing; run scripts/24_make_latex_tables.py"
        )
        text = fragment.read_text(encoding="utf-8")
        defined = dict(re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))

        series = nulls[column]
        expected = {
            "NullDraws": str(len(nulls)),
            "NullCopulaMean": f"{series.mean():.3f}",
            "NullCopulaSD": f"{series.std(ddof=1):.3f}",
            "NullCopulaMax": f"{series.max():.3f}",
            "NullCopulaSDDistance":
                f"{(1.0 - series.mean()) / series.std(ddof=1):.0f}",
        }
        for name, value in expected.items():
            assert defined.get(name) == value, (
                f"{name} is {defined.get(name)!r} but the CSV gives "
                f"{value!r}; regenerate with scripts/24_make_latex_tables.py"
            )

    def test_test_count_is_generated_not_typed(self, tex):
        """The Reproducibility statement's test count must be a macro.

        It said 416 while the suite held 457. A count typed into prose goes
        stale on the next commit that adds a test, so reproduce.py now writes
        the pytest inventory as a result file and the manuscript quotes it.
        """
        assert "\\TestCount" in tex, (
            "manuscript must quote the test count through the generated macro"
        )
        inventory = TABLES / "table_52_test_inventory.csv"
        fragment = ROOT / "paper" / "generated" / "numbers.tex"
        if not inventory.is_file():
            pytest.skip("test inventory not written; run reproduce.py")
        expected = str(int(pd.read_csv(inventory).iloc[0]["tests"]))
        defined = dict(re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
            fragment.read_text(encoding="utf-8")))
        assert defined.get("TestCount") == expected, (
            f"TestCount is {defined.get('TestCount')!r} but the inventory "
            f"records {expected!r}"
        )

    def test_manuscript_uses_the_generated_macros(self, tex):
        """A literal would silently detach from the CSV again."""
        assert "\\NullCopulaMean" in tex, (
            "manuscript must quote the copula null through the generated "
            "macro, not as a typed number"
        )

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
    def test_no_internal_figure_labels_inside_the_graphics(self, tex):
        """The same check, applied to what is actually *drawn*.

        A figure can carry an internal label in its own embedded title even
        when the LaTeX is clean, because the graphic is produced by a
        different script. Visual inspection of the built PDF found exactly
        that; this test is the regression guard.
        """
        if shutil.which("pdftotext") is None:
            pytest.skip("pdftotext unavailable")
        included = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
        assert included, "no figures found in the manuscript"
        offenders = []
        for name in sorted(included):
            source = FIGURES / name
            if not source.is_file():
                continue
            result = subprocess.run(
                ["pdftotext", "-q", str(source), "-"],
                capture_output=True, text=True,
            )
            if re.search(r"\bFig(?:ure)?\.?\s*[RE]\d+", result.stdout):
                offenders.append(name)
        assert not offenders, (
            f"internal figure labels drawn inside {offenders}; the manuscript "
            "numbers figures itself, so an embedded 'Figure R12' contradicts it"
        )


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


class TestRetractedClaimsStayRetracted:
    """Phrases removed during revision because they outran the evidence.

    Each was present in an earlier draft. A grep is the cheapest way to stop
    a later edit from quietly reinstating one -- which is exactly how the
    Conclusion came to contradict Sections V-D and V-E, undetected, until
    the final page-by-page inspection of the built PDF.
    """

    RETRACTED = {
        "record-for-record":
            "claims exact raw-value identity, which discretisation makes "
            "untestable; say 'discretised subset or re-release'",
        "is the larger effect":
            "asserts a decomposition the redundant feature sets cannot "
            "support; say 'appears principal but cannot be cleanly separated'",
        "no predictive ability":
            "case-mix separability is not an absence of predictive ability",
        "none of which is predictive ability":
            "same overclaim, earlier wording",
        "exactly identical patients":
            "same overclaim as record-for-record",
        "25 independent outer test sets":
            "the 25 outer folds are five repartitions of the same 200 patients",
        "definitively the largest":
            "no valid quantitative decomposition supports a ranking",
        "none is predictive ability":
            "found in the Discussion after the same claim was removed "
            "elsewhere; discrimination here is real, its transportability is "
            "what is unknown",
        "the largest of which is rarely examined":
            "ranks the three mechanisms, which the redundant feature sets "
            "cannot support",
    }

    def test_no_retracted_claim_reappears(self, tex):
        lowered = tex.lower()
        found = {phrase: why for phrase, why in self.RETRACTED.items()
                 if phrase in lowered}
        assert not found, (
            "retracted claims have reappeared in the manuscript:\n"
            + "\n".join(f"  {phrase!r}: {why}" for phrase, why in found.items())
        )

    def test_supplement_too(self):
        text = (ROOT / "paper" / "supplement.tex").read_text(
            encoding="utf-8").lower()
        found = [p for p in self.RETRACTED if p in text]
        assert not found, f"retracted claims in supplement: {found}"

    def test_effective_sample_language(self, tex):
        """The effective sample must be stated, not implied.

        Defect D1: an earlier draft described the 25 outer test sets as
        independent. They are five repartitions of the same 200 patients.
        The correction is only durable if the manuscript states the unit
        positively as well as avoiding the retracted phrase.
        """
        lowered = tex.lower()
        assert "200 unique" in lowered, (
            "manuscript must state the effective sample as 200 unique patients"
        )
        assert "five out-of-fold evaluations per patient" in lowered, (
            "manuscript must say how many evaluations each patient receives"
        )
        # Wherever the artefact size appears it must be qualified.
        for match in re.finditer(r"108,?000", tex):
            window = tex[max(0, match.start() - 400): match.end() + 400].lower()
            assert "200" in window and (
                "repeated" in window or "evaluation" in window), (
                "'108,000' appears without being identified as repeated "
                "evaluations of 200 patients"
            )


class TestFiguresAreTypeset:
    """Figures are typeset output, not console or markdown output.

    Three of the six included figures shipped with literal markdown backticks
    around variable names, because the same helper text served the internal
    markdown report and the paper. Nothing caught it until the built PDF was
    read page by page.
    """

    MARKUP = {
        "`": "markdown code backticks",
        "**": "markdown bold",
        "\\texttt": "LaTeX markup in a raster/vector figure",
    }

    def test_no_markup_characters_in_figures(self, tex):
        if shutil.which("pdftotext") is None:
            pytest.skip("pdftotext unavailable")
        included = set(re.findall(
            r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
        offenders = {}
        for name in sorted(included):
            source = FIGURES / name
            if not source.is_file():
                continue
            text = subprocess.run(
                ["pdftotext", "-q", str(source), "-"],
                capture_output=True, text=True).stdout
            hits = [why for token, why in self.MARKUP.items() if token in text]
            if hits:
                offenders[name] = hits
        assert not offenders, f"markup leaked into typeset figures: {offenders}"


class TestCorrectedTaxonomyReachesTheManuscript:
    """The KDIGO correction must hold in code, results and prose together.

    The reviewer found that the registry classified blood urea as a
    diagnostic-criterion input and urine albumin as an ordinary predictor,
    which inverts the KDIGO definition: albuminuria is the first-listed
    marker of kidney damage, and blood urea is not a criterion at all. Every
    "incorporation removed" statement was therefore false, because the
    albuminuria limb stayed in the model. These tests stop that returning.
    """

    def test_restricted_sets_actually_remove_both_kdigo_limbs(self):
        from ckd.features.registry import incorporation_variables

        table = table_or_skip("table_49_restricted_featuresets.csv")
        rows = table[table["featureset"]
                     == "minus all incorporation-risk variables"]
        assert len(rows), "the broad incorporation configuration was not run"
        kept = set(str(rows.iloc[0]["features"]).split(";"))
        leftover = kept & incorporation_variables()
        assert not leftover, (
            f"configuration claims to remove incorporation risk but retains "
            f"{sorted(leftover)}"
        )
        # The strict variant must still remove both criterion inputs.
        strict = table[table["featureset"]
                       == "minus criterion inputs only (strict)"]
        if len(strict):
            kept = set(str(strict.iloc[0]["features"]).split(";"))
            assert not (kept & incorporation_variables(strict=True))

    def test_manuscript_does_not_call_blood_urea_a_criterion(self, tex):
        """The retracted clinical claim, in the exact form it took."""
        lowered = tex.lower()
        for phrase in (
            "criterion --- serum creatinine and blood urea",
            "diagnostic criterion --- serum creatinine and blood urea",
        ):
            assert phrase not in lowered, (
                "manuscript still names blood urea as a diagnostic-criterion "
                "input; KDIGO does not list it"
            )

    def test_manuscript_states_both_limbs(self, tex):
        assert "albuminuria" in tex.lower(), (
            "the albuminuria limb of the KDIGO definition must be discussed"
        )


class TestMatchingNullsAreLabelledAsImplemented:
    def test_only_implemented_nulls_are_reported(self):
        summary = table_or_skip("table_46_provenance_null_summary.csv")
        names = set(summary["null"])
        assert names == {
            "independent_column_permutation",
            "gaussian_copula_marginals_and_correlations",
        }, f"unexpected null set: {sorted(names)}"

    def test_row_permutation_is_not_offered_as_a_null(self):
        """Bipartite matching is invariant to candidate row order, so a
        whole-row permutation cannot be an inferential null. It may be
        described as a property of the estimator; it may not appear in the
        table of nulls."""
        summary = table_or_skip("table_46_provenance_null_summary.csv")
        assert not any("row" in n and "permutation" in n
                       for n in summary["null"])

    def test_copula_null_preserves_dependence(self):
        """A null that flattened the correlation structure would be weaker
        than the manuscript claims it is."""
        diag = table_or_skip("table_53_copula_null_diagnostics.csv")
        pairs = diag[diag["quantity"] == "spearman_pair"]
        assert len(pairs) > 0
        gap = (pairs["real"] - pairs["null_mean"]).abs()
        assert gap.mean() < 0.10, (
            f"copula null does not preserve dependence: mean |real - null| "
            f"Spearman = {gap.mean():.3f}"
        )
        marg = diag[diag["quantity"] == "marginal"]
        assert int(marg["synthetic_values_off_observed_support"].sum()) == 0, (
            "synthetic values fall outside the observed support, which would "
            "depress the null for a mechanical reason"
        )


class TestLiteratureClaimsRestToVerifiedStudies:
    def test_no_prevalence_language_in_the_manuscript(self, tex):
        lowered = tex.lower()
        for word in ("frequently report", "routinely report",
                     "most studies", "commonly report"):
            assert word not in lowered, (
                f"prevalence claim {word!r} is not supported by a targeted, "
                "non-systematic sample"
            )

    def test_cross_release_claim_matches_verified_count(self, tex):
        summary = table_or_skip("table_26_prior_work_summary.csv")
        value = dict(zip(summary["quantity"], summary["value"]))
        verified = int(value["n_cross_dataset_VERIFIED_from_source"])
        # The manuscript must not assert the design of more studies than it
        # verified. It said "three" while only one is verifiable.
        assert verified >= 1
        assert "three validate\nacross the two releases" not in tex, (
            "manuscript asserts a cross-release design for studies whose "
            "text was never obtained"
        )
