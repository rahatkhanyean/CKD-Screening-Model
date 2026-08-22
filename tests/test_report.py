"""Generated-report consistency tests (Phase 1a).

The research report is generated from tables by stage 6, so its numbers
*should* be incapable of drifting. These tests make that a guarantee
instead of an intention:

* no unresolved template artifacts (``n/a`` in the abstract, ``nan``,
  ``{...}`` placeholders) may survive into the rendered markdown;
* the headline numbers quoted in the abstract and conclusion must equal the
  values in their source tables, re-derived here independently;
* configuration attribution must be honest: a Full-valid number may never
  be captioned as "laboratory".

They run against the report as committed on disk — regenerating stage 6 and
re-running these tests is the check that edits to the generator kept it
consistent.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ckd.config import project_root

REPORT = project_root() / "reports" / "research_report.md"
TABLES = project_root() / "reports" / "tables"


@pytest.fixture(scope="module")
def report_text() -> str:
    assert REPORT.is_file(), "reports/research_report.md missing - run stage 6"
    return REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def abstract(report_text) -> str:
    match = re.search(r"## Abstract\n(.*?)\n## 1\.", report_text, re.DOTALL)
    assert match, "abstract section not found"
    return match.group(1)


@pytest.fixture(scope="module")
def conclusion(report_text) -> str:
    match = re.search(r"## 10\. Conclusion\n(.*?)\n## References", report_text, re.DOTALL)
    assert match, "conclusion section not found"
    return match.group(1)


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


class TestNoTemplateArtifacts:
    def test_no_unrendered_placeholders(self, report_text):
        # Literal format-string leftovers like {fmt(...)} or {best['roc_auc']}.
        assert not re.search(r"\{[a-z_]+\(", report_text)
        assert not re.search(r"\{[a-z_]+\[", report_text)

    def test_no_nan_tokens_in_prose(self, report_text):
        # 'nan' as a standalone rendered value is always an injection bug.
        assert not re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", report_text)

    def test_abstract_never_says_slope_na(self, abstract):
        """The original defect: an unidentified calibration slope rendered
        as 'slope n/a'. The abstract must either give a finite slope or say
        explicitly that the slope is not identified."""
        assert "slope n/a" not in abstract
        selected = table("table_16_selected_models.csv")
        best = selected[selected["selection"] == "best_valid_overall"].iloc[0]
        if not np.isfinite(best["calibration_slope"]):
            assert "not identified" in abstract

    def test_na_appears_only_in_allowed_contexts(self, report_text):
        """'n/a' may appear only where a quantity is legitimately undefined
        AND the same line says why.

        Allowed reasons, each corresponding to a real situation in this
        study: a calibration slope that is not identified under complete
        separation, and a ratio against a zero-width (degenerate)
        reference interval. The list is deliberately short - the point is
        that an unexplained 'n/a' is a template bug, so a new reason must
        be added here consciously rather than slipping through.
        """
        allowed = ("undefined", "not identified", "degenerate")
        for line in report_text.splitlines():
            if re.search(r"(?<![A-Za-z])n/a(?![A-Za-z])", line):
                assert any(reason in line for reason in allowed), (
                    f"unexplained 'n/a' in report line: {line[:120]}"
                )


class TestHeadlineNumbersMatchTables:
    """Every number quoted below is re-derived from its source table and
    searched for verbatim in the relevant report section."""

    def test_abstract_low_cost_and_laboratory_auc(self, abstract):
        audit = table("table_13_leakage_audit.csv")
        low = audit[audit["config"] == "low_cost_model"].iloc[0]
        lab = audit[audit["config"] == "laboratory_model"].iloc[0]
        assert f"{low['roc_auc']:.3f}" in abstract
        assert f"{lab['roc_auc']:.3f}" in abstract

    def test_abstract_low_cost_sensitivity_and_npv(self, abstract):
        audit = table("table_13_leakage_audit.csv")
        low = audit[audit["config"] == "low_cost_model"].iloc[0]
        assert f"{low['sensitivity']:.3f}" in abstract
        assert f"{low['npv']:.3f}" in abstract

    def test_abstract_quotes_identified_low_cost_slope_when_best_is_undefined(self, abstract):
        selected = table("table_16_selected_models.csv")
        best = selected[selected["selection"] == "best_valid_overall"].iloc[0]
        best_lc = selected[selected["selection"] == "best_low_cost"].iloc[0]
        if not np.isfinite(best["calibration_slope"]):
            assert f"{best_lc['calibration_slope']:.2f}" in abstract

    def test_conclusion_selected_model_numbers(self, conclusion):
        selected = table("table_16_selected_models.csv")
        best = selected[selected["selection"] == "best_valid_overall"].iloc[0]
        best_lc = selected[selected["selection"] == "best_low_cost"].iloc[0]
        assert f"{best_lc['roc_auc']:.3f}" in conclusion
        assert f"{best['roc_auc']:.3f}" in conclusion
        assert f"{best_lc['calibration_slope']:.2f}" in conclusion

    def test_kendalls_w_consistent(self, report_text):
        stability = table("table_20_importance_stability.csv")
        perm = stability[stability["method"] == "permutation"]
        for target in ("best_valid", "low_cost"):
            subset = perm[perm["target"] == target]
            if subset.empty:
                continue
            w_value = float(subset["kendalls_w_all_features"].iloc[0])
            assert f"Kendall's W = {w_value:.3f}" in report_text, (
                f"{target}: W={w_value:.3f} not quoted in report"
            )


class TestReframedFraming:
    """Phase 2 locked the report onto the three-mechanisms thesis. These
    tests keep the framing from drifting back to a screening claim the
    data cannot support."""

    def test_title_is_the_mechanisms_thesis(self, report_text):
        first_line = report_text.splitlines()[0]
        assert first_line.startswith("# ")
        assert "mechanism" in first_line.lower()
        assert "Bangladesh" not in first_line

    def test_all_three_mechanisms_have_result_sections(self, report_text):
        headings = re.findall(r"^### 5\.\d+ .*$", report_text, re.MULTILINE)
        blob = " ".join(headings).lower()
        assert "leakage audit" in blob
        assert "case mix" in blob
        assert "pseudo-external" in blob

    def test_provenance_contradiction_is_stated_where_provenance_is(self, report_text):
        match = re.search(r"### 3\.1 .*?\n(.*?)\n### 3\.2", report_text, re.DOTALL)
        assert match, "section 3.1 not found"
        section = match.group(1)
        assert "Bangladesh" in section, "documented provenance should still be reported"
        assert "contradicted" in section.lower()
        assert "unverified" in section.lower()

    def test_no_screening_performance_claim_survives(self, report_text):
        """A headline number may never be asserted AS screening
        performance: the sample is not a screening series. The phrase may
        appear only inside a sentence that denies or qualifies it."""
        denial = re.compile(
            r"\bnot\b|\bnever\b|\bno\b|\bcannot\b|\bnothing\b"
            r"|overstate|than they are|far more like|would be required",
            re.IGNORECASE,
        )
        for match in re.finditer(r"screening performance", report_text, re.IGNORECASE):
            start = report_text.rfind(".", 0, max(0, match.start() - 1)) + 1
            end = report_text.find(".", match.end())
            sentence = report_text[start: end if end != -1 else len(report_text)]
            assert denial.search(sentence), (
                f"unqualified screening-performance claim: {sentence.strip()[:200]}"
            )

    def test_bangladesh_framing_confined_to_provenance_discussion(self, report_text):
        """The collection site is unverified, so it may appear only where
        the report is discussing provenance - never in a title, claim or
        conclusion."""
        for match in re.finditer(r"Bangladesh", report_text):
            window = report_text[max(0, match.start() - 400): match.start() + 400]
            assert re.search(
                r"provenance|contradict|documentation|unverified|states that",
                window, re.IGNORECASE,
            ), f"Bangladesh framing outside provenance context: ...{window[:160]}"


class TestHonestAttribution:
    def test_full_valid_auc_never_labelled_laboratory(self, conclusion):
        """The original conclusion defect: the Full-valid ROC-AUC captioned
        as 'the full laboratory panel'. Wherever the selected model's AUC is
        quoted in the conclusion, the configuration named next to it must be
        its own."""
        selected = table("table_16_selected_models.csv")
        best = selected[selected["selection"] == "best_valid_overall"].iloc[0]
        audit = table("table_13_leakage_audit.csv")
        lab = audit[audit["config"] == "laboratory_model"].iloc[0]
        if abs(best["roc_auc"] - lab["roc_auc"]) > 5e-4:
            value = f"{best['roc_auc']:.3f}"
            for match in re.finditer(re.escape(value), conclusion):
                window = conclusion[match.start(): match.start() + 160]
                assert not re.search(r"for (a|the) laboratory", window), (
                    "selected-model AUC attributed to the laboratory "
                    f"configuration: ...{window[:120]}..."
                )

    def test_every_selected_cell_quote_names_model_and_calibration(self, conclusion):
        """The 0.992-vs-0.993 reconciliation: when the conclusion quotes the
        best low-cost AUC it must name the cell it came from."""
        selected = table("table_16_selected_models.csv")
        best_lc = selected[selected["selection"] == "best_low_cost"].iloc[0]
        value = f"{best_lc['roc_auc']:.3f}"
        assert value in conclusion
        idx = conclusion.find(value)
        window = conclusion[max(0, idx - 300): idx + 300]
        assert re.search(r"SVM|forest|Logistic|EBM|XGBoost", window)
        assert re.search(r"[Ii]sotonic|[Ss]igmoid|[Pp]latt|[Uu]ncalibrated", window)

    def test_section_5_3_declares_uncalibrated_provenance(self, report_text):
        match = re.search(r"### 5\.3 .*?\n(.*?)\n### 5\.4", report_text, re.DOTALL)
        assert match
        section = match.group(1)
        assert "uncalibrated" in section
        assert "| Best model |" in section
