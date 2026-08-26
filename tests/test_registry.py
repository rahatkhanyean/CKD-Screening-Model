"""The feature registry must agree exactly with the declared configurations.

The registry is introduced as the declarative source of truth for feature
classification. That is only safe if deriving configurations from it
reproduces the definitions the reference analysis was run with, exactly. If
these tests fail, either the registry or `configs.py` has drifted and the
published results no longer correspond to the stated feature sets.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ckd.features.configs import (
    FEATURE_CONFIGS,
    FORBIDDEN_IN_VALID,
    valid_config_names,
)
from ckd.features.registry import (
    CAVEATED_CATEGORIES,
    PROHIBITED_CATEGORIES,
    caveated_variables,
    incorporation_variables,
    configuration_features,
    configuration_names,
    load_registry,
    prohibited_variables,
    variables_by_category,
)


@pytest.fixture(scope="module")
def registry() -> pd.DataFrame:
    return load_registry()


class TestRegistrySchema:
    def test_every_variable_appears_once(self, registry):
        assert registry["variable"].is_unique

    def test_every_variable_has_a_category(self, registry):
        assert registry["leakage_category"].notna().all()
        allowed = PROHIBITED_CATEGORIES | CAVEATED_CATEGORIES | {
            "legitimate_pre_index"
        }
        assert set(registry["leakage_category"]) <= allowed

    def test_every_variable_carries_evidence(self, registry):
        assert (registry["evidence"].fillna("").str.len() > 0).all()

    def test_uncertain_variables_carry_a_note(self, registry):
        uncertain = registry[registry["uncertainty_flag"]]
        assert (uncertain["uncertainty_note"].fillna("").str.len() > 0).all()


class TestRegistryMatchesDeclaredConfigurations:
    def test_prohibited_set_matches(self):
        """The registry's prohibited categories must reproduce exactly the
        set the leakage guards were built from."""
        assert prohibited_variables() == FORBIDDEN_IN_VALID

    def test_configuration_names_match(self):
        assert set(configuration_names()) == set(FEATURE_CONFIGS)

    @pytest.mark.parametrize("config_name", sorted(FEATURE_CONFIGS))
    def test_configuration_membership_matches(self, config_name):
        derived = set(configuration_features(config_name))
        declared = set(FEATURE_CONFIGS[config_name].features)
        assert derived == declared, (
            f"{config_name}: registry-derived features differ from "
            f"configs.py. Only in registry: {sorted(derived - declared)}; "
            f"only in configs.py: {sorted(declared - derived)}"
        )

    def test_no_prohibited_variable_in_any_valid_configuration(self):
        for config_name in valid_config_names():
            assert not (set(configuration_features(config_name))
                        & prohibited_variables())


class TestTaxonomyContent:
    def test_the_outcome_and_its_copy_are_direct_copies(self):
        assert set(variables_by_category("direct_outcome_copy")) == {
            "class", "affected"
        }

    def test_stage_and_egfr_are_deterministic_derivatives(self):
        assert set(variables_by_category("deterministic_outcome_derivative")) == {
            "stage", "grf"
        }

    def test_both_kdigo_limbs_are_flagged_as_criterion_inputs(self):
        """KDIGO defines CKD by the GFR limb *or* a kidney-damage marker.

        The released file carries both: creatinine feeds the eGFR equation,
        and urine albumin is the albuminuria marker (ACR >=30 mg/g is the
        first-listed damage criterion). An earlier registry recorded only
        creatinine and treated albumin as an ordinary predictor, which made
        every "incorporation removed" statement false --- the albuminuria
        limb stayed in the model. Both must be recorded.
        """
        caveats = caveated_variables()
        assert caveats.get("sc") == "diagnostic_criterion_input"
        assert caveats.get("al") == "diagnostic_criterion_input"

    def test_blood_urea_is_not_a_diagnostic_criterion(self):
        """Blood urea appears nowhere in the KDIGO definition of CKD.

        It was previously classified as an incorporation risk. That
        overstated the leakage claim: urea is a correlated renal biomarker,
        confounded by protein intake, catabolism, hydration and GI bleeding,
        not a criterion the diagnosis is built from.
        """
        caveats = caveated_variables()
        assert caveats.get("bu") == "correlated_renal_biomarker"
        assert "bu" not in incorporation_variables()
        assert "bu" not in incorporation_variables(strict=True)

    def test_incorporation_set_covers_albuminuria_in_both_readings(self):
        """No incorporation set may omit the albuminuria limb."""
        assert "al" in incorporation_variables(strict=True)
        assert "al" in incorporation_variables()
        # The broad reading additionally sweeps in sediment findings.
        assert incorporation_variables(strict=True) < incorporation_variables()

    def test_anaemia_markers_are_flagged_as_consequences(self):
        consequences = set(variables_by_category("consequence_of_advanced_disease"))
        assert {"hemo", "pcv", "rbcc"} <= consequences

    def test_every_caveated_variable_is_disclosed_with_a_reason(self):
        for variable, category in caveated_variables().items():
            assert category in CAVEATED_CATEGORIES, (variable, category)
