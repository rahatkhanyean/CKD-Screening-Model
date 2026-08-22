"""Tests for raw loading, metadata-row removal and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.data.bins import BinParseError, is_bin_label, parse_bin, representative
from ckd.data.clean import STAGE_ORDER
from ckd.data.load import verify_metadata_rows


class TestMetadataRemoval:
    def test_raw_has_202_rows_before_removal(self, raw_result):
        assert raw_result.shape_before == (202, 29)

    def test_exactly_two_metadata_rows_removed(self, raw_result):
        assert len(raw_result.metadata) == 2

    def test_200_patient_records_remain(self, raw_result):
        assert len(raw_result.patients) == 200

    def test_dropped_rows_really_are_metadata(self, raw_result):
        ev = verify_metadata_rows(raw_result.metadata)
        assert ev["row0_is_type_declaration"], "first dropped row is not a pure type declaration"
        assert ev["row1_declares_target"], "second dropped row does not declare the target"
        assert ev["row1_declares_meta"]

    def test_no_patient_row_looks_like_metadata(self, clean_result):
        """The literal metadata tokens must not survive into the patient table."""
        clean = clean_result.clean
        as_text = clean.astype(str)
        for token in ("discrete", "meta"):
            assert not (as_text == token).any().any(), f"token {token!r} survived into patient data"

    def test_source_line_provenance_is_contiguous(self, raw_result):
        lines = raw_result.patients["source_csv_line"].to_numpy()
        assert lines[0] == 4, "first patient should be on CSV line 4"
        assert np.array_equal(lines, np.arange(4, 204))


class TestBinParsing:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("1.019 - 1.021", 1.020),
            ("112 - 154", 133.0),
            ("< 112", 112.0),
            ("≥ 227.944", 227.944),
            (">= 227.944", 227.944),
            ("< 0", 0.0),
            ("3 - 3", 3.0),
            ("0", 0.0),
            ("1", 1.0),
        ],
    )
    def test_representative_values(self, label, expected):
        assert representative(label) == pytest.approx(expected)

    def test_open_bounds_are_infinite(self):
        assert parse_bin("< 112")[0] == -np.inf
        assert parse_bin("≥ 448")[1] == np.inf

    def test_closed_bounds(self):
        lo, hi, rep = parse_bin("112 - 154")
        assert (lo, hi) == (112.0, 154.0)
        assert lo < rep < hi

    def test_unparseable_raises(self):
        with pytest.raises(BinParseError):
            parse_bin("p")
        with pytest.raises(BinParseError):
            parse_bin("")

    def test_is_bin_label(self):
        assert is_bin_label("< 3.65")
        assert not is_bin_label(" p ")

    def test_encoding_is_stateless(self, clean_result):
        """The same label must map to the same number regardless of context.

        This is the property that makes it safe to encode before splitting.
        """
        for col, bins in clean_result.bin_map.items():
            for b in bins:
                if col == "stage":
                    continue
                if str(b["label"]) == "nan":
                    continue
                assert representative(b["label"]) == pytest.approx(b["representative"])

    def test_encoding_uses_no_cross_row_information(self, clean_result):
        """Encoding a single row alone gives the same values as encoding all rows."""
        from ckd.data.bins import representative as rep

        clean = clean_result.clean
        encoded = clean_result.encoded
        for row in (0, 17, 99, 199):
            for col in encoded.columns:
                if col == "source_csv_line":
                    continue
                label = clean.iloc[row][col]
                if pd.isna(label):
                    assert np.isnan(encoded.iloc[row][col])
                    continue
                expected = STAGE_ORDER[label] if col == "stage" else rep(label)
                assert encoded.iloc[row][col] == pytest.approx(expected)


class TestBinMonotonicity:
    def test_representatives_are_non_decreasing_within_each_feature(self, clean_result):
        for col, bins in clean_result.bin_map.items():
            if col == "class" or len(bins) < 2:
                continue
            reps = [b["representative"] for b in bins]
            assert reps == sorted(reps), f"{col} representatives are not ordered"

    def test_only_su_has_tied_representatives(self, clean_result):
        """Documented data defect: the published su bins overlap.

        If this test starts failing, the released file changed and the
        data-quality report must be regenerated.
        """
        from ckd.data.quality import audit_bin_monotonicity

        findings = audit_bin_monotonicity(clean_result.bin_map)
        columns = {f["column"] for f in findings}
        assert columns == {"su"}, f"unexpected bin anomalies: {columns}"


class TestCleanedDataset:
    def test_target_is_binary_and_complete(self, y):
        assert set(np.unique(y)) == {0, 1}
        assert len(y) == 200

    def test_class_counts_match_source(self, y):
        assert int(y.sum()) == 128
        assert int((1 - y).sum()) == 72

    def test_p_token_became_missing(self, clean_result):
        cells = [c for c in clean_result.missing_cells if c["column"] == "grf"]
        assert len(cells) == 1, "expected exactly one anomalous grf cell"
        assert cells[0]["raw_value"] == "' p '"
        assert cells[0]["source_csv_line"] == 183

    def test_only_one_missing_cell_in_whole_dataset(self, clean_result):
        total = clean_result.clean.drop(columns=["source_csv_line"]).isna().sum().sum()
        assert total == 1

    def test_encoded_matrix_is_numeric(self, X):
        assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)

    def test_encoded_matrix_has_no_target_column(self, X):
        assert "class" not in X.columns
        assert "target" not in X.columns

    def test_raw_file_untouched(self, raw_result):
        """The pipeline must never rewrite the raw input."""
        from ckd.data.load import sha256_of

        assert sha256_of(raw_result.source_path) == raw_result.sha256

    def test_no_exact_duplicate_records(self, clean_result):
        dupes = clean_result.clean.drop(columns=["source_csv_line"]).duplicated().sum()
        assert dupes == 0
