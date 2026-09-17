"""Tests for ``molactagent.io`` — loading + validation."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from molactagent.io import (
    PreparedData,
    ValidationResult,
    apply_id_column,
    canonical_class_order,
    load_background,
    load_feature_descriptions,
    load_table,
    prepare_for_agent,
    validate_inputs,
)


# ── load_table ───────────────────────────────────────────────────────


class TestLoadTable:
    def test_csv(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("a,b\n1,2\n3,4\n")
        df = load_table(path)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["a", "b"]

    def test_tsv(self, tmp_path):
        path = tmp_path / "x.tsv"
        path.write_text("a\tb\n1\t2\n3\t4\n")
        df = load_table(path)
        assert df.shape == (2, 2)

    def test_txt_autodetect(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("a;b\n1;2\n3;4\n")
        df = load_table(path)
        assert df.shape == (2, 2)

    def test_json(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text(json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}]))
        df = load_table(path)
        assert df.shape == (2, 2)

    def test_jsonl(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')
        df = load_table(path)
        assert df.shape == (2, 2)

    def test_parquet(self, tmp_path):
        pa = pytest.importorskip("pyarrow")  # noqa: F841
        path = tmp_path / "x.parquet"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(path)
        df = load_table(path)
        assert df.shape == (2, 2)

    def test_xlsx(self, tmp_path):
        pytest.importorskip("openpyxl")
        path = tmp_path / "x.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(path, index=False)
        df = load_table(path)
        assert df.shape == (2, 2)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_table(tmp_path / "nope.csv")

    def test_directory(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            load_table(tmp_path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="0 bytes"):
            load_table(path)

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "x.weird"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported table format"):
            load_table(path)

    def test_malformed_csv(self, tmp_path):
        path = tmp_path / "broken.csv"
        path.write_text('a,b\n1,"unterminated\n')
        with pytest.raises(ValueError):
            load_table(path)

    def test_header_only_csv_raises_or_empty(self, tmp_path):
        # Pandas treats 'a,b\n' as a 0-row, 2-column DataFrame (no error).
        path = tmp_path / "header.csv"
        path.write_text("a,b\n")
        df = load_table(path)
        assert df.shape == (0, 2)


# ── load_feature_descriptions ────────────────────────────────────────


class TestLoadDescriptions:
    def test_json(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"x": "a", "y": "b"}))
        d = load_feature_descriptions(path)
        assert d == {"x": "a", "y": "b"}

    def test_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "d.yaml"
        path.write_text("x: a\ny: b\n")
        d = load_feature_descriptions(path)
        assert d == {"x": "a", "y": "b"}

    def test_none_passthrough(self):
        assert load_feature_descriptions(None) is None

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_feature_descriptions(tmp_path / "nope.json")

    def test_bad_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_feature_descriptions(path)

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "d.xml"
        path.write_text("<x/>")
        with pytest.raises(ValueError, match="must be .json"):
            load_feature_descriptions(path)

    def test_not_a_mapping(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="must be a mapping"):
            load_feature_descriptions(path)

    def test_value_coerced_to_str(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"x": 42, "y": None}))
        d = load_feature_descriptions(path)
        assert d == {"x": "42", "y": ""}


# ── load_background ──────────────────────────────────────────────────


class TestLoadBackground:
    def test_none(self):
        assert load_background(None) is None

    def test_verbatim(self):
        assert load_background("hello world") == "hello world"

    def test_at_path_syntax(self, tmp_path):
        p = tmp_path / "bg.txt"
        p.write_text(" research notes go here \n")
        assert load_background(f"@{p}") == "research notes go here"

    def test_at_path_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_background(f"@{tmp_path}/nope.txt")

    def test_existing_path_string(self, tmp_path):
        p = tmp_path / "bg.txt"
        p.write_text("hello\n")
        assert load_background(str(p)) == "hello"

    def test_blank_string_returns_none(self):
        assert load_background("   ") is None


# ── canonical_class_order ────────────────────────────────────────────


class TestCanonicalClassOrder:
    def test_two_classes_sorted(self):
        assert canonical_class_order(["inactive", "active"]) == ["active", "inactive"]

    def test_three_classes_sorted(self):
        assert canonical_class_order(["c", "a", "b"]) == ["a", "b", "c"]

    def test_dedupe(self):
        assert canonical_class_order(["x", "y", "x"]) == ["x", "y"]

    def test_drops_none(self):
        assert canonical_class_order(["a", None, "b"]) == ["a", "b"]

    def test_coerces_non_string_labels(self):
        assert canonical_class_order([1, 0, 1]) == ["0", "1"]


# ── validate_inputs (the heart of the test suite) ────────────────────


class TestValidateInputs:
    def _ref(self, classes=("active", "inactive"), per_class=3, n_features=4):
        rng = np.random.default_rng(0)
        rows, labels = [], []
        for ci, c in enumerate(classes):
            mu = 5.0 if ci == 0 else -5.0
            for _ in range(per_class):
                rows.append(rng.normal(loc=mu, scale=0.5, size=n_features))
                labels.append(c)
        df = pd.DataFrame(rows, columns=[f"f{i}" for i in range(n_features)])
        df["activity"] = labels
        return df

    def _test(self, n=2, n_features=4):
        rng = np.random.default_rng(1)
        rows = rng.normal(size=(n, n_features))
        df = pd.DataFrame(rows, columns=[f"f{i}" for i in range(n_features)])
        return df

    # happy path ----------------------------------------------------
    def test_happy_path(self):
        result = validate_inputs(self._ref(), self._test(), "activity", k=2)
        assert result.is_valid is True
        assert bool(result) is True
        assert result.errors == []
        assert result.summary["n_features"] == 4
        # Multi-class: summary now carries the canonical class order
        # rather than positive/negative class names.
        assert result.summary["class_labels"] == ["active", "inactive"]

    def test_happy_path_with_descriptions(self):
        descs = {f"f{i}": f"feature{i}" for i in range(4)}
        descs["activity"] = "label"
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, feature_descriptions=descs,
        )
        assert result.is_valid

    # ── label column problems ────────────────────────────────────
    def test_label_column_missing(self):
        result = validate_inputs(self._ref(), self._test(), "nope", k=2)
        assert not result.is_valid
        assert any("Label column 'nope' is not in the reference" in e for e in result.errors)

    def test_label_column_one_class(self):
        ref = self._ref(classes=("active",), per_class=4)
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert not result.is_valid
        assert any("only contains one class" in e for e in result.errors)

    def test_label_column_three_classes(self):
        # Multi-class is now supported; three classes is a valid input.
        ref = self._ref(classes=("a", "b", "c"), per_class=2)
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert result.is_valid
        assert sorted(result.summary["reference_classes"]) == ["a", "b", "c"]
        assert result.summary["class_labels"] == ["a", "b", "c"]

    def test_too_many_classes_warns(self):
        # 11 classes triggers a non-blocking warning that the LLM may
        # struggle to weigh many candidates per call.
        labels = [f"cls{i}" for i in range(11)]
        ref = self._ref(classes=tuple(labels), per_class=2, n_features=4)
        result = validate_inputs(ref, self._test(), "activity", k=1)
        assert result.is_valid
        assert any("11 distinct" in w for w in result.warnings)

    def test_label_column_all_nan(self):
        ref = self._ref()
        ref["activity"] = np.nan
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert not result.is_valid
        assert any("Every value in the label column" in e for e in result.errors)

    def test_label_column_some_nan(self):
        ref = self._ref()
        ref.loc[0, "activity"] = np.nan
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert not result.is_valid
        assert any("missing value" in e for e in result.errors)

    def test_minimum_total_samples(self):
        # 1 sample per class — below the spec minimum of 2 total IS still
        # 2 total. We verify the strict "needs >= 2 reference samples" path.
        ref = self._ref(per_class=1)
        result = validate_inputs(ref, self._test(), "activity", k=1)
        # 2 total, so it should pass on the row-count check
        assert any(c == 1 for c in result.summary["reference_class_counts"].values())
        assert result.is_valid

    def test_truly_below_minimum(self):
        ref = self._ref(classes=("active",), per_class=1)
        result = validate_inputs(ref, self._test(), "activity", k=1)
        assert not result.is_valid

    # ── reference structural problems ────────────────────────────
    def test_reference_zero_rows(self):
        ref = pd.DataFrame(columns=["f0", "f1", "activity"])
        result = validate_inputs(ref, self._test(n_features=2), "activity", k=2)
        assert not result.is_valid
        assert any("Reference dataset has 0 rows" in e for e in result.errors)

    def test_reference_zero_columns(self):
        ref = pd.DataFrame()
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert not result.is_valid
        assert any("Reference dataset has 0 columns" in e for e in result.errors)

    def test_reference_only_label_column(self):
        ref = pd.DataFrame({"activity": ["a", "a", "b", "b"]})
        tst = pd.DataFrame({"activity": ["a"]})
        result = validate_inputs(ref, tst, "activity", k=2)
        assert not result.is_valid
        assert any("no feature columns" in e for e in result.errors)

    # ── test dataset problems ────────────────────────────────────
    def test_test_zero_rows(self):
        tst = pd.DataFrame(columns=["f0", "f1", "f2", "f3"])
        result = validate_inputs(self._ref(), tst, "activity", k=2)
        assert not result.is_valid
        assert any("Test dataset has 0 rows" in e for e in result.errors)

    def test_test_missing_features(self):
        tst = self._test().drop(columns=["f0", "f1"])
        result = validate_inputs(self._ref(), tst, "activity", k=2)
        assert not result.is_valid
        assert any("missing 2 feature column" in e for e in result.errors)

    def test_test_extra_features_warns(self):
        tst = self._test()
        tst["extra"] = 1.0
        result = validate_inputs(self._ref(), tst, "activity", k=2)
        assert result.is_valid
        assert any("extra column" in w for w in result.warnings)

    def test_test_with_label_column(self):
        tst = self._test()
        tst["activity"] = ["active", "inactive"]
        result = validate_inputs(self._ref(), tst, "activity", k=2)
        assert result.is_valid

    # ── feature column issues ────────────────────────────────────
    def test_non_numeric_feature(self):
        ref = self._ref()
        ref["f0"] = ["x", "y", "z", "x", "y", "z"]
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert not result.is_valid
        assert any("non-numeric" in e for e in result.errors)

    def test_all_nan_feature_warns(self):
        ref = self._ref()
        ref["f0"] = np.nan
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert result.is_valid
        assert any("entirely missing" in w for w in result.warnings)

    def test_constant_feature_warns(self):
        ref = self._ref()
        ref["f0"] = 1.0
        result = validate_inputs(ref, self._test(), "activity", k=2)
        assert result.is_valid
        assert any("zero variance" in w for w in result.warnings)

    # ── k parameter ───────────────────────────────────────────────
    def test_k_zero(self):
        result = validate_inputs(self._ref(), self._test(), "activity", k=0)
        assert not result.is_valid
        assert any("k must be >= 1" in e for e in result.errors)

    def test_k_negative(self):
        result = validate_inputs(self._ref(), self._test(), "activity", k=-3)
        assert not result.is_valid
        assert any("k must be >= 1" in e for e in result.errors)

    def test_k_float(self):
        result = validate_inputs(self._ref(), self._test(), "activity", k=2.5)  # type: ignore[arg-type]
        assert not result.is_valid
        assert any("must be an integer" in e for e in result.errors)

    def test_k_bool_rejected(self):
        # bool is technically a subclass of int — make sure we reject it.
        result = validate_inputs(self._ref(), self._test(), "activity", k=True)  # type: ignore[arg-type]
        assert not result.is_valid

    def test_k_too_large_warns(self):
        result = validate_inputs(self._ref(), self._test(), "activity", k=10)
        assert result.is_valid
        assert any("but k=10" in w for w in result.warnings)

    # ── feature_descriptions completeness ─────────────────────────
    def test_descriptions_missing_columns(self):
        descs = {"f0": "x"}
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, feature_descriptions=descs,
        )
        assert not result.is_valid
        assert any("Feature descriptions are missing" in e for e in result.errors)
        # Must mention the label column was missing
        assert any("activity" in e for e in result.errors)

    def test_descriptions_missing_label_column(self):
        descs = {f"f{i}": "x" for i in range(4)}
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, feature_descriptions=descs,
        )
        assert not result.is_valid
        assert any("activity" in e for e in result.errors)

    def test_descriptions_complete(self):
        descs = {f"f{i}": "x" for i in range(4)}
        descs["activity"] = "label"
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, feature_descriptions=descs,
        )
        assert result.is_valid

    def test_descriptions_extra_keys_warn(self):
        descs = {f"f{i}": "x" for i in range(4)}
        descs["activity"] = "label"
        descs["bogus"] = "ignored"
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, feature_descriptions=descs,
        )
        assert result.is_valid
        assert any("not present in the dataset" in w for w in result.warnings)

    def test_descriptions_non_string_value(self):
        descs = {f"f{i}": "x" for i in range(4)}
        descs["activity"] = 42  # type: ignore[assignment]
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, feature_descriptions=descs,
        )
        assert not result.is_valid
        assert any("must be strings" in e for e in result.errors)

    # ── background ────────────────────────────────────────────────
    def test_long_background_warns(self):
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, background="x" * 9000,
        )
        assert result.is_valid
        assert any("very long" in w for w in result.warnings)

    # ── positive_class is deprecated under multi-class ────────────
    def test_positive_class_emits_deprecation_warning(self):
        # The flag is still accepted (back-compat) but its only effect
        # is to surface a deprecation warning in the validation report.
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, positive_class="active",
        )
        assert result.is_valid
        assert any(
            "positive_class" in w and "deprecated" in w
            for w in result.warnings
        )

    def test_positive_class_unknown_value_still_accepted(self):
        # Even a non-existent class name is silently ignored under the
        # multi-class design (a deprecation warning is emitted instead).
        result = validate_inputs(
            self._ref(), self._test(), "activity",
            k=2, positive_class="missing",
        )
        assert result.is_valid
        assert any("deprecated" in w for w in result.warnings)

    # ── type errors (programmer mistakes, not user data) ──────────
    def test_typeerror_when_not_dataframe(self):
        with pytest.raises(TypeError):
            validate_inputs([[1, 2]], self._test(), "activity", k=2)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            validate_inputs(self._ref(), {"a": 1}, "activity", k=2)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            validate_inputs(self._ref(), self._test(), "", k=2)

    # ── multiple errors collected at once ─────────────────────────
    def test_multiple_errors_collected(self):
        ref = self._ref(classes=("a", "b", "c"), per_class=2)  # 3 classes
        ref["f0"] = ["x"] * len(ref)  # non-numeric
        tst = self._test().drop(columns=["f0"])  # missing column
        result = validate_inputs(ref, tst, "activity", k=0)  # k invalid
        assert not result.is_valid
        # Should have AT LEAST 3 problems flagged
        assert len(result.errors) >= 3

    # ── ValidationResult formatting ───────────────────────────────
    def test_format_report_contains_summary(self):
        result = validate_inputs(self._ref(), self._test(), "activity", k=2)
        report = result.format_report()
        assert "Validation: OK" in report
        assert "n_features" in report
        assert "n_reference_samples" in report

    def test_format_report_failed(self):
        result = validate_inputs(self._ref(), self._test(), "nope", k=2)
        report = result.format_report()
        assert "Validation: FAILED" in report
        assert "Errors" in report


# ── prepare_for_agent ────────────────────────────────────────────────


class TestPrepareForAgent:
    def test_basic(self, simple_reference, simple_test):
        prep = prepare_for_agent(simple_reference, simple_test, "activity")
        assert isinstance(prep, PreparedData)
        assert prep.X_ref.shape == (6, 4)
        assert prep.X_test.shape == (2, 4)
        # Multi-class: y_ref carries the original user labels verbatim;
        # no internal Pos/Neg remap. class_labels is the canonical
        # sorted order.
        assert prep.class_labels == ["active", "inactive"]
        assert set(prep.y_ref) == {"active", "inactive"}
        assert prep.feature_names == ["alpha", "beta", "gamma", "delta"]

    def test_three_classes(self, multiclass_reference, multiclass_test):
        prep = prepare_for_agent(multiclass_reference, multiclass_test, "bucket")
        assert prep.class_labels == ["high", "low", "mid"]
        assert set(prep.y_ref) == {"high", "low", "mid"}
        assert prep.X_ref.shape == (12, 4)
        assert prep.X_test.shape == (3, 4)

    def test_positive_class_silently_ignored(self, simple_reference, simple_test):
        # Back-compat: positive_class is accepted as a kwarg but has no
        # effect on the prepared output under the multi-class design.
        prep = prepare_for_agent(
            simple_reference, simple_test, "activity",
            positive_class="active",
        )
        assert prep.class_labels == ["active", "inactive"]
        assert set(prep.y_ref) == {"active", "inactive"}

    def test_drops_all_nan_columns(self, simple_reference, simple_test):
        ref = simple_reference.copy()
        ref["alpha"] = np.nan
        tst = simple_test.copy()
        prep = prepare_for_agent(ref, tst, "activity")
        assert "alpha" not in prep.feature_names
        assert prep.X_ref.shape[1] == 3

    def test_test_column_reordering(self, simple_reference, simple_test):
        scrambled = simple_test[["delta", "gamma", "beta", "alpha", "activity"]]
        prep = prepare_for_agent(simple_reference, scrambled, "activity")
        # X_test should be reordered to match the reference column order
        assert prep.feature_names == ["alpha", "beta", "gamma", "delta"]
        np.testing.assert_array_almost_equal(
            prep.X_test[:, 0], scrambled["alpha"].to_numpy(),
        )

    def test_captures_test_truth(self, simple_reference, simple_test):
        prep = prepare_for_agent(simple_reference, simple_test, "activity")
        assert prep.test_true_labels is not None
        assert prep.test_true_labels.tolist() == ["active", "inactive"]

    def test_no_test_truth(self, simple_reference, simple_test):
        tst = simple_test.drop(columns=["activity"])
        prep = prepare_for_agent(simple_reference, tst, "activity")
        assert prep.test_true_labels is None


# ── Tricky label handling ────────────────────────────────────────────


class TestTrickyLabels:
    def _ref(self, labels):
        rng = np.random.default_rng(0)
        n = len(labels)
        X = rng.normal(size=(n, 3))
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["lbl"] = labels
        return df

    def _test(self, n=2):
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            rng.normal(size=(n, 3)),
            columns=["a", "b", "c"],
        )

    def test_whitespace_in_labels_normalized(self):
        ref = self._ref([" yes ", "yes", "no", " no  "])
        result = validate_inputs(ref, self._test(), "lbl", k=1)
        assert result.is_valid
        assert sorted(result.summary["reference_classes"]) == ["no", "yes"]

    def test_mixed_int_and_str_labels(self):
        # A common bug: pandas reads CSVs with mixed int/str into object dtype
        ref = self._ref([1, 1, "1", "0", 0])
        result = validate_inputs(ref, self._test(), "lbl", k=1)
        assert result.is_valid
        assert sorted(result.summary["reference_classes"]) == ["0", "1"]

    def test_boolean_labels(self):
        ref = self._ref([True, True, False, False])
        result = validate_inputs(ref, self._test(), "lbl", k=1)
        assert result.is_valid
        assert sorted(result.summary["reference_classes"]) == ["False", "True"]

    def test_float_integer_labels(self):
        ref = self._ref([1.0, 1.0, 0.0, 0.0])
        result = validate_inputs(ref, self._test(), "lbl", k=1)
        assert result.is_valid
        assert sorted(result.summary["reference_classes"]) == ["0", "1"]

    def test_one_class_with_only_one_sample_still_valid(self):
        # Spec: minimum 2 samples total, so 1 + 1 is the floor.
        ref = self._ref(["yes", "no"])
        result = validate_inputs(ref, self._test(n=1), "lbl", k=1)
        assert result.is_valid

    def test_one_class_with_many_one_with_one(self):
        ref = self._ref(["yes", "yes", "yes", "yes", "no"])
        result = validate_inputs(ref, self._test(), "lbl", k=2)
        assert result.is_valid
        assert any("but k=2" in w for w in result.warnings)

    def test_unicode_labels(self):
        ref = self._ref(["阳性", "阳性", "阴性", "阴性"])
        result = validate_inputs(ref, self._test(), "lbl", k=1)
        assert result.is_valid


# ── Robust validation reporting ─────────────────────────────────────


class TestValidationReportFormatting:
    def test_warnings_section_appears(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, 3))
        ref = pd.DataFrame(X, columns=["a", "b", "c"])
        ref["lbl"] = ["yes", "yes", "no", "no"]
        ref["a"] = 0.0  # constant column -> warning
        tst = pd.DataFrame(rng.normal(size=(2, 3)), columns=["a", "b", "c"])
        tst["extra"] = "junk"  # extra column -> warning
        result = validate_inputs(ref, tst, "lbl", k=2)
        assert result.is_valid
        report = result.format_report()
        assert "Warnings" in report
        assert "zero variance" in report
        assert "extra column" in report

    def test_summary_keys(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(rng.normal(size=(4, 3)), columns=["a", "b", "c"])
        ref["lbl"] = ["yes", "yes", "no", "no"]
        tst = pd.DataFrame(rng.normal(size=(2, 3)), columns=["a", "b", "c"])
        result = validate_inputs(ref, tst, "lbl", k=2)
        for key in (
            "reference_classes", "reference_class_counts",
            "n_features", "feature_columns",
            "n_reference_samples", "n_test_samples",
            "k", "class_labels",
        ):
            assert key in result.summary, f"Missing summary key: {key}"


# ── apply_id_column ─────────────────────────────────────────────────


class TestApplyIdColumn:
    def _df(self):
        return pd.DataFrame({
            "sid": ["A", "B", "C"],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })

    def test_none_passthrough(self):
        df = self._df()
        out = apply_id_column(df, None)
        assert out is df  # same object

    def test_moves_column_to_index(self):
        out = apply_id_column(self._df(), "sid")
        assert list(out.columns) == ["x", "y"]
        assert list(out.index) == ["A", "B", "C"]
        assert out.index.name == "sid"

    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found"):
            apply_id_column(self._df(), "nope")

    def test_duplicate_ids_raise(self):
        df = self._df()
        df.loc[2, "sid"] = "A"
        with pytest.raises(ValueError, match="duplicate"):
            apply_id_column(df, "sid")

    def test_missing_id_values_raise(self):
        df = self._df()
        df.loc[1, "sid"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            apply_id_column(df, "sid")

    def test_does_not_mutate_original(self):
        df = self._df()
        before = df.copy()
        apply_id_column(df, "sid")
        pd.testing.assert_frame_equal(df, before)

    def test_integer_ids_become_strings(self):
        df = pd.DataFrame({"id": [10, 20, 30], "x": [1.0, 2.0, 3.0]})
        out = apply_id_column(df, "id")
        assert list(out.index) == ["10", "20", "30"]
