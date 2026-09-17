"""Tests for ``molactagent.tl`` — evaluation, baselines, calibration,
and cohort-level driver attribution."""

from __future__ import annotations

import numpy as np
import pytest

from molactagent import (
    MolActAgentClassifier,
    MolActAgentRegressor,
)
from molactagent.tl import (
    aggregate_drivers,
    compare_baselines,
    confidence_calibration,
    expected_calibration_error,
    kfold_cv,
    loocv,
)


# ── loocv on the FakeLLM-backed estimator ─────────────────────────


def test_loocv_classification_runs(
    multiclass_reference, fake_llm,
):
    """LOOCV with a tiny multi-class fixture; smoke-test the runner."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y = multiclass_reference["bucket"].to_numpy()

    clf = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    )

    result = loocv(clf, X, y, progress=False)
    assert result.task == "classification"
    assert result.n_folds == len(y)
    assert result.metrics.n_samples == len(y)
    assert len(result.fold_records) == len(y)
    # Every fold record must carry the expected keys.
    for r in result.fold_records:
        assert "true" in r
        assert "pred" in r
        assert "confidence" in r
        assert "raw" in r


def test_loocv_regression_runs(
    regression_reference, fake_llm,
):
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X = regression_reference[feature_cols].to_numpy(dtype=float)
    y = regression_reference["target"].to_numpy(dtype=float)

    reg = MolActAgentRegressor(
        llm_client=fake_llm,
        k=5,
        feature_names=list(feature_cols),
        max_workers=2,
    )

    result = loocv(reg, X, y, progress=False)
    assert result.task == "regression"
    assert result.n_folds == len(y)
    assert result.metrics.n == len(y)
    # Predictions are floats (or NaN for refusals).
    assert result.predictions.dtype == float


def test_loocv_dataframe_excludes_raw(
    multiclass_reference, fake_llm,
):
    """``to_dataframe`` must NOT drag the heavy ``raw`` dict — keeps
    notebook output tractable."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y = multiclass_reference["bucket"].to_numpy()
    clf = MolActAgentClassifier(
        llm_client=fake_llm, k=3, feature_names=list(feature_cols),
        max_workers=2,
    )
    df = loocv(clf, X, y, progress=False).to_dataframe()
    assert "raw" not in df.columns
    assert "fold" in df.columns
    assert "pred" in df.columns


# ── kfold_cv parity with sklearn ───────────────────────────────


def test_kfold_cv_classification(
    multiclass_reference, fake_llm,
):
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y = multiclass_reference["bucket"].to_numpy()

    clf = MolActAgentClassifier(
        llm_client=fake_llm, k=3, feature_names=list(feature_cols),
        max_workers=2,
    )

    result = kfold_cv(clf, X, y, n_splits=3, progress=False)
    assert result.n_folds == 3
    # Stratified split — all 12 rows assigned exactly once.
    assert sum(1 for _ in result.fold_records) == 12


# ── compare_baselines ───────────────────────────────────────────


def test_compare_baselines_classification_minimal(
    multiclass_reference, fake_llm,
):
    """ML-only baselines (cheap, no LLM tokens spent)."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y = multiclass_reference["bucket"].to_numpy()

    comp = compare_baselines(
        molactagent_estimator=None,
        X=X, y=y,
        baselines=["KNN3", "LR"],
        cv="loocv",
        include_molactagent=False,
        progress=False,
    )
    assert set(comp.methods) == {"KNN3", "LR"}
    table = comp.metrics_table()
    assert "KNN3" in table
    assert "LR" in table


def test_compare_baselines_with_molactagent(
    multiclass_reference, fake_llm,
):
    """Full comparison with MolActAgent + 1 baseline."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y = multiclass_reference["bucket"].to_numpy()

    clf = MolActAgentClassifier(
        llm_client=fake_llm, k=3, feature_names=list(feature_cols),
        max_workers=2,
    )
    comp = compare_baselines(
        molactagent_estimator=clf,
        X=X, y=y,
        baselines=["KNN3"],
        cv="loocv",
        include_molactagent=True,
        progress=False,
    )
    assert "MolActAgent" in comp.methods
    assert "KNN3" in comp.methods
    table = comp.metrics_table()
    assert "MolActAgent" in table


def test_compare_baselines_default_keyword():
    """``baselines='default'`` resolves to the curated subset."""
    from molactagent.tl.baselines import (
        _DEFAULT_CLF_BASELINES,
        _resolve_baseline_keyword,
    )
    assert _resolve_baseline_keyword("default", "classification") == _DEFAULT_CLF_BASELINES


def test_compare_baselines_regression(
    regression_reference, fake_llm,
):
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X = regression_reference[feature_cols].to_numpy(dtype=float)
    y = regression_reference["target"].to_numpy(dtype=float)

    comp = compare_baselines(
        molactagent_estimator=None,
        X=X, y=y,
        baselines=["KNN3", "Ridge"],
        cv="loocv",
        include_molactagent=False,
        progress=False,
    )
    assert "KNN3" in comp.methods
    assert "Ridge" in comp.methods


# ── calibration ──────────────────────────────────────────────────


def test_calibration_perfect_predictions():
    """Perfect predictions with 1.0 confidence: ECE = 0, Brier = 0."""
    y_true = ["A", "B", "A", "B", "A"]
    y_pred = ["A", "B", "A", "B", "A"]
    confidences = [1.0, 1.0, 1.0, 1.0, 1.0]
    report = confidence_calibration(y_true, y_pred, confidences, bins=5)
    assert report.expected_calibration_error == pytest.approx(0.0, abs=1e-9)
    assert report.brier_score == pytest.approx(0.0, abs=1e-9)


def test_calibration_overconfident_wrong():
    """Wrong predictions with 1.0 confidence: ECE=1.0, Brier=1.0."""
    y_true = ["A", "B"]
    y_pred = ["B", "A"]
    confidences = [1.0, 1.0]
    report = confidence_calibration(y_true, y_pred, confidences, bins=5)
    assert report.expected_calibration_error == pytest.approx(1.0, abs=1e-9)
    assert report.brier_score == pytest.approx(1.0, abs=1e-9)


def test_calibration_from_fold_records():
    """``fold_records=`` shortcut matches the explicit-array path."""
    fold_records = [
        {"true": "A", "pred": "A", "confidence": 0.9},
        {"true": "B", "pred": "A", "confidence": 0.8},
        {"true": "B", "pred": "B", "confidence": 0.7},
    ]
    rep_via_records = confidence_calibration(
        fold_records=fold_records, bins=5,
    )
    rep_via_arrays = confidence_calibration(
        ["A", "B", "B"], ["A", "A", "B"], [0.9, 0.8, 0.7], bins=5,
    )
    assert rep_via_records.expected_calibration_error == pytest.approx(
        rep_via_arrays.expected_calibration_error,
    )


def test_expected_calibration_error_scalar():
    ece = expected_calibration_error(
        ["A", "B"], ["A", "B"], [0.9, 0.9], bins=5,
    )
    assert isinstance(ece, float)


def test_calibration_correctly_handles_binary_integer_labels():
    """Regression test for an earlier implementation-comment
    mismatch.

    The previous ``_correctness_array`` heuristic tried floats
    first and only fell back to string equality on coercion
    failure. Binary integer labels (``[0, 1, 0, 1]``) coerce to
    floats, so they were silently treated as a regression task —
    correctness was computed as ``|y - y_hat| < median(error)``,
    which is nonsensical for binary classification.

    The fixed implementation routes through
    :func:`molactagent.tl.evaluation._detect_task` and treats
    integer labels with ≤10 distinct values as classification.
    """
    # 3 correct out of 4 — the only honest answer is 75%.
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    confidences = [1.0, 1.0, 1.0, 1.0]
    report = confidence_calibration(
        y_true, y_pred, confidences, bins=5,
    )
    # ECE = |1.0 - 0.75| = 0.25 (one bin holding all four samples).
    assert report.expected_calibration_error == pytest.approx(0.25, abs=1e-6)
    assert report.brier_score == pytest.approx(0.25, abs=1e-6)


# ── attribution ──────────────────────────────────────────────────


def test_aggregate_drivers_classification(
    multiclass_reference, multiclass_test, fake_llm,
):
    """Cohort aggregation across a small classification batch."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_train = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)

    clf = MolActAgentClassifier(
        llm_client=fake_llm, k=3, feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)

    full_results = clf.predict_full(X_test)
    agg = aggregate_drivers(full_results)
    df = agg.to_dataframe()
    assert "feature" in df.columns
    assert "n_cited" in df.columns
    # Aggregation should have at least one feature recorded.
    assert len(agg.records) >= 1


def test_aggregate_drivers_handles_fold_records():
    """Aggregation accepts fold-record-shaped dicts as well as
    typed result wrappers."""
    records = [
        {
            "raw": {
                "driver_attribution": [
                    {
                        "feature": "alpha", "n_cited": 1,
                        "abs_strength": 0.5, "dominant_class": "A",
                    },
                ],
            },
        },
        {
            "raw": {
                "driver_attribution": [
                    {
                        "feature": "alpha", "n_cited": 2,
                        "abs_strength": 1.0, "dominant_class": "A",
                    },
                    {
                        "feature": "beta", "n_cited": 1,
                        "abs_strength": 0.3, "dominant_class": "B",
                    },
                ],
            },
        },
    ]
    agg = aggregate_drivers(records)
    feats = {r["feature"]: r for r in agg.records}
    assert feats["alpha"]["n_cited"] == 3
    assert feats["alpha"]["abs_strength"] == pytest.approx(1.5, abs=1e-6)
    assert feats["beta"]["n_cited"] == 1
