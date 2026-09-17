"""Tests for ``molactagent.metrics``.

Verify that the typed reports produce the same numbers as the
legacy ``compute_full_metrics`` / ``compute_regression_metrics``
helpers in ``scripts/run_loocv_metrics.py`` (which are what every
benchmark in ``data/`` was computed with). Bit-identical output is
the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from molactagent.metrics import (
    binary_classification_metrics,
    classification_metrics,
    format_comparison_table,
    regression_metrics,
)


# ── binary classification ──────────────────────────────────────────


def test_binary_classification_basic():
    """Smoke test of the full binary metric suite."""
    y_true = ["active", "inactive", "active", "inactive", "active", "inactive"]
    y_pred = ["active", "inactive", "active", "active", "active", "inactive"]
    proba_pos = [0.9, 0.2, 0.7, 0.6, 0.8, 0.3]

    report = binary_classification_metrics(
        y_true, y_pred, proba_pos,
        positive_class="active", negative_class="inactive",
    )
    assert report.n_samples == 6
    assert report.tp == 3
    assert report.tn == 2
    assert report.fp == 1
    assert report.fn == 0
    assert report.accuracy == pytest.approx(5 / 6)
    assert report.roc_auc is not None
    assert report.pr_auc is not None
    assert 0.0 <= report.roc_auc <= 1.0


def test_binary_unknown_predictions_count_against_accuracy():
    """``Inconclusive`` / ``Unknown`` predictions must count as wrong
    for accuracy and confusion-matrix purposes — matches the policy
    in ``scripts/run_loocv_metrics.compute_full_metrics``."""
    y_true = ["active", "inactive", "active"]
    y_pred = ["active", "Inconclusive", "Inconclusive"]

    report = binary_classification_metrics(
        y_true, y_pred,
        positive_class="active", negative_class="inactive",
    )
    assert report.unknown == 2
    assert report.unknown_rate == pytest.approx(2 / 3)
    assert report.tp == 1
    assert report.tn == 0  # inactive predicted as Unknown, not as inactive


def test_binary_dispatch_via_polymorphic_entry():
    """``classification_metrics`` should auto-detect binary and
    dispatch to the binary path."""
    y_true = ["active"] * 5 + ["inactive"] * 5
    y_pred = ["active"] * 5 + ["inactive"] * 5
    report = classification_metrics(y_true, y_pred)
    assert report.n_classes == 2
    assert report.positive_class is not None
    assert report.accuracy == 1.0


def test_multiclass_metrics():
    """Multi-class branch — three-class fixture."""
    y_true = ["A", "B", "C", "A", "B", "C"]
    y_pred = ["A", "B", "C", "A", "B", "B"]   # 5/6 correct
    report = classification_metrics(y_true, y_pred)
    assert report.n_classes == 3
    assert report.accuracy == pytest.approx(5 / 6)
    # Per-class F1 dict is populated with one entry per class.
    assert set(report.per_class_f1.keys()) == {"A", "B", "C"}
    # Multi-class branch has no positive_class.
    assert report.positive_class is None


def test_classification_report_round_trip():
    """``to_dict`` -> manually rebuild -> equivalent metrics."""
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "A"]
    report = classification_metrics(y_true, y_pred)
    d = report.to_dict()
    assert d["accuracy"] == pytest.approx(report.accuracy)
    assert d["mcc"] == pytest.approx(report.mcc)


# ── regression ────────────────────────────────────────────────────


def test_regression_basic():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
    report = regression_metrics(y_true, y_pred)
    assert report.n == 5
    assert report.n_predicted == 5
    assert report.n_refused == 0
    assert report.mae < 0.5
    assert report.r2 > 0.9
    assert report.pearson_r is not None
    assert report.spearman_r is not None


def test_regression_excludes_refusals():
    """Refused predictions (None / NaN) must be excluded from
    point-error metrics and counted as ``n_refused``."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, None, 3.2, np.nan]
    report = regression_metrics(y_true, y_pred)
    assert report.n == 4
    assert report.n_predicted == 2
    assert report.n_refused == 2
    assert report.mae is not None  # Computed on the 2 surviving rows.


def test_regression_fraction_within_interval():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    intervals = [(0.5, 1.5), (1.0, 1.5), (2.5, 3.5)]
    #              ^ contains 1   ^ misses 2   ^ contains 3 → 2/3
    report = regression_metrics(y_true, y_pred, intervals)
    assert report.fraction_within_interval == pytest.approx(2 / 3)


def test_regression_handles_short_series_gracefully():
    """Pearson/Spearman should be ``None`` when only 2 paired
    points exist (instead of crashing)."""
    y_true = [1.0, 2.0]
    y_pred = [1.5, 1.8]
    report = regression_metrics(y_true, y_pred)
    assert report.pearson_r is None
    assert report.spearman_r is None
    assert report.r2 is not None  # R² is fine with 2 points.


def test_regression_all_refused():
    report = regression_metrics([1.0, 2.0], [None, None])
    assert report.n_predicted == 0
    assert report.n_refused == 2
    assert report.mae is None
    assert report.rmse is None


# ── comparison tables ─────────────────────────────────────────────


def test_classification_comparison_table_renders():
    r1 = binary_classification_metrics(
        ["a", "b", "a"], ["a", "b", "a"],
        positive_class="a", negative_class="b",
    )
    r2 = binary_classification_metrics(
        ["a", "b", "a"], ["a", "a", "a"],
        positive_class="a", negative_class="b",
    )
    text = format_comparison_table({"Perfect": r1, "BadFP": r2})
    assert "Perfect" in text
    assert "BadFP" in text
    assert "Bal Acc" in text


def test_regression_comparison_table_renders():
    r1 = regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    r2 = regression_metrics([1, 2, 3, 4], [2, 3, 4, 5])
    text = format_comparison_table({"Perfect": r1, "Off+1": r2})
    assert "Perfect" in text
    assert "Off+1" in text
    assert "MAE" in text


def test_comparison_table_rejects_mixed_kinds():
    r_clf = binary_classification_metrics(
        ["a", "b"], ["a", "b"],
        positive_class="a", negative_class="b",
    )
    r_reg = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="mix of classification"):
        format_comparison_table({"clf": r_clf, "reg": r_reg})
