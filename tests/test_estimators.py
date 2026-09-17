"""Parity tests for the sklearn-style estimator wrappers.

These tests assert that the new wrappers (:class:`MolActAgentClassifier`,
:class:`MolActAgentRegressor`) produce **bit-identical** outputs
relative to the existing engine APIs (:meth:`MolActAgent.predict_one`
and the high-level ``run_classification`` / ``run_regression``).

The motivation is the user's "no functional or performance
degradation" constraint. The wrappers are pure projections on top of
the engine; if a parity test ever fails, that failure is the
canary that the wrapper has acquired prediction logic of its own,
which violates the design.

Every test runs against ``conftest.FakeLLM`` so it is fully
deterministic — no API key required, no network access, runs in
sub-second time alongside the rest of the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from molactagent import (
    MolActAgent,
    MolActAgentClassifier,
    MolActAgentRegressor,
    ClassificationResult,
    RegressionResult,
)


# ── classification parity ─────────────────────────────────────────


def test_classifier_matches_molactagent_labels(
    multiclass_reference, multiclass_test, fake_llm,
):
    """``MolActAgentClassifier.predict`` returns the same labels as
    ``MolActAgent.predict_one`` for every test row.

    This is the baseline parity assertion: if the wrapper diverges,
    every other estimator-related promise is suspect.
    """
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_train = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)

    # Engine-direct labels.
    engine = MolActAgent(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    )
    engine.fit(X_train, y_train, task="classification")
    engine_labels = [r["final_label"] for r in engine.predict(X_test)]

    # Wrapper labels (via a fresh FakeLLM so call counts are
    # comparable; the FakeLLM is stateless w.r.t. determinism).
    wrapper = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    )
    wrapper.fit(X_train, y_train)
    wrapper_labels = wrapper.predict(X_test).tolist()

    assert engine_labels == wrapper_labels, (
        "Estimator wrapper diverged from engine on labels; this "
        "violates the no-functional-regression constraint."
    )


def test_classifier_predict_proba_matches_vote_summary(
    multiclass_reference, multiclass_test, fake_llm,
):
    """``predict_proba`` rows must equal the normalised weighted
    votes from each sample's ``vote_summary``.

    This is the contract described in the wrapper docstring; if it
    breaks, downstream consumers (sklearn metrics, calibration
    diagnostics) will silently produce wrong numbers.
    """
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_train = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)

    clf = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    # Shape contract.
    assert proba.shape == (X_test.shape[0], len(clf.classes_))
    # Each row sums to 1 within float epsilon (or to 0 — but with
    # the FakeLLM's deterministic responses every test sample gets
    # at least one decisive vote, so the all-zero case shouldn't
    # arise here).
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    # Recompute the expected values directly from the engine's
    # raw result dicts and compare.
    full_results = clf.predict_full(X_test)
    classes = list(clf.classes_)
    for i, res in enumerate(full_results):
        per_class = res.vote_summary.get("per_class", {})
        weights = {
            cls: float((per_class.get(cls) or {}).get("weighted", 0.0) or 0.0)
            for cls in res.candidate_classes
        }
        total = sum(weights.values())
        if total > 0:
            expected = {
                cls: weights.get(cls, 0.0) / total
                for cls in classes
            }
        else:
            # Uniform fallback over the candidate classes — see
            # ``ClassificationResult.proba`` docstring.
            uniform = 1.0 / len(res.candidate_classes)
            expected = {
                cls: (
                    uniform if cls in res.candidate_classes else 0.0
                )
                for cls in classes
            }
        for j, cls in enumerate(classes):
            assert proba[i, j] == pytest.approx(expected[cls], abs=1e-9), (
                f"row {i} class {cls!r}: wrapper says {proba[i, j]}, "
                f"engine says {expected[cls]}"
            )


def test_classifier_predict_with_confidence_returns_engine_field(
    multiclass_reference, multiclass_test, fake_llm,
):
    """``predict_with_confidence`` must return the engine's
    ``confidence`` field verbatim — NOT ``proba.max()``."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_train = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)

    clf = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)

    labels, confs = clf.predict_with_confidence(X_test)
    full = clf.predict_full(X_test)

    # Labels still match.
    assert labels.tolist() == [r.label for r in full]
    # Confidences are the engine's calibrated number, not proba.max.
    np.testing.assert_allclose(confs, [r.confidence for r in full])


def test_classifier_score_matches_sklearn_accuracy(
    multiclass_reference, multiclass_test, fake_llm,
):
    """``score(X, y)`` returns mean accuracy — sklearn contract."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_train = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)
    y_test = multiclass_test["bucket"].to_numpy()

    clf = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)
    expected = float((clf.predict(X_test) == y_test).mean())
    assert clf.score(X_test, y_test) == pytest.approx(expected, abs=1e-12)


def test_classifier_predict_full_returns_typed_wrappers(
    multiclass_reference, multiclass_test, fake_llm,
):
    """``predict_full`` returns :class:`ClassificationResult` instances
    that round-trip via ``to_dict`` to the engine's raw schema."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_train = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)

    clf = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)
    full = clf.predict_full(X_test)
    assert all(isinstance(r, ClassificationResult) for r in full)
    # Round-trip preserves the engine schema.
    for r in full:
        d = r.to_dict()
        assert "final_label" in d
        assert "confidence" in d
        assert "vote_summary" in d
        assert "candidate_classes" in d


def test_classifier_caches_within_batch(
    multiclass_reference, multiclass_test, fake_llm,
):
    """Repeated ``predict``-family calls on the same X must NOT
    re-trigger LLM calls — keeps token spend honest."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_train = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)

    clf = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)

    n_calls_after_predict = fake_llm.call_count
    clf.predict(X_test)
    n_after_first = fake_llm.call_count
    assert n_after_first > n_calls_after_predict

    # Second call should hit the cache.
    clf.predict_proba(X_test)
    clf.predict_with_confidence(X_test)
    clf.predict_full(X_test)
    assert fake_llm.call_count == n_after_first, (
        "Wrapper made additional LLM calls on a cached batch — this "
        "wastes tokens and breaks the no-performance-regression "
        "constraint."
    )


# ── regression parity ─────────────────────────────────────────────


def test_regressor_matches_engine_values(
    regression_reference, regression_test, fake_llm,
):
    """``MolActAgentRegressor.predict`` returns the same float values
    as the engine for every test row (NaN for refusals)."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = regression_reference[feature_cols].to_numpy(dtype=float)
    y_train = regression_reference["target"].to_numpy(dtype=float)
    X_test = regression_test[feature_cols].to_numpy(dtype=float)

    engine = MolActAgent(
        llm_client=fake_llm,
        k=5,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train, task="regression")
    engine_preds = [r.get("predicted_value") for r in engine.predict(X_test)]

    wrapper = MolActAgentRegressor(
        llm_client=fake_llm,
        k=5,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)
    wrapper_preds = wrapper.predict(X_test).tolist()

    for i, (e, w) in enumerate(zip(engine_preds, wrapper_preds)):
        if e is None:
            assert np.isnan(w), (
                f"row {i}: engine refused but wrapper produced {w}"
            )
        else:
            assert w == pytest.approx(float(e), abs=1e-12), (
                f"row {i}: engine={e}, wrapper={w}"
            )


def test_regressor_predict_with_interval(
    regression_reference, regression_test, fake_llm,
):
    """``predict_with_interval`` returns the engine's empirical
    [min, max] for each prediction."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = regression_reference[feature_cols].to_numpy(dtype=float)
    y_train = regression_reference["target"].to_numpy(dtype=float)
    X_test = regression_test[feature_cols].to_numpy(dtype=float)

    reg = MolActAgentRegressor(
        llm_client=fake_llm,
        k=5,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)
    preds, lo, hi = reg.predict_with_interval(X_test)
    full = reg.predict_full(X_test)

    for i, r in enumerate(full):
        if r.is_refused:
            assert np.isnan(preds[i])
            assert np.isnan(lo[i])
            assert np.isnan(hi[i])
        else:
            assert preds[i] == pytest.approx(float(r.predicted_value), abs=1e-12)
            iv = r.interval
            assert lo[i] == pytest.approx(float(iv[0]), abs=1e-12)
            assert hi[i] == pytest.approx(float(iv[1]), abs=1e-12)


def test_regressor_score_handles_refusals(
    regression_reference, regression_test, fake_llm,
):
    """``score`` masks refusals before computing R² — refusals
    should not pollute the metric."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = regression_reference[feature_cols].to_numpy(dtype=float)
    y_train = regression_reference["target"].to_numpy(dtype=float)
    X_test = regression_test[feature_cols].to_numpy(dtype=float)
    y_test = regression_test["target"].to_numpy(dtype=float)

    reg = MolActAgentRegressor(
        llm_client=fake_llm,
        k=5,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    assert isinstance(score, float)
    # The FakeLLM produces deterministic, finite predictions on this
    # fixture, so R² should be a finite number (not NaN).
    assert np.isfinite(score)


def test_regressor_predict_full_returns_typed_wrappers(
    regression_reference, regression_test, fake_llm,
):
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_train = regression_reference[feature_cols].to_numpy(dtype=float)
    y_train = regression_reference["target"].to_numpy(dtype=float)
    X_test = regression_test[feature_cols].to_numpy(dtype=float)

    reg = MolActAgentRegressor(
        llm_client=fake_llm,
        k=5,
        feature_names=list(feature_cols),
        max_workers=2,
    ).fit(X_train, y_train)
    full = reg.predict_full(X_test)
    assert all(isinstance(r, RegressionResult) for r in full)
    for r in full:
        d = r.to_dict()
        assert d.get("task") == "regression"
        assert "predicted_value" in d
        assert "interval" in d
        assert "confidence" in d


# ── interop ──────────────────────────────────────────────────────


def test_classifier_get_set_params_round_trip(fake_llm):
    """``get_params`` / ``set_params`` mirror sklearn's contract."""
    clf = MolActAgentClassifier(
        llm_client=fake_llm,
        k=3,
        metric="euclidean",
        scaler="standard",
        max_workers=4,
    )
    params = clf.get_params()
    assert params["k"] == 3
    assert params["metric"] == "euclidean"

    clf.set_params(k=5, metric="manhattan")
    assert clf.k == 5
    assert clf.metric == "manhattan"


def test_not_fitted_error_raised(fake_llm):
    from molactagent.estimators._base import _NotFittedError

    clf = MolActAgentClassifier(llm_client=fake_llm)
    with pytest.raises(_NotFittedError):
        clf.predict(np.zeros((1, 4)))


def test_wrap_result_factory_dispatches_on_task():
    from molactagent import wrap_result

    cls_dict = {"task": "classification", "final_label": "active"}
    reg_dict = {"task": "regression", "predicted_value": 1.5}
    legacy = {"final_label": "x"}  # legacy dict without task key

    assert isinstance(wrap_result(cls_dict), ClassificationResult)
    assert isinstance(wrap_result(reg_dict), RegressionResult)
    # Legacy dicts default to classification.
    assert isinstance(wrap_result(legacy), ClassificationResult)
