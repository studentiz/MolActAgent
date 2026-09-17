"""End-to-end tests for the regression code path.

These tests exercise the full pipeline (validate -> agent -> reports)
on small synthetic datasets using the regression FakeLLM. They cover:

- Auto-detection of regression from the label column dtype.
- The per-sample numeric prediction + interval flow.
- Reference info / driver attribution shapes.
- The refuse-to-predict policy (all tuples flag extrapolation AND
  predictions disagree widely).
- Pipeline-level summary CSV / JSON / interpretability artefacts.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from molactagent import MolActAgent, run_prediction, run_regression
from molactagent.io import detect_task, prepare_for_agent, validate_inputs


# ── Detection / preparation ─────────────────────────────────────────


class TestDetectAndPrepareRegression:
    def test_detect_task_continuous_target(self, regression_reference):
        task = detect_task(regression_reference["target"])
        assert task == "regression"

    def test_validate_inputs_marks_regression(
        self, regression_reference, regression_test,
    ):
        result = validate_inputs(
            regression_reference, regression_test, "target", k=3,
        )
        assert result.is_valid
        assert result.summary["task"] == "regression"
        target_min, target_max = result.summary["target_range"]
        assert target_min < target_max
        assert "n_distinct_targets" in result.summary
        assert result.summary["n_distinct_targets"] >= 3

    def test_prepare_for_agent_keeps_float_target(
        self, regression_reference, regression_test,
    ):
        prep = prepare_for_agent(
            regression_reference, regression_test, "target",
        )
        assert prep.task == "regression"
        assert prep.y_ref.dtype == float
        assert prep.target_range is not None
        assert prep.class_labels == []

    def test_validate_rejects_constant_target(self):
        ref = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],  # constant
        })
        # Constant target with 1 distinct value triggers the
        # classification branch (≤10 distinct integers → classify);
        # but here the target is float, so the regression branch
        # rejects it (only 1 distinct value → < 3 distinct).
        # Either branch surfaces the problem as a hard error.
        test = pd.DataFrame({
            "f1": [1.5, 2.5],
            "f2": [1.5, 2.5],
        })
        result = validate_inputs(ref, test, "target", k=2)
        assert not result.is_valid
        assert any(
            "distinct" in e.lower() or "one class" in e.lower()
            for e in result.errors
        )


# ── End-to-end regression ──────────────────────────────────────────


class TestRegressionPipelineEndToEnd:
    def test_run_regression_produces_numeric_predictions(
        self,
        fake_llm,
        regression_reference,
        regression_test,
        tmp_workspace,
    ):
        out = run_regression(
            reference=regression_reference,
            test=regression_test,
            target_column="target",
            k=4,
            llm_client=fake_llm,
            output_dir=tmp_workspace,
            per_sample_reports=True,
            write_summary_json=True,
        )
        assert out.validation.is_valid
        assert out.validation.summary["task"] == "regression"
        assert out.n_succeeded == len(regression_test)
        for o in out.outcomes:
            assert o.task == "regression"
            assert o.predicted_value is not None
            assert isinstance(o.predicted_value, float)
            assert np.isfinite(o.predicted_value)
            assert o.interval is not None
            lo, hi = o.interval
            assert lo <= o.predicted_value <= hi
            # FakeLLM echoes the witness baseline (~50ish), which
            # falls inside the observed target range; it must not
            # ever flag extrapolation in this synthetic setup.
            raw = o.raw_result
            assert raw["task"] == "regression"
            assert "predicted_value" in raw
            assert "interval" in raw
            assert "driver_attribution" in raw

    def test_run_regression_summary_csv_has_regression_columns(
        self,
        fake_llm,
        regression_reference,
        regression_test,
        tmp_workspace,
    ):
        run_regression(
            reference=regression_reference,
            test=regression_test,
            target_column="target",
            k=3,
            llm_client=fake_llm,
            output_dir=tmp_workspace,
        )
        csv_path = tmp_workspace / "summary.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        for col in ("task", "predicted_value", "interval_lo", "interval_hi"):
            assert col in df.columns
        assert (df["task"] == "regression").all()
        assert df["predicted_value"].notna().all()

    def test_run_regression_summary_json_carries_target_range(
        self,
        fake_llm,
        regression_reference,
        regression_test,
        tmp_workspace,
    ):
        run_regression(
            reference=regression_reference,
            test=regression_test,
            target_column="target",
            k=3,
            llm_client=fake_llm,
            output_dir=tmp_workspace,
            write_summary_json=True,
        )
        with open(tmp_workspace / "summary.json") as f:
            blob = json.load(f)
        assert blob["task"] == "regression"
        assert blob["target_range"] is not None
        assert blob["target_range"][0] < blob["target_range"][1]


# ── Type-mismatch guards on the wrapper functions ──────────────────


class TestTaskOverrideMismatch:
    def test_run_regression_raises_on_classification_data(
        self,
        fake_llm,
        simple_reference,
        simple_test,
        tmp_workspace,
    ):
        with pytest.raises(ValueError, match="task='regression'"):
            run_regression(
                reference=simple_reference,
                test=simple_test,
                target_column="activity",
                k=2,
                llm_client=fake_llm,
                output_dir=tmp_workspace,
                raise_on_validation_error=True,
            )

    def test_run_classification_raises_on_regression_data(
        self,
        fake_llm,
        regression_reference,
        regression_test,
        tmp_workspace,
    ):
        from molactagent import run_classification
        with pytest.raises(ValueError, match="task='classification'"):
            run_classification(
                reference=regression_reference,
                test=regression_test,
                label_column="target",
                k=3,
                llm_client=fake_llm,
                output_dir=tmp_workspace,
                raise_on_validation_error=True,
            )


# ── Direct MolActAgent regression API ───────────────────────────────


class TestMolActAgentRegressionAPI:
    def test_agent_fit_detects_regression(
        self, fake_llm, regression_reference,
    ):
        X = regression_reference[["alpha", "beta", "gamma", "delta"]].to_numpy()
        y = regression_reference["target"].to_numpy()
        ag = MolActAgent(
            llm_client=fake_llm,
            k=3,
            max_workers=1,
            feature_names=["alpha", "beta", "gamma", "delta"],
        )
        ag.fit(X, y)
        assert ag.task == "regression"
        assert ag.target_range is not None
        assert ag.class_labels == []

    def test_agent_predict_one_returns_numeric_prediction(
        self,
        fake_llm,
        regression_reference,
    ):
        X = regression_reference[["alpha", "beta", "gamma", "delta"]].to_numpy()
        y = regression_reference["target"].to_numpy()
        ag = MolActAgent(
            llm_client=fake_llm,
            k=3,
            max_workers=1,
            feature_names=["alpha", "beta", "gamma", "delta"],
            regression_n_tuples=2,
        )
        ag.fit(X, y)
        result = ag.predict_one(X[0] + 0.5, test_id=0)
        assert result["task"] == "regression"
        assert result["predicted_value"] is not None
        assert isinstance(result["predicted_value"], float)
        lo, hi = result["interval"]
        assert lo <= result["predicted_value"] <= hi
        assert "driver_attribution" in result
        # FakeLLM cites the discriminative features so attribution
        # should be non-empty for any non-degenerate test sample.
        assert len(result["driver_attribution"]) >= 1


# ── CCR backend internals (replaces the legacy bootstrap-tuple tests) ─


class TestCCRBackendStructure:
    """Spot-check that MolActAgent's regression path is now backed by
    the Cross-Classification Regressor and that its outputs carry the
    fields downstream code (pipeline + report) expects."""

    def _fit_agent(self, fake_llm, regression_reference):
        from molactagent import MolActAgent

        X = regression_reference[
            ["alpha", "beta", "gamma", "delta"]
        ].to_numpy()
        y = regression_reference["target"].to_numpy()
        ag = MolActAgent(
            llm_client=fake_llm,
            k=3, max_workers=1,
            feature_names=["alpha", "beta", "gamma", "delta"],
        )
        ag.fit(X, y, task="regression")
        return ag, X, y

    def test_fit_constructs_ccr_backend(
        self, fake_llm, regression_reference,
    ):
        from molactagent import CrossClassificationRegressor

        ag, _, _ = self._fit_agent(fake_llm, regression_reference)
        assert ag.task == "regression"
        assert ag._regression_backend is not None
        assert isinstance(ag._regression_backend, CrossClassificationRegressor)
        # CCR fitted two inner MolActAgent classifiers.
        assert ag._regression_backend.agent_target is not None
        assert ag._regression_backend.agent_feature is not None
        # N and M default to floor(sqrt(n)) — for n=24 reference rows
        # that's 4; clamped to [2, min(10, n-1)].
        assert 2 <= ag._regression_backend._n_bins_used <= 10
        assert 2 <= ag._regression_backend._n_clusters_used <= 10

    def test_predict_one_returns_ccr_shape(
        self, fake_llm, regression_reference,
    ):
        ag, X, _ = self._fit_agent(fake_llm, regression_reference)
        result = ag.predict_one(X[0] + 0.1, test_id=0)

        # Schema fields the downstream pipeline + report code reads.
        assert result["task"] == "regression"
        assert result["method"] == "CCR"
        assert isinstance(result["predicted_value"], float)
        assert np.isfinite(result["predicted_value"])
        lo, hi = result["interval"]
        assert lo <= result["predicted_value"] <= hi
        assert 0.0 <= result["confidence"] <= 1.0

        # CCR-specific fields surfaced for diagnostics / interpretability.
        for key in (
            "phase_target", "phase_feature",
            "selected_indices", "selected_targets", "selected_distances",
            "aggregator", "phase_views_agree",
        ):
            assert key in result, f"missing CCR field: {key!r}"
        # Two-phase mechanism: Phase 1 is target-bin classification,
        # Phase 2 is feature-cluster classification. Both must report
        # a winning class (FakeLLM never refuses on these synthetic
        # inputs).
        assert result["phase_target"]["predicted_class"] is not None
        assert result["phase_feature"]["predicted_class"] is not None
        # The selected references' actual targets are what get
        # aggregated; the empirical interval brackets them.
        assert result["selected_targets"]
        assert lo == min(result["selected_targets"])
        assert hi == max(result["selected_targets"])

        # Backward-compatible fields for the legacy regression
        # explanation prompt.
        assert "tuple_results" in result
        assert "driver_attribution" in result
        assert "confidence_basis" in result
        assert "reference_info" in result
        assert "test_feature_values" in result
        assert "extrapolation_flagged" in result
        # Phase 1 + Phase 2 each contribute a synthesized
        # ``tuple_results`` row so the explanation report has per-
        # phase reasoning records to render.
        assert len(result["tuple_results"]) >= 1


class TestCCRConfWeightingDefault:
    """The deployed default aggregator is per-phase confidence-
    weighted (the BSLT-ablation winner). Verify the dispatch routes
    correctly even on small synthetic data."""

    def test_default_aggregator_is_conf(
        self, fake_llm, regression_reference,
    ):
        from molactagent import MolActAgent

        X = regression_reference[
            ["alpha", "beta", "gamma", "delta"]
        ].to_numpy()
        y = regression_reference["target"].to_numpy()
        ag = MolActAgent(
            llm_client=fake_llm,
            k=3, max_workers=1,
            feature_names=["alpha", "beta", "gamma", "delta"],
        )
        ag.fit(X, y, task="regression")
        assert ag.ccr_aggregator == "conf"
        # When the backend is invoked, the aggregator is propagated
        # to the result.
        result = ag.predict_one(X[0] + 0.1, test_id=0)
        assert result["aggregator"] == "conf"


# ── Refuse-to-predict policy (CCR refusal mechanism) ───────────────


class _AlwaysInconclusiveFakeLLM:
    """FakeLLM that emits classification verdicts which the agent
    aggregator collapses to ``"Inconclusive"`` AND zero per-class
    weighted votes. This forces ``_phase_winner_class`` to return
    ``None`` for every CCR phase, exercising the refusal path.
    """

    throttle_count = 0

    def __init__(self):
        self.calls = 0
        self._call_log: list[dict] = []

    @property
    def call_log(self):
        return list(self._call_log)

    @property
    def stats(self):
        return {
            "calls": self.calls,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_time_s": 0.0,
            "avg_time_s": 0.0,
        }

    def chat(self, messages, tag="", max_tokens=None, temperature=None):
        self.calls += 1
        self._call_log.append({"call_id": self.calls, "tag": tag})
        if "EXPLAIN" in tag.upper():
            return "## Executive Summary\nRefused.\n"
        # Tuple-classification prompts: emit ``"Unknown"`` with
        # zero per-class strength so the aggregator collapses to
        # Inconclusive AND every per-class weighted score is 0.
        return json.dumps({
            "label": "Unknown",
            "confidence": 0.0,
            "key_drivers": [
                {
                    "feature": "alpha", "favored_class": "Unknown",
                    "strength": 0.0, "test_value": 0.0,
                    "favored_value": 0.0,
                    "why": "synthetic refusal",
                },
                {
                    "feature": "beta", "favored_class": "Unknown",
                    "strength": 0.0, "test_value": 0.0,
                    "favored_value": 0.0,
                    "why": "synthetic refusal",
                },
            ],
            "supporting_features": [],
            "opposing_features": [],
            "summary": "Synthetic refusal verdict.",
            "counter_evidence": "none",
            "uncertainty": "Synthetic FakeLLM never produces a winner.",
        })

    def ping(self, *, timeout: float = 30.0) -> dict:
        return {"ok": True, "elapsed_s": 0.0, "response": "ping"}


def test_ccr_refuses_when_both_phases_produce_no_winner(
    fake_llm, regression_reference,
):
    """When neither Phase 1 nor Phase 2 nominates a usable class,
    CCR must refuse to predict rather than fabricate a number from
    an empty selection set.

    MolActAgent's classification path has a deterministic fallback
    that picks a winning class even on adversarial LLM output, so
    triggering the CCR refusal end-to-end requires monkeypatching
    each inner agent's ``predict_one`` to emit an Inconclusive
    verdict with zero per-class weighted votes — the only state in
    which ``_phase_winner_class`` returns ``None``.
    """
    from molactagent import MolActAgent

    X = regression_reference[
        ["alpha", "beta", "gamma", "delta"]
    ].to_numpy()
    y = regression_reference["target"].to_numpy()
    ag = MolActAgent(
        llm_client=fake_llm,
        k=3, max_workers=1,
        feature_names=["alpha", "beta", "gamma", "delta"],
    )
    ag.fit(X, y, task="regression")

    inconclusive_result = {
        "test_index": 0,
        "task": "classification",
        "final_label": "Inconclusive",
        "confidence": 0.0,
        "vote_summary": {"per_class": {}, "total": 0},
        "candidate_classes": [],
        "driver_attribution": [],
        "reference_info": {},
        "tuple_results": [],
    }
    ag._regression_backend.agent_target.predict_one = (
        lambda x, test_id=None: dict(inconclusive_result)
    )
    ag._regression_backend.agent_feature.predict_one = (
        lambda x, test_id=None: dict(inconclusive_result)
    )

    result = ag.predict_one(X[0] + 0.5, test_id=0)
    assert result["task"] == "regression"
    assert result["method"] == "CCR"
    assert result["predicted_value"] is None, (
        "expected refuse-to-predict when both phases produce no "
        "usable winning class"
    )
    assert result["interval"] == (None, None)
    reason = (result.get("reason") or "").lower()
    assert "refus" in reason or "both phases" in reason
    assert result["selected_targets"] == []


# ``_AlwaysInconclusiveFakeLLM`` was the original (failed) attempt
# to drive a CCR refusal through realistic LLM output. MolActAgent's
# classification fallback to a deterministic baseline absorbed the
# Unknown verdicts and picked a winning class anyway. Kept as a
# documented stub for anyone reading the test history.
_AlwaysInconclusiveFakeLLM  # noqa: B018  - intentional reference

