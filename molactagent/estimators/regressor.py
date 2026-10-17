"""scikit-learn-compatible regressor wrapper around the CCR backend.

Experimental support: this backend remains under development and is not
evaluated in the accompanying manuscript. It is shared for community
exploration and collaboration; reliability and applicability require
further validation. See ``docs/experimental-regression.md``.

Mirrors the design of :class:`MolActAgentClassifier` but for the
regression path. Same posture: a thin projection layer over
:class:`molactagent.agent.MolActAgent` (which itself delegates to
:class:`molactagent.cross_classification.CrossClassificationRegressor`
when ``task='regression'``). No prediction logic lives here.

What's specific to regression:

- ``predict`` returns floats. Refused predictions become ``np.nan``,
  matching the convention sklearn's regressors use for sentinel
  values; callers can do ``mask = np.isfinite(y_pred)`` to filter.
- ``predict_with_interval`` exposes the empirical [min, max] of the
  picked references' actual targets — see ``docs/design.md`` §19.7.
  This is an empirical reference range, not a calibrated prediction interval.
- ``predict_with_confidence`` exposes a heuristic CCR confidence score
  (mean of the per-phase classification confidences, restricted to
  phases that produced a winner).
- ``score`` returns R² (the sklearn ``RegressorMixin`` contract) and
  excludes refused predictions before scoring so they don't pollute
  the metric. The number of excluded samples is available via
  ``self.last_refused_count_``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# scikit-learn is a hard dependency of the estimator wrappers; we
# inherit from ``BaseEstimator + RegressorMixin`` so sklearn 1.6+
# tag introspection (``__sklearn_tags__``) works and so
# ``is_regressor(estimator)`` returns True without any further wiring.
from sklearn.base import BaseEstimator, RegressorMixin

from ..agent import MolActAgent
from ..config import LLMConfig
from ..result import RegressionResult
from ._base import (
    _check_fitted,
    _coerce_X,
    _coerce_y_regression,
    _ensure_llm_client,
)


class MolActAgentRegressor(RegressorMixin, BaseEstimator):
    """sklearn-style regressor backed by the CCR engine.

    Experimental support: this backend remains under development and is
    not evaluated in the accompanying manuscript. It is shared for
    community exploration and collaboration; reliability and applicability
    require further validation. See ``docs/experimental-regression.md``.

    Parameters
    ----------
    llm_config, llm_client :
        See :class:`MolActAgentClassifier`.
    target_name :
        Optional human-readable name for the target column.
        Forwarded into the explanation prompt so reports refer to
        ``"body mass"`` rather than ``"target"``.
    feature_names, feature_descriptions, background :
        See :class:`MolActAgentClassifier`.
    k :
        Per-phase ``k`` inside CCR's inner classifiers.
    metric :
        Distance metric for contextual similarity.
    scaler :
        ``"standard"`` / ``"minmax"`` / ``"none"``.
    max_workers :
        Per-test-sample LLM-call concurrency.
    **agent_kwargs :
        Additional kwargs forwarded to :class:`MolActAgent` —
        notably the CCR-specific knobs ``ccr_aggregator``,
        ``ccr_k_per_class``, ``ccr_agent_k``, ``ccr_n_mode``,
        ``ccr_n_target_bins``, ``ccr_n_feature_clusters``.
    """

    def __init__(
        self,
        *,
        # ── In-code credentials (Python-first ergonomic path) ──
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        # ── Pre-built client / config (alternative paths) ──
        llm_config: LLMConfig | None = None,
        llm_client: Any | None = None,
        # ── Optional context ──
        target_name: str = "target",
        feature_names: list[str] | None = None,
        feature_descriptions: dict[str, str] | None = None,
        background: str | None = None,
        # ── Hyperparameters ──
        k: int = 5,
        metric: str = "euclidean",
        scaler: str = "standard",
        max_workers: int = 8,
        **agent_kwargs: Any,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.llm_config = llm_config
        self.llm_client = llm_client
        self.target_name = target_name
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions
        self.background = background
        self.k = k
        self.metric = metric
        self.scaler = scaler
        self.max_workers = max_workers
        # See classifier.py for the agent_kwargs__ stripping rationale.
        self.agent_kwargs = {
            (k[len("agent_kwargs__"):] if k.startswith("agent_kwargs__") else k): v
            for k, v in agent_kwargs.items()
        }

        self._agent: MolActAgent | None = None
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None
        self.target_range_: tuple[float, float] | None = None
        self.last_refused_count_: int = 0
        self._last_predict_X: np.ndarray | None = None
        self._last_predict_results: list[RegressionResult] | None = None

    # ── sklearn parameter introspection ────────────────────────

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "llm_config": self.llm_config,
            "llm_client": self.llm_client,
            "target_name": self.target_name,
            "feature_names": self.feature_names,
            "feature_descriptions": self.feature_descriptions,
            "background": self.background,
            "k": self.k,
            "metric": self.metric,
            "scaler": self.scaler,
            "max_workers": self.max_workers,
            **{f"agent_kwargs__{k}": v for k, v in self.agent_kwargs.items()},
        }

    def set_params(self, **params) -> "MolActAgentRegressor":
        for k, v in params.items():
            if k.startswith("agent_kwargs__"):
                self.agent_kwargs[k[len("agent_kwargs__"):]] = v
            else:
                setattr(self, k, v)
        # Reset fitted state.
        self._agent = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.target_range_ = None
        self.last_refused_count_ = 0
        self._last_predict_X = None
        self._last_predict_results = None
        return self

    # ── fit ────────────────────────────────────────────────────

    def fit(self, X, y, sample_weight=None) -> "MolActAgentRegressor":
        del sample_weight  # accepted for sklearn parity, intentionally unused

        # Auto-infer feature_names from a pandas DataFrame's columns
        # if the user didn't supply them — same Python-first
        # ergonomics as the classifier wrapper.
        inferred_feature_names = self.feature_names
        if inferred_feature_names is None:
            try:
                import pandas as _pd
                if isinstance(X, _pd.DataFrame):
                    inferred_feature_names = list(X.columns.astype(str))
            except ImportError:
                pass

        X = _coerce_X(X)
        y = _coerce_y_regression(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]}."
            )

        llm_client = _ensure_llm_client(
            self.llm_client, self.llm_config,
            api_key=self.api_key, base_url=self.base_url,
            model_name=self.model_name, temperature=self.temperature,
            max_tokens=self.max_tokens, timeout=self.timeout,
            max_retries=self.max_retries,
        )

        agent = MolActAgent(
            llm_client=llm_client,
            k=int(self.k),
            metric=self.metric,
            scaler=self.scaler,
            feature_names=(
                list(inferred_feature_names)
                if inferred_feature_names else None
            ),
            feature_descriptions=(
                dict(self.feature_descriptions)
                if self.feature_descriptions else None
            ),
            background=self.background,
            max_workers=int(self.max_workers),
            **self.agent_kwargs,
        )
        # Force regression mode for the same reason MolActAgentClassifier
        # forces classification: avoid surprising auto-detection edge
        # cases on integer-typed continuous targets.
        agent.fit(X, y, task="regression")

        self._agent = agent
        self.n_features_in_ = int(agent._X_ref.shape[1])
        self.feature_names_in_ = np.asarray(agent.feature_names)
        self.target_range_ = agent.target_range
        self._last_predict_X = None
        self._last_predict_results = None
        self.last_refused_count_ = 0
        return self

    # ── prediction ─────────────────────────────────────────────

    def _predict_full(self, X) -> list[RegressionResult]:
        _check_fitted(self)
        X_arr = _coerce_X(X)

        if self._last_predict_X is not None and self._last_predict_results is not None:
            same_obj = self._last_predict_X is X_arr
            same_val = (
                self._last_predict_X.shape == X_arr.shape
                and np.array_equal(self._last_predict_X, X_arr)
            )
            if same_obj or same_val:
                return self._last_predict_results

        raw_results = self._agent.predict(X_arr)
        wrapped = [RegressionResult(raw=r) for r in raw_results]

        # Cache the input + results so subsequent ``predict_with_*``
        # calls on the same X don't re-trigger the LLM round-trips.
        self._last_predict_X = np.array(X_arr, copy=True)
        self._last_predict_results = wrapped
        # Track refusals on the most recent batch — useful for
        # diagnostic reporting and for ``score`` (which masks them
        # before computing R²).
        self.last_refused_count_ = sum(1 for r in wrapped if r.is_refused)
        return wrapped

    def predict(self, X) -> np.ndarray:
        """Predict numeric targets for ``X``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Float predictions; refused predictions are ``np.nan``.
            Use ``mask = np.isfinite(y_pred)`` to drop refusals
            before plugging into a metric that does not tolerate NaN.
        """
        return np.array(
            [
                np.nan if r.is_refused else float(r.predicted_value)
                for r in self._predict_full(X)
            ],
            dtype=float,
        )

    def predict_with_interval(
        self, X,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(predictions, lo, hi)`` numpy arrays.

        ``lo[i]`` and ``hi[i]`` are the empirical min/max of the
        picked references' actual targets for sample ``i``. They are
        set to ``np.nan`` for refused predictions and for predictions
        where the engine could not pick any references at all.
        """
        results = self._predict_full(X)
        preds = np.array(
            [np.nan if r.is_refused else float(r.predicted_value)
             for r in results],
            dtype=float,
        )
        lo = np.zeros(len(results), dtype=float)
        hi = np.zeros(len(results), dtype=float)
        for i, r in enumerate(results):
            iv = r.interval
            lo[i] = float(iv[0]) if iv[0] is not None else np.nan
            hi[i] = float(iv[1]) if iv[1] is not None else np.nan
        return preds, lo, hi

    def predict_with_confidence(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(predictions, confidences)``.

        ``confidences[i] == 0.0`` for refused predictions (the engine
        sets the same field on the refusal record).
        """
        results = self._predict_full(X)
        preds = np.array(
            [np.nan if r.is_refused else float(r.predicted_value)
             for r in results],
            dtype=float,
        )
        confs = np.array([r.confidence for r in results], dtype=float)
        return preds, confs

    def predict_full(self, X) -> list[RegressionResult]:
        return list(self._predict_full(X))

    # ── score (sklearn contract) ──────────────────────────────

    def score(self, X, y) -> float:
        """Coefficient of determination R² — the sklearn
        ``RegressorMixin`` contract.

        Refused predictions (``np.nan``) are dropped before scoring
        so they do not pollute the metric. ``self.last_refused_count_``
        records how many were dropped on the most recent call.
        """
        from sklearn.metrics import r2_score

        y_true = _coerce_y_regression(y)
        y_pred = self.predict(X)
        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        if mask.sum() < 2:
            # R² needs at least 2 paired finite values to be meaningful.
            return 0.0
        return float(r2_score(y_true[mask], y_pred[mask]))

    # ── introspection ─────────────────────────────────────────

    @property
    def driver_attribution_(self) -> list[dict[str, Any]]:
        """Aggregate per-feature driver attribution across the LAST
        predicted batch. Empty before the first ``predict`` call.

        The attribution merges per-sample records and preserves the
        regression-flavoured ``dominant_direction`` field
        (``"increases"`` / ``"decreases"`` / ``"mixed"``) from the
        engine's per-feature Pearson correlation against the
        training target. Direction is taken from the first sample
        that mentions each feature — Pearson is a property of the
        training set, not of any individual test sample, so it is
        identical across samples in a batch.
        """
        if not self._last_predict_results:
            return []
        return _aggregate_regression_driver_attribution(
            [r.driver_attribution for r in self._last_predict_results]
        )

    @property
    def llm_stats_(self) -> dict[str, Any]:
        if self._agent is None:
            return {}
        client = self._agent.llm
        stats = getattr(client, "stats", None)
        if callable(stats):
            stats = stats()
        return dict(stats or {})

    def __repr__(self) -> str:
        if self._agent is None:
            return f"MolActAgentRegressor(k={self.k}, fitted=False)"
        tr = self.target_range_ or (None, None)
        tr_str = (
            f"[{tr[0]:.4g}, {tr[1]:.4g}]"
            if tr[0] is not None else "(no target_range)"
        )
        return (
            f"MolActAgentRegressor(k={self.k}, "
            f"target={self.target_name!r}, target_range={tr_str})"
        )


def _aggregate_regression_driver_attribution(
    per_sample: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge per-sample regression attributions into a cohort list.

    Same shape as the classification aggregator but preserves the
    ``dominant_direction`` field (which is a property of the training
    set, identical across samples).
    """
    by_feature: dict[str, dict[str, Any]] = {}
    for sample_attribution in per_sample:
        for entry in sample_attribution:
            fn = entry.get("feature")
            if not fn:
                continue
            slot = by_feature.setdefault(
                fn,
                {
                    "feature": fn,
                    "description": entry.get("description", ""),
                    "n_cited": 0,
                    "abs_strength": 0.0,
                    "weighted_score": 0.0,
                    "dominant_direction": entry.get(
                        "dominant_direction", "mixed"
                    ),
                    "rationales": [],
                },
            )
            slot["n_cited"] += int(entry.get("n_cited", 0) or 0)
            slot["abs_strength"] += float(entry.get("abs_strength", 0.0) or 0.0)
            slot["weighted_score"] += float(
                entry.get("weighted_score", 0.0) or 0.0
            )
            for r in (entry.get("rationales") or [])[:1]:
                if r and len(slot["rationales"]) < 3:
                    slot["rationales"].append(r)

    out = list(by_feature.values())
    for s in out:
        s["abs_strength"] = round(s["abs_strength"], 3)
        s["weighted_score"] = round(s["weighted_score"], 3)
    out.sort(
        key=lambda s: (s["abs_strength"], s["n_cited"]),
        reverse=True,
    )
    return out
