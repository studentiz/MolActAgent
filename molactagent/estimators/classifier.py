"""scikit-learn-compatible classifier wrapper around :class:`MolActAgent`.

Why this exists: the engine's native ``MolActAgent`` returns a list of
rich result dicts. Most of the broader ML ecosystem — sklearn
pipelines, ``cross_val_score``, GridSearchCV, the calibration tools
the user wants — speaks "fit / predict / predict_proba / score" with
plain numpy arrays. This wrapper bridges those two worlds without
modifying either.

Architectural posture (per ``docs/design.md`` §29.2):

- The wrapper is a **thin** projection. It owns no prediction logic.
  Every numeric output ultimately comes from
  :meth:`MolActAgent.predict_one`.
- ``predict_proba`` derives its probabilities from the engine's
  ``vote_summary`` weighted votes (see ``docs/design.md`` §18.1).
  This is the same continuous score the LOOCV evaluation script
  has used since day one — a documented projection of the engine's
  output, not a new computation.
- ``predict_with_confidence`` exposes the calibrated **aggregated
  confidence** (engine field ``confidence``) as a separate axis from
  ``predict_proba``. The two answer different questions: ``proba``
  asks "how is the evidence distributed across classes?"; ``confidence``
  asks "how trustworthy is the verdict overall?". They are deliberately
  not the same number — see the user-facing Python API tutorial for
  the rationale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# scikit-learn is a hard dependency of the estimator wrappers (see
# pyproject.toml). We must inherit from ``BaseEstimator`` AND the
# right Mixin so sklearn 1.6+'s ``get_tags()`` introspection works
# — without these, ``cross_val_score`` / ``GridSearchCV`` fail
# immediately with an AttributeError on ``__sklearn_tags__``. The
# inheritance also makes ``is_classifier(estimator)`` return True
# without any further work.
from sklearn.base import BaseEstimator, ClassifierMixin

from ..agent import MolActAgent
from ..config import LLMConfig
from ..result import ClassificationResult
from ._base import (
    _check_fitted,
    _coerce_X,
    _coerce_y_classification,
    _ensure_llm_client,
)


class MolActAgentClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-style multi-class classifier backed by :class:`MolActAgent`.

    The classifier accepts LLM credentials in **four** equivalent
    ways. Pick whichever suits the moment:

    1. **In-code (recommended for notebooks and ad-hoc Python work).**
       Pass ``api_key`` / ``base_url`` / ``model_name`` directly to
       the constructor::

           clf = ca.MolActAgentClassifier(
               api_key="sk-...",
               base_url="https://api.deepseek.com",
               model_name="deepseek-chat",
           )

    2. **Pre-built client (for tests and shared sessions).**
       Pass ``llm_client=`` an :class:`molactagent.LLMClient` you
       built yourself.

    3. **Pre-built config (for batch jobs).** Pass
       ``llm_config=ca.load_llm_config('llm_config.jsonc')``.

    4. **Auto-discovery (for projects with a checked-in config).**
       Don't pass any of the above — the wrapper walks up from cwd
       looking for ``llm_config.jsonc`` / ``LLM_API.txt`` / honours
       ``MOLACTAGENT_*`` env vars.

    Parameters
    ----------
    api_key, base_url, model_name :
        Direct in-code credentials. The most ergonomic path for
        notebooks. Combine with the optional ``temperature`` /
        ``max_tokens`` / ``timeout`` / ``max_retries`` kwargs to
        tune client behaviour.
    llm_config :
        Pre-resolved :class:`molactagent.config.LLMConfig`.
    llm_client :
        A pre-built LLM client (or test fake). If supplied, all
        other credential kwargs are ignored.
    feature_names :
        Optional human-readable feature names. If omitted and ``X``
        is a pandas DataFrame, the column names are used; otherwise
        a generic ``feature_0 … feature_{n}`` series is generated.
    feature_descriptions :
        Optional ``{feature_name: description}`` dict — used by the
        prompt builder to weight which features matter most.
        Strongly recommended for any non-toy dataset, but the engine
        works without it.
    background :
        Optional free-text domain context handed to the LLM in the
        system message. ``"@path/to/file.txt"`` is treated as a
        path. Defaults to no extra background.
    k :
        Per-class neighbours retained in the witness layer.
    metric :
        Distance metric for the contextual-similarity computation.
    scaler :
        ``"standard"`` / ``"minmax"`` / ``"none"``.
    max_workers :
        Per-test-sample LLM-call concurrency (across the M+1 tuples).
    **agent_kwargs :
        Any additional keyword argument supported by
        :class:`molactagent.agent.MolActAgent` is forwarded
        verbatim — keeps the wrapper future-proof.

    Attributes (after :meth:`fit`)
    ------------------------------
    classes_ :
        Sorted ndarray of unique class labels seen at fit time.
    n_features_in_ :
        Number of features after the engine's variance filter.
    feature_names_in_ :
        Feature names (after variance filter).
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
        feature_names: list[str] | None = None,
        feature_descriptions: dict[str, str] | None = None,
        background: str | None = None,
        # ── Hyperparameters ──
        k: int = 3,
        metric: str = "euclidean",
        scaler: str = "standard",
        max_workers: int = 8,
        **agent_kwargs: Any,
    ):
        # In-code credential fields.
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        # Pre-built credential fields (back-compat).
        self.llm_config = llm_config
        self.llm_client = llm_client
        # Context.
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions
        self.background = background
        # Hyperparameters.
        self.k = k
        self.metric = metric
        self.scaler = scaler
        self.max_workers = max_workers
        # ``sklearn.clone()`` round-trips params through
        # ``get_params(deep=False)`` → ``type(est)(**params)``. Our
        # ``get_params`` exposes engine-specific kwargs with an
        # ``agent_kwargs__`` prefix (the GridSearchCV convention).
        # Strip that prefix here so clone() reconstructs the
        # original ``agent_kwargs`` dict instead of nesting it.
        self.agent_kwargs = {
            (k[len("agent_kwargs__"):] if k.startswith("agent_kwargs__") else k): v
            for k, v in agent_kwargs.items()
        }

        # Lazy state — populated by ``fit``.
        self._agent: MolActAgent | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None
        # Cache of the most recent batch of full result wrappers, so
        # repeated ``predict``/``predict_proba`` calls within the same
        # batch don't re-call the LLM.
        self._last_predict_X: np.ndarray | None = None
        self._last_predict_results: list[ClassificationResult] | None = None

    # ── sklearn-style hyperparameter introspection ─────────────

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Mirror sklearn's ``get_params`` so cross-validation /
        grid-search / clone helpers can introspect this estimator.
        """
        # ``deep`` is accepted for API parity — we have no nested
        # estimators with their own ``get_params``.
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
            "feature_names": self.feature_names,
            "feature_descriptions": self.feature_descriptions,
            "background": self.background,
            "k": self.k,
            "metric": self.metric,
            "scaler": self.scaler,
            "max_workers": self.max_workers,
            **{f"agent_kwargs__{k}": v for k, v in self.agent_kwargs.items()},
        }

    def set_params(self, **params) -> "MolActAgentClassifier":
        # Strip the ``agent_kwargs__`` prefix when present so that
        # GridSearchCV-style flat parameter dicts work.
        for k, v in params.items():
            if k.startswith("agent_kwargs__"):
                self.agent_kwargs[k[len("agent_kwargs__"):]] = v
            else:
                setattr(self, k, v)
        # Invalidate any cached fit/predict state since the
        # hyperparameters changed.
        self._agent = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._last_predict_X = None
        self._last_predict_results = None
        return self

    # ── fit ────────────────────────────────────────────────────

    def fit(self, X, y, sample_weight=None) -> "MolActAgentClassifier":
        """Fit the estimator on a labelled reference set.

        ``sample_weight`` is accepted for API parity with sklearn but
        is **ignored** — the engine doesn't yet support per-sample
        weighting (every reference is weighted equally during
        contextual neighbour selection). We accept it without raising
        so this estimator can drop into existing sklearn workflows
        that always pass ``sample_weight``; we emit no warning so the
        common case stays quiet.
        """
        del sample_weight  # accepted for sklearn parity, intentionally unused

        # Auto-infer feature_names from a pandas DataFrame's columns
        # if the user didn't supply them — strict opt-out, not opt-in,
        # because passing column names manually for every estimator
        # construction is exactly the kind of friction the user
        # complained about.
        inferred_feature_names = self.feature_names
        if inferred_feature_names is None:
            try:
                import pandas as _pd
                if isinstance(X, _pd.DataFrame):
                    inferred_feature_names = list(X.columns.astype(str))
            except ImportError:
                pass

        X = _coerce_X(X)
        y = _coerce_y_classification(y)

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

        # Build the engine. We forward every classifier-relevant
        # hyperparameter; the engine's own constructor silently
        # absorbs unknown legacy kwargs, so future-only kwargs in
        # ``agent_kwargs`` won't break older engine releases.
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
        # Force classification mode — the engine's auto-detection
        # would correctly pick this for string labels too, but being
        # explicit defends against a caller passing integer labels
        # that the auto-detector might bucket as classification only
        # for ``≤10 distinct integer values``.
        agent.fit(X, y, task="classification")

        # Capture sklearn-style fitted attributes from the engine.
        self._agent = agent
        self.classes_ = np.asarray(agent.class_labels)
        self.n_features_in_ = int(agent._X_ref.shape[1])
        self.feature_names_in_ = np.asarray(agent.feature_names)

        # Invalidate any prior predict cache.
        self._last_predict_X = None
        self._last_predict_results = None
        return self

    # ── prediction ─────────────────────────────────────────────

    def _predict_full(self, X) -> list[ClassificationResult]:
        """Run the engine on every test row and wrap each result.

        Caches the most recent (X, results) pair: if the caller calls
        ``predict``, then ``predict_proba``, then
        ``predict_with_confidence`` on the *same* X (object identity
        OR value equality), we reuse the cached results. This
        matters because each engine call costs real LLM tokens.
        """
        _check_fitted(self)
        X_arr = _coerce_X(X)

        # Fast path: identical input to last call.
        if self._last_predict_X is not None and self._last_predict_results is not None:
            same_obj = self._last_predict_X is X_arr
            same_val = (
                self._last_predict_X.shape == X_arr.shape
                and np.array_equal(self._last_predict_X, X_arr)
            )
            if same_obj or same_val:
                return self._last_predict_results

        raw_results = self._agent.predict(X_arr)
        wrapped = [ClassificationResult(raw=r) for r in raw_results]
        # Stash a copy of the input so a later mutation of ``X`` by
        # the caller doesn't make the cache silently stale.
        self._last_predict_X = np.array(X_arr, copy=True)
        self._last_predict_results = wrapped
        return wrapped

    def predict(self, X) -> np.ndarray:
        """Predict class labels for ``X``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (or ``"Inconclusive"`` when the
            engine could not pick a unique winner).
        """
        return np.array(
            [r.label for r in self._predict_full(X)],
            dtype=object,
        )

    def predict_proba(self, X) -> np.ndarray:
        """Per-class probability estimates.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Each row is a probability distribution over
            ``self.classes_``. Classes not in the test sample's
            adaptive top-M candidate set get probability ``0`` —
            that's the faithful representation of "the agent did not
            consider this class for this sample" (see
            ``docs/design.md`` §15 for the adaptive top-M rationale).

            For ``Inconclusive`` verdicts where votes WERE cast,
            the normalised weighted-vote distribution is returned.
            For verdicts where NO decisive votes were cast at all
            (every tuple ``Unknown``), the distribution falls back
            to uniform over the candidate classes.

        Notes
        -----
        This is **not** the same as a calibrated probability — the
        engine doesn't fit a probability calibration. The values
        reflect the agent's per-class voting evidence, normalised so
        each row sums to (approximately) 1. For a separately-
        calibrated confidence in the verdict, see
        :meth:`predict_with_confidence`.
        """
        _check_fitted(self)
        results = self._predict_full(X)
        classes = list(self.classes_)
        return np.vstack([r.proba_vector(classes) for r in results])

    def predict_log_proba(self, X) -> np.ndarray:
        """Natural log of :meth:`predict_proba`, with safe handling
        of zeros (mapped to ``-inf``)."""
        proba = self.predict_proba(X)
        with np.errstate(divide="ignore"):
            return np.log(proba)

    def predict_with_confidence(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, confidences)`` per test sample.

        ``confidences[i]`` is the engine's calibrated confidence
        score (margin × sample-size × coverage; see
        ``docs/design.md`` §18.2), NOT ``proba.max()``. The two
        carry different information:

        - ``proba.max()`` is "of the candidates considered, how much
          weighted evidence backs the winner?".
        - ``confidence`` further discounts that for small reference
          pools and a high inconclusive-tuple rate.

        Both are useful and we expose them as separate axes — the
        user explicitly asked for this.
        """
        results = self._predict_full(X)
        labels = np.array([r.label for r in results], dtype=object)
        confs = np.array([r.confidence for r in results], dtype=float)
        return labels, confs

    def predict_full(self, X) -> list[ClassificationResult]:
        """Return the rich :class:`ClassificationResult` wrapper per sample.

        Use this when you need the full evidence — driver
        attribution, per-tuple records, vote summary, the original
        test feature values — instead of just labels and
        probabilities.
        """
        return list(self._predict_full(X))

    # ── score (sklearn contract) ──────────────────────────────

    def score(self, X, y) -> float:
        """Mean accuracy on ``(X, y)`` — the sklearn ``ClassifierMixin``
        contract. ``"Inconclusive"`` predictions count as wrong, which
        matches every existing benchmark in this repository.
        """
        y_true = _coerce_y_classification(y)
        y_pred = self.predict(X)
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(y_pred == y_true))

    # ── introspection ─────────────────────────────────────────

    @property
    def driver_attribution_(self) -> list[dict[str, Any]]:
        """Aggregate driver attribution across the LAST predicted batch.

        Empty list before the first ``predict`` call. Aggregation
        merges per-feature ``n_cited`` and ``abs_strength`` across
        all samples in the batch and re-sorts. Useful for
        cohort-level "which features mattered" summaries.
        """
        if not self._last_predict_results:
            return []
        return _aggregate_driver_attribution(
            [r.driver_attribution for r in self._last_predict_results]
        )

    @property
    def llm_stats_(self) -> dict[str, Any]:
        """Underlying LLM client statistics (calls, tokens, time)."""
        if self._agent is None:
            return {}
        client = self._agent.llm
        stats = getattr(client, "stats", None)
        if callable(stats):
            stats = stats()
        return dict(stats or {})

    def __repr__(self) -> str:
        if self._agent is None:
            return f"MolActAgentClassifier(k={self.k}, fitted=False)"
        return (
            f"MolActAgentClassifier(k={self.k}, "
            f"n_classes={len(self.classes_)}, "
            f"classes={list(self.classes_)})"
        )


def _aggregate_driver_attribution(
    per_sample: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge per-sample driver attributions into a cohort-level list.

    Each per-sample attribution is the engine's own per-feature
    record (see ``MolActAgent._compute_driver_attribution``). We
    additively accumulate ``n_cited`` and ``abs_strength`` across
    samples, take the union of dominant classes (set ``"Mixed"`` when
    they conflict), and concatenate up to three example rationales.

    Sorted by total ``abs_strength`` descending, then by ``n_cited``
    descending (matches the per-sample sort key inside the engine).
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
                    "dominant_classes": set(),
                    "rationales": [],
                },
            )
            slot["n_cited"] += int(entry.get("n_cited", 0) or 0)
            slot["abs_strength"] += float(entry.get("abs_strength", 0.0) or 0.0)
            slot["weighted_score"] += float(
                entry.get("weighted_score", 0.0) or 0.0
            )
            dom = entry.get("dominant_class")
            if dom:
                slot["dominant_classes"].add(str(dom))
            for r in (entry.get("rationales") or [])[:1]:
                if r and len(slot["rationales"]) < 3:
                    slot["rationales"].append(r)

    out: list[dict[str, Any]] = []
    for fn, slot in by_feature.items():
        dom_set = slot.pop("dominant_classes")
        if len(dom_set) == 1:
            slot["dominant_class"] = next(iter(dom_set))
        else:
            slot["dominant_class"] = "Mixed"
        slot["abs_strength"] = round(slot["abs_strength"], 3)
        slot["weighted_score"] = round(slot["weighted_score"], 3)
        out.append(slot)

    out.sort(
        key=lambda s: (s["abs_strength"], s["n_cited"]),
        reverse=True,
    )
    return out
