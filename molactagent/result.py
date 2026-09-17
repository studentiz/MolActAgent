"""Lightweight wrappers around the engine's per-sample result dicts.

The classification and regression engines (:class:`molactagent.agent.MolActAgent`
and :class:`molactagent.cross_classification.CrossClassificationRegressor`)
return rich Python dicts: ``final_label`` / ``predicted_value``,
``confidence``, ``vote_summary``, ``tuple_results``,
``driver_attribution``, ``reference_info``, ``test_feature_values``, and
so on. Those dicts are the canonical "wire format" for the system —
everything serialises through them, every test asserts against them,
and downstream consumers (reports, summaries, the LLM-explanation
prompt) treat them as the source of truth.

This module adds a typed, attribute-accessible **wrapper** layer on
top of those dicts. The wrappers exist for one reason: to turn
result inspection from "remember the dict key" into "look at
tab-completion". Every wrapper is a thin facade — it stores the raw
dict verbatim, exposes the most-used fields as named properties, and
adds a few convenience helpers (Markdown / HTML report rendering,
tidy-DataFrame export). ``raw`` and :meth:`to_dict` always give back
the engine's exact dict shape, so any code that already works on
those dicts continues to work unchanged when it receives a wrapper.

Why a wrapper instead of replacing the dict?

- The engine, the prompt builder, the report renderer, and the test
  suite all key into the dict by string name. Touching that contract
  would cascade through the codebase. A wrapper preserves the
  contract and adds the ergonomics on top.
- Keeping the dict authoritative means new fields the engine adds
  later (e.g. additional CCR diagnostics) become immediately
  available via ``result.raw[new_key]`` with no wrapper change
  required.
- Pickle / JSON serialisation still goes through the underlying dict
  (``to_dict``), so caches and on-disk artefacts remain
  format-stable across wrapper-only changes.

The wrappers do **not** add any new prediction logic. They are pure
projections of the engine output. If you find yourself computing
something derived in a wrapper, ask whether it belongs in the engine
or in :mod:`molactagent.tl` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

__all__ = [
    "DriverEntry",
    "ClassificationResult",
    "RegressionResult",
    "wrap_result",
]


# ── per-driver record ────────────────────────────────────────────────


@dataclass(frozen=True)
class DriverEntry:
    """One entry in a per-tuple ``key_drivers`` list, attribute-accessible.

    Mirrors the schema produced by the LLM and validated in
    :func:`molactagent.agent.MolActAgent._check_tuple` / ``_finalize_tuple``:

    - ``feature``        : the discriminative feature this driver cites
    - ``favored_class``  : which candidate class this feature pushes toward
                           (always one of the tuple's ``candidate_classes``)
    - ``strength``       : the LLM's stated decisiveness, clamped to [0, 1]
    - ``test_value``     : the test sample's value on this feature
    - ``favored_value``  : the favoured class's reference value on this feature
    - ``why``            : the LLM's one-sentence justification

    Frozen because driver records are immutable evidence — the engine
    never mutates them after :meth:`_finalize_tuple` returns.
    """

    feature: str
    favored_class: str
    strength: float
    test_value: float | None
    favored_value: float | None
    why: str

    @classmethod
    def from_dict(cls, d: dict) -> "DriverEntry":
        # Defensive coercion — the wrapper layer must accept every
        # variant the engine has ever produced, including legacy
        # records cached from older versions where some fields might
        # be missing or string-typed.
        def _num(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            feature=str(d.get("feature", "")),
            favored_class=str(d.get("favored_class", "")),
            strength=float(d.get("strength", 0.0) or 0.0),
            test_value=_num(d.get("test_value")),
            favored_value=_num(d.get("favored_value")),
            why=str(d.get("why", "")),
        )


# ── classification result ────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """Attribute-accessible wrapper around a per-sample classification dict.

    The underlying dict is exactly what
    :func:`molactagent.agent.MolActAgent.predict_one` returns when
    ``task='classification'``. See ``docs/design.md`` §18.5 for the
    canonical schema.
    """

    raw: dict[str, Any]

    # ── identity ────────────────────────────────────────────────────

    @property
    def test_index(self) -> Any:
        """Position of this test sample in the input matrix.

        Defaults to ``None`` for results assembled outside the normal
        per-sample loop (e.g. dummy refusal records).
        """
        return self.raw.get("test_index")

    @property
    def task(self) -> str:
        return self.raw.get("task", "classification")

    # ── verdict ─────────────────────────────────────────────────────

    @property
    def label(self) -> str:
        """The predicted class name (or ``"Inconclusive"``).

        Note: even when the verdict is ``"Inconclusive"`` the
        :attr:`proba` distribution is still meaningful — it carries
        the per-class weighted votes that the aggregator was unable
        to resolve into a unique winner.
        """
        return self.raw.get("final_label", "Inconclusive")

    @property
    def confidence(self) -> float:
        """Calibrated confidence in ``[0, 1]``.

        This is the witness-layer calibration from §18.2 of
        ``docs/design.md``: ``margin × sample-size × coverage``. It
        is **not** the same as ``max(self.proba)`` — see
        :attr:`proba` for that.
        """
        return float(self.raw.get("confidence", 0.0) or 0.0)

    @property
    def candidate_classes(self) -> list[str]:
        """The classes the adaptive top-M selector kept "in contention"
        for this sample. May be a strict subset of the agent's
        ``classes_``."""
        return list(self.raw.get("candidate_classes") or [])

    # ── probability distribution ────────────────────────────────────

    def proba(
        self, classes: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """Per-class probabilities derived from the weighted vote tally.

        For each candidate class the probability is its weighted-vote
        share of the per-class weighted total:
        ``proba[c] = vote_summary['per_class'][c]['weighted'] / sum_of_weighted``.

        Classes outside the candidate set always get probability
        ``0`` — this is the faithful representation of "the agent did
        not consider this class for this sample". Callers who want a
        full-class vector can pass ``classes=`` to control the order
        and pad with zeros for classes the agent never saw at fit
        time.

        When the aggregator returned ``Inconclusive`` because no
        single class won decisively but votes were cast, the
        normalised weighted votes are still returned. When **no**
        decisive votes were cast (e.g. every tuple returned
        ``Unknown``), the distribution falls back to uniform over
        the candidate classes — which is the maximally-honest "no
        information" prior.
        """
        vs = self.raw.get("vote_summary") or {}
        per_class = vs.get("per_class") or {}
        target_classes = (
            list(classes) if classes is not None else self.candidate_classes
        )

        # Sum of weighted votes across the candidate classes the
        # engine actually saw (not the user-supplied ``classes``
        # argument — we want the denominator to match the evidence).
        weights: dict[str, float] = {
            cls: float((per_class.get(cls) or {}).get("weighted", 0.0) or 0.0)
            for cls in self.candidate_classes
        }
        total = sum(weights.values())

        if total <= 1e-12 and self.candidate_classes:
            # No decisive evidence anywhere → uniform over the
            # candidate set. Classes outside the candidate set still
            # get 0.
            uniform_p = 1.0 / len(self.candidate_classes)
            uniform = {cls: uniform_p for cls in self.candidate_classes}
            return {cls: float(uniform.get(cls, 0.0)) for cls in target_classes}

        if not target_classes:
            return {}

        # By the time we reach this branch ``total > 1e-12`` (the
        # zero-total case was handled by the uniform-fallback
        # short-circuit above). Dividing by ``total`` directly is
        # safe; we don't carry the redundant ``if total > 0`` guard.
        out: dict[str, float] = {}
        for cls in target_classes:
            w = weights.get(cls, 0.0)
            out[cls] = w / total
        return out

    def proba_vector(self, classes: list[str]) -> np.ndarray:
        """The :meth:`proba` distribution as a fixed-order numpy vector.

        Used by :class:`molactagent.estimators.MolActAgentClassifier`
        to assemble the ``(n_test, n_classes)`` matrix that
        scikit-learn's ``predict_proba`` contract expects.
        """
        d = self.proba(classes=classes)
        return np.array([d[c] for c in classes], dtype=float)

    # ── evidence accessors ─────────────────────────────────────────

    @property
    def vote_summary(self) -> dict[str, Any]:
        return dict(self.raw.get("vote_summary") or {})

    @property
    def confidence_basis(self) -> dict[str, Any]:
        """The decomposed confidence score (margin / sample-size /
        coverage components plus a human-readable rationale).

        Schema documented in ``docs/design.md`` §18.2.
        """
        return dict(self.raw.get("confidence_basis") or {})

    @property
    def reason(self) -> str:
        return str(self.raw.get("reason", ""))

    @property
    def supporting_evidence(self) -> str:
        return str(self.raw.get("supporting_evidence", ""))

    @property
    def opposing_evidence(self) -> str:
        return str(self.raw.get("opposing_evidence", ""))

    @property
    def uncertainty(self) -> str:
        return str(self.raw.get("uncertainty", ""))

    @property
    def driver_attribution(self) -> list[dict[str, Any]]:
        """Aggregated per-feature driver attribution across all tuples."""
        return list(self.raw.get("driver_attribution") or [])

    @property
    def key_features_per_class(self) -> dict[str, list[str]]:
        return dict(self.raw.get("key_features_per_class") or {})

    @property
    def tuple_results(self) -> list[dict[str, Any]]:
        return list(self.raw.get("tuple_results") or [])

    @property
    def reference_info(self) -> dict[str, Any]:
        return dict(self.raw.get("reference_info") or {})

    @property
    def test_feature_values(self) -> dict[str, float]:
        return dict(self.raw.get("test_feature_values") or {})

    @property
    def drivers(self) -> list[DriverEntry]:
        """Flat list of driver entries from the FIRST tuple.

        For multi-tuple aggregation use :attr:`driver_attribution`
        instead — that's the version that's been cross-tuple
        weighted. ``drivers`` here is a quick accessor for the simple
        common case where M=2 and there is exactly one tuple to
        introspect.
        """
        if not self.tuple_results:
            return []
        first_drivers = self.tuple_results[0].get("key_drivers") or []
        return [DriverEntry.from_dict(d) for d in first_drivers]

    # ── interop ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying raw dict (engine wire format).

        Returned dict is a shallow copy — mutating its top level does
        not mutate this wrapper, but mutating nested containers does.
        Callers who need full isolation should ``copy.deepcopy`` the
        result.
        """
        return dict(self.raw)

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(label={self.label!r}, "
            f"confidence={self.confidence:.3f}, "
            f"candidates={self.candidate_classes})"
        )


# ── regression result ────────────────────────────────────────────────


@dataclass
class RegressionResult:
    """Attribute-accessible wrapper around a per-sample regression dict.

    The underlying dict is what
    :func:`molactagent.cross_classification.CrossClassificationRegressor.predict_one`
    returns. See ``docs/design.md`` §19.9 for the canonical schema.
    """

    raw: dict[str, Any]

    # ── identity ────────────────────────────────────────────────────

    @property
    def test_index(self) -> Any:
        return self.raw.get("test_index")

    @property
    def task(self) -> str:
        return self.raw.get("task", "regression")

    @property
    def method(self) -> str:
        return self.raw.get("method", "CCR")

    @property
    def aggregator(self) -> str:
        return self.raw.get("aggregator", "")

    # ── prediction ─────────────────────────────────────────────────

    @property
    def predicted_value(self) -> float | None:
        """The numeric prediction, or ``None`` for a refused prediction.

        See ``docs/design.md`` §19.6 for the refusal policy: CCR
        returns ``None`` only when both phases produced no usable
        winner OR the confidence-weighted aggregator returned no
        usable weight OR a simple aggregator produced a non-finite
        value.
        """
        v = self.raw.get("predicted_value")
        if v is None:
            return None
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    @property
    def is_refused(self) -> bool:
        return self.predicted_value is None

    @property
    def interval(self) -> tuple[float | None, float | None]:
        """Empirical [min, max] of the picked references' actual targets.

        See ``docs/design.md`` §19.7 — this is an honest, tight
        bracket; not a parametric interval.
        """
        iv = self.raw.get("interval") or (None, None)
        try:
            lo = float(iv[0]) if iv[0] is not None else None
            hi = float(iv[1]) if iv[1] is not None else None
        except (TypeError, ValueError):
            return (None, None)
        return (lo, hi)

    @property
    def confidence(self) -> float:
        return float(self.raw.get("confidence", 0.0) or 0.0)

    @property
    def confidence_basis(self) -> dict[str, Any]:
        return dict(self.raw.get("confidence_basis") or {})

    @property
    def extrapolation_flagged(self) -> bool:
        """Whether the prediction falls outside the training-target range."""
        return bool(self.raw.get("extrapolation_flagged", False))

    # ── selected references ────────────────────────────────────────

    @property
    def selected_indices(self) -> list[int]:
        return [int(i) for i in (self.raw.get("selected_indices") or [])]

    @property
    def selected_targets(self) -> list[float]:
        return [float(t) for t in (self.raw.get("selected_targets") or [])]

    @property
    def selected_distances(self) -> list[float]:
        return [float(d) for d in (self.raw.get("selected_distances") or [])]

    # ── per-phase records ─────────────────────────────────────────

    @property
    def phase_target(self) -> dict[str, Any]:
        """CCR Phase 1 — target-bin classification result."""
        return dict(self.raw.get("phase_target") or {})

    @property
    def phase_feature(self) -> dict[str, Any]:
        """CCR Phase 2 — feature-cluster classification result."""
        return dict(self.raw.get("phase_feature") or {})

    @property
    def phase_views_agree(self) -> bool:
        """Whether at least one reference was picked by both CCR phases."""
        return bool(self.raw.get("phase_views_agree", False))

    # ── attribution / context ────────────────────────────────────

    @property
    def driver_attribution(self) -> list[dict[str, Any]]:
        return list(self.raw.get("driver_attribution") or [])

    @property
    def tuple_results(self) -> list[dict[str, Any]]:
        return list(self.raw.get("tuple_results") or [])

    @property
    def reference_info(self) -> dict[str, Any]:
        return dict(self.raw.get("reference_info") or {})

    @property
    def test_feature_values(self) -> dict[str, float]:
        return dict(self.raw.get("test_feature_values") or {})

    @property
    def target_range(self) -> tuple[float, float] | None:
        """Min/max of the reference target set, captured at fit time."""
        ri = self.reference_info
        tr = ri.get("target_range")
        if tr is None:
            return None
        try:
            return (float(tr[0]), float(tr[1]))
        except (TypeError, ValueError):
            return None

    # ── interop ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def __repr__(self) -> str:
        if self.is_refused:
            return "RegressionResult(REFUSED)"
        lo, hi = self.interval
        iv_str = (
            f"[{lo:.4g}, {hi:.4g}]"
            if (lo is not None and hi is not None)
            else "(no interval)"
        )
        return (
            f"RegressionResult(value={self.predicted_value:.4g}, "
            f"interval={iv_str}, confidence={self.confidence:.3f})"
        )


# ── factory ──────────────────────────────────────────────────────────


def wrap_result(raw: dict[str, Any]) -> ClassificationResult | RegressionResult:
    """Wrap a raw engine result in the appropriate typed wrapper.

    Dispatches on the dict's ``task`` key so callers don't need to
    know in advance whether they're holding a classification or
    regression record. Defaults to classification when the key is
    missing (the legacy contract).
    """
    task = (raw or {}).get("task", "classification")
    if task == "regression":
        return RegressionResult(raw=raw)
    return ClassificationResult(raw=raw)
