"""Cohort-level driver attribution synthesis.

The engine returns a per-sample driver attribution list (see
``MolActAgent._compute_driver_attribution`` for classification and
``CrossClassificationRegressor._synthesize_driver_attribution`` for
regression). This module aggregates those per-sample lists across a
batch of predictions so the user can ask "across this whole test
set, which features drove decisions, and in which direction?".

Two callers benefit:

- The estimator wrappers expose ``driver_attribution_`` as a
  property that delegates here.
- :class:`molactagent.tl.LOOCVResult` consumers can pass
  ``[r['raw'].get('driver_attribution', []) for r in fold_records]``
  directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DriverAggregation:
    """Cohort-level per-feature driver summary.

    ``records`` is the legacy schema-compatible list of dicts that
    the engine and the explanation prompt already understand.
    ``to_dataframe()`` exposes the same content as a tidy table for
    notebook use.
    """

    records: list[dict[str, Any]] = field(default_factory=list)

    def top_features(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the top-``n`` features by absolute strength."""
        return self.records[:n]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "feature", "n_cited", "abs_strength",
                    "weighted_score", "dominant_class",
                    "dominant_direction", "description",
                ]
            )
        # Drop the bulky ``rationales`` list to keep the frame
        # printable; users who need the rationales can index
        # ``.records`` directly.
        cleaned = [
            {k: v for k, v in r.items() if k != "rationales"}
            for r in self.records
        ]
        return pd.DataFrame(cleaned)

    def per_class_top_features(self, n: int = 6) -> dict[str, list[str]]:
        """Top ``n`` feature names per dominant class.

        Only meaningful for classification aggregations (regression
        records have ``dominant_direction``, not ``dominant_class``).
        Returns an empty dict for regression-style records.
        """
        per_class: dict[str, list[str]] = {}
        for r in self.records:
            cls = r.get("dominant_class")
            if cls is None or cls == "Mixed":
                continue
            per_class.setdefault(cls, []).append(r["feature"])
        return {cls: feats[:n] for cls, feats in per_class.items()}


def aggregate_drivers(
    per_sample: list[list[dict[str, Any]]] | list[Any],
) -> DriverAggregation:
    """Merge per-sample driver attributions into a cohort summary.

    Accepts either:

    - A list of per-sample driver-attribution lists (the raw
      shape produced by the engine):
      ``[ [{feature: ..., ...}, ...],   # sample 0
          [{feature: ..., ...}, ...],   # sample 1
          ... ]``
    - A list of result wrappers (:class:`ClassificationResult` or
      :class:`RegressionResult`); we extract ``driver_attribution``
      from each.
    - A list of fold-record dicts from
      :class:`molactagent.tl.LOOCVResult.fold_records`; we look at
      each record's ``raw['driver_attribution']``.

    The aggregation logic itself is identical to the per-batch
    aggregator in
    :func:`molactagent.estimators.classifier._aggregate_driver_attribution`
    (and the regression analogue) — kept here as the canonical
    standalone entry point so users don't need an estimator
    instance just to roll up an existing prediction batch.
    """
    rows = _coerce_to_attribution_lists(per_sample)
    if not rows:
        return DriverAggregation(records=[])

    by_feature: dict[str, dict[str, Any]] = {}
    for sample_attribution in rows:
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
                    "dominant_directions": set(),
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
            direction = entry.get("dominant_direction")
            if direction:
                slot["dominant_directions"].add(str(direction))
            for r in (entry.get("rationales") or [])[:1]:
                if r and len(slot["rationales"]) < 3:
                    slot["rationales"].append(r)

    out: list[dict[str, Any]] = []
    for slot in by_feature.values():
        dom_classes = slot.pop("dominant_classes")
        dom_dirs = slot.pop("dominant_directions")
        if dom_classes:
            slot["dominant_class"] = (
                next(iter(dom_classes)) if len(dom_classes) == 1
                else "Mixed"
            )
        if dom_dirs:
            slot["dominant_direction"] = (
                next(iter(dom_dirs)) if len(dom_dirs) == 1 else "mixed"
            )
        slot["abs_strength"] = round(slot["abs_strength"], 3)
        slot["weighted_score"] = round(slot["weighted_score"], 3)
        out.append(slot)

    out.sort(
        key=lambda s: (s["abs_strength"], s["n_cited"]),
        reverse=True,
    )
    return DriverAggregation(records=out)


def _coerce_to_attribution_lists(
    per_sample: list[Any],
) -> list[list[dict[str, Any]]]:
    """Accept any of the three input shapes and return a clean
    list-of-lists-of-dicts."""
    out: list[list[dict[str, Any]]] = []
    for entry in per_sample:
        if isinstance(entry, list):
            # Already in attribution-list shape.
            out.append([d for d in entry if isinstance(d, dict)])
            continue
        # Result wrapper or fold record.
        if hasattr(entry, "driver_attribution"):
            attribution = entry.driver_attribution
            if isinstance(attribution, list):
                out.append(attribution)
                continue
        if isinstance(entry, dict):
            # Fold record: look in ``raw['driver_attribution']``.
            raw = entry.get("raw") or {}
            if isinstance(raw, dict):
                attribution = raw.get("driver_attribution") or []
                if isinstance(attribution, list):
                    out.append(attribution)
                    continue
            # Last-ditch: maybe the dict IS the per-sample
            # attribution itself in the legacy script-record
            # shape.
            if isinstance(entry.get("feature"), str):
                out.append([entry])
                continue
        # Unknown type — skip rather than crash.
    return out
