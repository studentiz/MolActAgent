"""MolActAgent — contrastive predictor for both classification and
regression, with rich, evidence-based explanations.

Classification path (per test sample):
  1. Find k nearest neighbours per class.
  2. Adaptive top-M: keep only the classes whose closest neighbour
     sits within ``m_threshold`` x the closest class's closest
     neighbour distance (default 1.5x), capped at ``max_m`` (default
     4) and floored at 2 candidates.
  3. Form (M+1)-tuples = one neighbour per candidate class + Test.
  4. For every tuple, run ONE structured-JSON LLM call that produces
     a verdict + named drivers + per-driver justification +
     counter-evidence (``_classify_tuple``).
  5. Aggregate tuple verdicts deterministically into a final class
     label (or ``"Inconclusive"``), a confidence score, and a
     weighted per-class driver attribution (``_aggregate_results``).
  6. Optional ``generate_explanation`` makes ONE more LLM call that
     writes a Markdown report from the rich tuple evidence.

Binary classification falls out as the M=2 special case: the same
code paths run with two candidate classes whose actual user labels
(e.g. ``"active"`` / ``"inactive"``) flow through the prompts and
reports verbatim.

Regression path (per test sample) — delegated to the Cross-
Classification Regressor (CCR), see
:class:`molactagent.cross_classification.CrossClassificationRegressor`:
  Phase 1 ("target view")  : equal-frequency bin the targets into
      N=floor(sqrt(n))+1 classes; classify with MolActAgent (top-M=2).
  Phase 2 ("feature view") : hierarchical / average-linkage cluster
      the standardised features under cosine distance into M=
      floor(sqrt(n))+1 clusters; classify with MolActAgent (top-M=2).
  Selection                : top-2 nearest references in each
      predicted class via the project's contextual distance.
  Aggregation              : confidence-weighted mean of those
      references' actual target values; refs picked by both phases
      get summed weights.
The classification engine above is reused twice per test sample as
the regression backend, so the same contrastive + adaptive-top-M
machinery serves both tasks.
"""

import logging
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from .cross_classification import (
    DEFAULT_AGENT_K as CCR_DEFAULT_AGENT_K,
    DEFAULT_AGGREGATOR as CCR_DEFAULT_AGGREGATOR,
    DEFAULT_K_PER_CLASS as CCR_DEFAULT_K_PER_CLASS,
    DEFAULT_N_MODE as CCR_DEFAULT_N_MODE,
    CrossClassificationRegressor,
)
from .neighbors import (
    DEFAULT_K_LOCAL,
    DEFAULT_SIMILARITY_BLEND,
    NeighborFinder,
)
from .preprocessing import DataPreprocessor
from .prompts import PromptBuilder
from .utils import validate_inputs

logger = logging.getLogger(__name__)


class MolActAgent:
    def __init__(
        self,
        llm_client,
        k: int = 6,
        metric: str = "euclidean",
        scaler: str = "standard",
        feature_names: list[str] | None = None,
        feature_descriptions: dict[str, str] | None = None,
        background: str | None = None,
        use_scaled_for_prompt: bool = False,
        max_workers: int = 100,
        variance_threshold: float = 0.0,
        diversity_ratio: float = 0.1,
        random_state: int | None = None,
        max_check_retries: int = 3,
        llm_temperature: float = 0.0,
        m_threshold: float = 1.5,
        max_m: int = 4,
        k_local: int = DEFAULT_K_LOCAL,
        similarity_blend: float = DEFAULT_SIMILARITY_BLEND,
        # ── regression backend (CCR — Cross-Classification Regressor) ──
        ccr_aggregator: str = CCR_DEFAULT_AGGREGATOR,
        ccr_k_per_class: int = CCR_DEFAULT_K_PER_CLASS,
        ccr_agent_k: int = CCR_DEFAULT_AGENT_K,
        ccr_n_mode: str = CCR_DEFAULT_N_MODE,
        ccr_n_target_bins: int | None = None,
        ccr_n_feature_clusters: int | None = None,
        **_legacy_kwargs,
    ):
        # Silently absorb legacy kwargs (use_reflection,
        # use_json_formatter, use_distance_weighted_agg, the old
        # bootstrap-tuple regression knobs ``regression_n_tuples`` /
        # ``regression_local_slopes``, etc.) so existing caller
        # scripts keep working without touching them. DEBUG-level
        # so this doesn't pollute INFO logs during evaluation.
        if _legacy_kwargs:
            ignored = sorted(_legacy_kwargs.keys())
            logger.debug(
                f"MolActAgent: ignoring legacy kwargs {ignored} "
                "(removed in pipeline rewrite)"
            )

        self.llm = llm_client
        self.k = k
        self.metric = metric
        self.scaler_type = scaler
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions
        self.background = background
        self.use_scaled_for_prompt = use_scaled_for_prompt
        self.max_workers = max_workers
        self.variance_threshold = variance_threshold
        self.diversity_ratio = diversity_ratio
        self.random_state = random_state
        self.max_check_retries = max_check_retries
        self.llm_temperature = llm_temperature
        self.m_threshold = m_threshold
        self.max_m = max_m
        self.k_local = k_local
        self.similarity_blend = similarity_blend
        # CCR (regression backend) configuration. The defaults bake
        # in the choices that earned their place in the BSLT n=16
        # and penguins n=24 ablation suites: confidence-weighted
        # aggregation, top-2 nearest per predicted class, MolActAgent
        # k=2 per phase, floor(sqrt(n))+1 bins/clusters per phase.
        self.ccr_aggregator = ccr_aggregator
        self.ccr_k_per_class = ccr_k_per_class
        self.ccr_agent_k = ccr_agent_k
        self.ccr_n_mode = ccr_n_mode
        self.ccr_n_target_bins = ccr_n_target_bins
        self.ccr_n_feature_clusters = ccr_n_feature_clusters
        self._regression_backend: CrossClassificationRegressor | None = None

        self._feature_mask: np.ndarray | None = None
        self._preprocessor: DataPreprocessor | None = None
        self._neighbor_finder: NeighborFinder | None = None
        self._prompt_builder: PromptBuilder | None = None
        self._X_ref: np.ndarray | None = None
        self._X_ref_scaled: np.ndarray | None = None
        self._y_ref: np.ndarray | None = None
        self._is_fitted = False
        self.class_labels: list[str] = []
        self.task: str = "classification"
        self.target_range: tuple[float, float] | None = None

    # ── public API ───────────────────────────────────────────────────

    def fit(self, X_ref, y_ref, task: str | None = None):
        """Fit the agent on a labelled reference set.

        ``task`` auto-detects from ``y_ref`` dtype + cardinality when
        not given. Pass ``task="classification"`` or
        ``task="regression"`` to override the detection (e.g. to
        force regression on a small-cardinality integer target).
        """
        X_ref = np.array(X_ref, dtype=np.float64)
        y_ref_raw = np.asarray(y_ref)

        # Detect (or accept) the task before dtype coercion, so we
        # don't accidentally stringify a continuous numeric target.
        if task is None:
            self.task = self._detect_task_from_array(y_ref_raw)
        elif task in ("classification", "regression"):
            self.task = task
        else:
            raise ValueError(
                f"task must be 'classification' or 'regression'; "
                f"got {task!r}"
            )

        if self.task == "regression":
            y_ref = y_ref_raw.astype(np.float64)
        else:
            y_ref = y_ref_raw.astype(str)

        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(X_ref.shape[1])]

        variances = np.var(X_ref, axis=0)
        self._feature_mask = variances > self.variance_threshold
        n_removed = int((~self._feature_mask).sum())
        if n_removed > 0:
            logger.info(
                f"Variance filter: removed {n_removed}/{len(self._feature_mask)} "
                f"features (threshold={self.variance_threshold}), "
                f"kept {self._feature_mask.sum()}"
            )
            X_ref = X_ref[:, self._feature_mask]
            self.feature_names = [
                n for n, keep in zip(self.feature_names, self._feature_mask) if keep
            ]
            if self.feature_descriptions:
                self.feature_descriptions = {
                    k: v for k, v in self.feature_descriptions.items()
                    if k in self.feature_names
                }

        validate_inputs(X_ref, y_ref, X_ref, self.feature_names)

        self._preprocessor = DataPreprocessor(self.scaler_type)
        self._X_ref_scaled = self._preprocessor.fit_transform(X_ref)
        self._X_ref = X_ref
        self._y_ref = y_ref

        if self.task == "regression":
            self.class_labels = []
            self.target_range = (
                float(y_ref.min()), float(y_ref.max()),
            )
        else:
            unique_classes, _ = np.unique(y_ref, return_counts=True)
            self.class_labels = sorted(unique_classes.tolist())
            self.target_range = None

        self._neighbor_finder = NeighborFinder(
            k=self.k,
            metric=self.metric,
            diversity_ratio=self.diversity_ratio,
            m_threshold=self.m_threshold,
            max_m=self.max_m,
            k_local=self.k_local,
            similarity_blend=self.similarity_blend,
        )
        self._neighbor_finder.fit(self._X_ref_scaled)
        self._prompt_builder = PromptBuilder(
            feature_names=self.feature_names,
            feature_descriptions=self.feature_descriptions,
            background=self.background,
        )

        # Regression backend (CCR — Cross-Classification Regressor):
        # in regression mode, ``predict_one`` delegates to a CCR
        # instance constructed here. CCR internally builds two of
        # its own MolActAgent classifiers — one over equal-frequency
        # target bins, one over feature-space clusters — and
        # aggregates their picks via confidence-weighted mean. The
        # outer MolActAgent stays the user-facing API; the bootstrap-
        # tuple regression backend is removed.
        if self.task == "regression":
            self._regression_backend = CrossClassificationRegressor(
                llm_client=self.llm,
                feature_names=list(self.feature_names),
                feature_descriptions=dict(self.feature_descriptions or {}),
                background=self.background,
                n_target_bins=self.ccr_n_target_bins,
                n_feature_clusters=self.ccr_n_feature_clusters,
                n_mode=self.ccr_n_mode,
                k_per_class=self.ccr_k_per_class,
                agent_k=self.ccr_agent_k,
                max_workers=self.max_workers,
                llm_temperature=self.llm_temperature,
                aggregator=self.ccr_aggregator,
                random_state=self.random_state,
            ).fit(X_ref, y_ref)
        else:
            self._regression_backend = None

        self._is_fitted = True

        if self.task == "regression":
            logger.info(
                f"MolActAgent fitted [regression / CCR]: "
                f"{X_ref.shape[0]} samples, {X_ref.shape[1]} features, "
                f"target range [{self.target_range[0]:.4g}, "
                f"{self.target_range[1]:.4g}], "
                f"N={self._regression_backend._n_bins_used} bins, "
                f"M={self._regression_backend._n_clusters_used} clusters, "
                f"aggregator={self.ccr_aggregator!r}"
            )
        else:
            unique_classes, counts = np.unique(y_ref, return_counts=True)
            per_class = ", ".join(
                f"{lbl}={int(c)}" for lbl, c in zip(unique_classes, counts)
            )
            logger.info(
                f"MolActAgent fitted [classification]: {X_ref.shape[0]} "
                f"samples ({per_class}), {X_ref.shape[1]} features, "
                f"{len(self.class_labels)} classes"
            )
        return self

    @staticmethod
    def _detect_task_from_array(y_ref: np.ndarray) -> str:
        """Auto-detect task from a numpy y_ref array."""
        if y_ref.dtype == bool:
            return "classification"
        if not np.issubdtype(y_ref.dtype, np.number):
            return "classification"
        try:
            finite = y_ref[~np.isnan(y_ref.astype(float, copy=False))]
        except (TypeError, ValueError):
            finite = y_ref
        if len(finite) == 0:
            return "classification"
        n_distinct = int(np.unique(finite).size)
        try:
            all_integer = bool(np.all(finite == finite.astype(int)))
        except (TypeError, ValueError):
            all_integer = False
        if all_integer and n_distinct <= 10:
            return "classification"
        return "regression"

    def predict(self, X_test) -> list[dict]:
        if not self._is_fitted:
            raise RuntimeError("MolActAgent is not fitted. Call fit() first.")

        X_test = np.array(X_test, dtype=np.float64)
        if self._feature_mask is not None:
            X_test = X_test[:, self._feature_mask]
        if X_test.shape[1] != self._X_ref.shape[1]:
            raise ValueError(
                f"Feature dim mismatch: expected {self._X_ref.shape[1]}, "
                f"got {X_test.shape[1]}"
            )

        results = []
        for i in range(X_test.shape[0]):
            logger.info(
                f"[PREDICT] sample {i + 1}/{X_test.shape[0]} starting..."
            )
            t0 = time.time()
            result = self.predict_one(X_test[i], test_id=i)
            elapsed = time.time() - t0
            logger.info(
                f"[PREDICT] sample {i + 1}/{X_test.shape[0]} done  "
                f"{elapsed:.1f}s  label={result.get('final_label')}"
            )
            results.append(result)
        return results

    def predict_one(self, x_test, test_id=None) -> dict:
        if not self._is_fitted:
            raise RuntimeError("MolActAgent is not fitted. Call fit() first.")

        if self.task == "regression":
            return self._predict_one_regression(x_test, test_id=test_id)
        return self._predict_one_classification(x_test, test_id=test_id)

    def _predict_one_classification(self, x_test, test_id=None) -> dict:
        x_test = np.array(x_test, dtype=np.float64)
        if self._feature_mask is not None and x_test.shape[0] == len(self._feature_mask):
            x_test = x_test[self._feature_mask]
        x_test_scaled = self._preprocessor.transform(x_test.reshape(1, -1))[0]

        nbr = self._neighbor_finder.find_local_references(
            self._X_ref_scaled, self._y_ref, x_test_scaled,
        )
        candidate_classes = self._neighbor_finder.select_top_m_classes(nbr)
        if len(candidate_classes) < 2:
            return self._empty_result(
                test_id,
                "Fewer than 2 candidate classes available in the "
                "reference set; cannot run a contrastive comparison.",
            )

        per_class_summary = "  ".join(
            f"{cls}[{len(nbr.class_indices.get(cls, []))} d_min="
            f"{nbr.min_distance(cls):.2f}]"
            for cls in candidate_classes
        )
        logger.info(
            f"[TEST {test_id}] M={len(candidate_classes)} "
            f"candidates={candidate_classes}  {per_class_summary}"
        )

        # Per-class neighbour pools (use all neighbours including
        # degenerate ones; tuple-level filtering below will drop tuples
        # whose references coincide with the test sample).
        class_neighbors = {
            cls: nbr.class_indices.get(cls, np.array([], dtype=int)).tolist()
            for cls in candidate_classes
        }
        # Guard against an empty neighbour pool for any candidate class.
        if any(len(v) == 0 for v in class_neighbors.values()):
            return self._empty_result(
                test_id,
                "At least one candidate class has no neighbours in the "
                "reference set.",
            )

        all_combos = list(product(*[class_neighbors[c] for c in candidate_classes]))
        if not all_combos:
            return self._empty_result(test_id, "No usable tuple combinations.")

        X_ref_prompt = (
            self._X_ref_scaled if self.use_scaled_for_prompt else self._X_ref
        )
        x_prompt = x_test_scaled if self.use_scaled_for_prompt else x_test

        tuples_to_eval: list[dict] = []
        n_degen_tuples = 0
        for combo in all_combos:
            is_degen = False
            for ref_idx in combo:
                if np.allclose(X_ref_prompt[ref_idx], x_prompt, atol=1e-6):
                    is_degen = True
                    break
            if is_degen:
                n_degen_tuples += 1
                continue
            tuples_to_eval.append({
                cls: int(idx)
                for cls, idx in zip(candidate_classes, combo)
            })

        if n_degen_tuples:
            logger.warning(
                f"[TEST {test_id}] {n_degen_tuples} degenerate tuples "
                "excluded (a reference coincided with the test sample)"
            )

        if not tuples_to_eval:
            return self._empty_result(
                test_id,
                f"All {n_degen_tuples} candidate tuples were degenerate "
                "(test sample identical to the chosen references).",
            )

        logger.info(
            f"[TEST {test_id}] tuples to evaluate: {len(tuples_to_eval)} "
            f"(M={len(candidate_classes)}, "
            f"+{n_degen_tuples} degenerate excluded)"
        )

        tuple_results = self._evaluate_tuples(
            tuples_to_eval, candidate_classes, X_ref_prompt, x_prompt, test_id,
        )

        n_per_class_ref = {
            cls: int(np.sum(self._y_ref == cls))
            for cls in self.class_labels
        }
        n_per_class_nbr = {
            cls: int(len(nbr.class_indices.get(cls, [])))
            for cls in candidate_classes
        }

        result = self._aggregate_results(
            tuple_results, test_id,
            candidate_classes=candidate_classes,
            n_degenerate=n_degen_tuples,
            class_reference_counts=n_per_class_ref,
        )

        result["task"] = "classification"
        result["reference_info"] = {
            "total_reference_samples": int(len(self._y_ref)),
            "class_reference_counts": n_per_class_ref,
            "class_neighbor_counts": n_per_class_nbr,
            "class_neighbor_indices": {
                cls: nbr.class_indices.get(cls, np.array([], dtype=int)).tolist()
                for cls in candidate_classes
            },
            "n_comparisons": len(tuples_to_eval),
            "n_degenerate_excluded": n_degen_tuples,
            "candidate_classes": list(candidate_classes),
        }
        result["test_feature_values"] = {
            n: float(v) for n, v in zip(self.feature_names, x_prompt)
        }
        return result

    # ── regression path ─────────────────────────────────────────────

    def _predict_one_regression(self, x_test, test_id=None) -> dict:
        """Regression prediction — delegates to the CCR backend
        (Cross-Classification Regressor) constructed at fit time.

        CCR runs two MolActAgent classifications under the hood:
        Phase 1 over equal-frequency target bins, Phase 2 over
        feature-space clusters. The picks from each phase are
        combined into a single numeric verdict by the configured
        ``ccr_aggregator`` (default: confidence-weighted mean).

        See :class:`molactagent.cross_classification.CrossClassificationRegressor`
        for the full schema.

        Variance-mask handling: when this method is called directly
        with a full-dimensional ``x_test`` (e.g. by
        ``molactagent.pipeline.run_prediction``, which dispatches per-
        sample via ``predict_one`` rather than the batch ``predict``),
        we MUST apply the outer agent's ``_feature_mask`` here. CCR
        was fit on the already-filtered reference matrix, so its
        inner classifiers expect the filtered dimension; passing the
        original (pre-filter) ``x_test`` straight through would
        fail at the inner ``StandardScaler.transform`` call. The
        check mirrors the same dim-match guard used in
        :meth:`_predict_one_classification`, so callers that have
        already applied the mask (e.g. ``self.predict(X_test)``)
        are not double-masked.
        """
        if self._regression_backend is None:
            raise RuntimeError(
                "Regression backend is not initialised. Did you call "
                "fit() with a continuous-numeric target?"
            )
        x_test = np.asarray(x_test, dtype=np.float64)
        if (
            self._feature_mask is not None
            and x_test.shape[0] == len(self._feature_mask)
        ):
            x_test = x_test[self._feature_mask]
        return self._regression_backend.predict_one(
            x_test, test_id=test_id,
        )

    def analyze(
        self,
        x_test,
        test_id=None,
        class_labels: list[str] | None = None,
    ) -> tuple[dict, str]:
        """Convenience: run prediction AND generate Markdown explanation
        in one call. Returns ``(result_dict, markdown_text)``.
        """
        result = self.predict_one(x_test, test_id=test_id)
        explanation = self.generate_explanation(
            result, class_labels=class_labels,
        )
        return result, explanation

    # ── tuple evaluation (single LLM call per tuple, with retry) ─────

    def _evaluate_tuples(
        self,
        tuples: list[dict],
        candidate_classes: list[str],
        X_ref_prompt: np.ndarray,
        x_prompt: np.ndarray,
        test_id,
    ) -> list[dict]:
        """Evaluate every tuple in parallel.

        Returns a list ordered deterministically by input position so
        downstream aggregation and attribution are reproducible across
        runs even when the thread pool finishes tuples in a different
        sequence.
        """
        total = len(tuples)
        results_by_idx: dict[int, dict] = {}

        def _run(idx, tup):
            tag_prefix = f"t{test_id}_tup{idx+1}/{total}"
            t0 = time.time()
            try:
                references = {
                    cls: X_ref_prompt[ref_idx]
                    for cls, ref_idx in tup.items()
                }
                res = self._classify_tuple(
                    references=references,
                    test_sample=x_prompt,
                    ref_indices=tup,
                    tag_prefix=tag_prefix,
                )
                elapsed = time.time() - t0
                logger.debug(
                    f"[TUPLE {idx+1}/{total}] {elapsed:.1f}s  "
                    f"refs={tup} -> {res.get('label')}  "
                    f"conf={res.get('confidence', 0):.2f}  "
                    f"drivers={len(res.get('key_drivers', []))}"
                )
                return res
            except Exception as e:
                elapsed = time.time() - t0
                logger.error(
                    f"[TUPLE {idx+1}/{total}] FAILED {elapsed:.1f}s  "
                    f"error={e}"
                )
                return self._fallback_tuple(tup, candidate_classes, str(e))

        effective_workers = min(self.max_workers, total)
        throttle_before = self.llm.throttle_count

        if effective_workers > 1:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(_run, idx, tup): idx
                    for idx, tup in enumerate(tuples)
                }
                for f in as_completed(futures):
                    idx = futures[f]
                    results_by_idx[idx] = f.result()
        else:
            for idx, tup in enumerate(tuples):
                results_by_idx[idx] = _run(idx, tup)

        throttle_new = self.llm.throttle_count - throttle_before
        if throttle_new > 0:
            logger.warning(
                f"[TEST {test_id}] {throttle_new} throttle event(s) "
                "observed; consider lowering max_workers for this run."
            )

        return [results_by_idx[i] for i in range(total)]

    def _classify_tuple(
        self,
        references: dict[str, np.ndarray],
        test_sample: np.ndarray,
        ref_indices: dict[str, int],
        tag_prefix: str = "",
    ) -> dict:
        candidate_classes = list(references.keys())
        ref_stack = np.stack(
            [references[cls] for cls in candidate_classes], axis=0,
        )
        n_total = ref_stack.shape[1]

        # Discriminative mask: a feature is "discriminative" for
        # this tuple iff its value differs across the M reference
        # rows. We drop non-discriminative features from the
        # prompt entirely — they cannot inform the contrast and
        # would only consume LLM tokens. Implementation: a feature
        # is discriminative iff at least one row differs from row
        # 0 (transitive across all rows because we OR over rows).
        disc_mask = (ref_stack != ref_stack[0]).any(axis=0)
        n_disc = int(disc_mask.sum())

        logger.debug(
            f"[{tag_prefix}] discriminative features: {n_disc}/{n_total}"
        )

        if n_disc == 0:
            logger.warning(
                f"[{tag_prefix}] No discriminative features — all "
                "references are numerically identical"
            )
            return {
                "candidate_classes": candidate_classes,
                "ref_indices": ref_indices,
                "label": "Unknown",
                "confidence": 0.0,
                "class_scores": {
                    c: round(1.0 / len(candidate_classes), 3)
                    for c in candidate_classes
                },
                "n_aligned": {c: 0 for c in candidate_classes},
                "n_neutral": 0,
                "baseline_class": "Unknown",
                "summary": (
                    "All references are numerically identical; no "
                    "comparison possible."
                ),
                "reason": "No discriminative features between references",
                "discriminative_features": [],
                "key_drivers": [],
                "supporting_features": [],
                "opposing_features": [],
                "counter_evidence": "",
                "uncertainty": "References are degenerate.",
            }

        disc_refs = ref_stack[:, disc_mask]
        test_disc = test_sample[disc_mask]
        disc_names = [
            n for n, m in zip(self._prompt_builder.feature_names, disc_mask) if m
        ]
        disc_set = set(disc_names)

        # Per-feature multi-way alignment tally: which class's
        # reference is numerically closest to Test on this
        # feature?
        #
        # This is the **witness layer** in action (see
        # docs/design.md §5). The tally is a deterministic,
        # embarrassing-to-get-wrong computation Python performs
        # and presents to the LLM as ground truth. The LLM is
        # told to default to the argmax of this tally unless it
        # has decisive domain-grounded evidence to override it
        # (the "soft baseline check" in ``_check_tuple``).
        #
        # Tie semantics: when ≥2 classes are exactly equidistant
        # to Test on a feature, that feature contributes to the
        # ``n_neutral`` bucket rather than splitting the vote.
        # Splitting would let the LLM treat the feature as
        # informative when it carries no discriminative signal;
        # routing it to neutral keeps the witness honest.
        abs_diffs = np.abs(disc_refs - test_disc)
        min_per_feature = abs_diffs.min(axis=0)
        n_aligned = {cls: 0 for cls in candidate_classes}
        n_neutral = 0
        for fi in range(n_disc):
            min_d = min_per_feature[fi]
            equal_count = int(np.sum(abs_diffs[:, fi] == min_d))
            if equal_count > 1:
                # Multiple classes equidistant to Test → not
                # discriminative for this tuple; bucket as neutral.
                n_neutral += 1
                continue
            # Unique closest class → that class earns this
            # feature's vote.
            winner_cls = candidate_classes[int(np.argmin(abs_diffs[:, fi]))]
            n_aligned[winner_cls] += 1

        baseline_class = self._baseline_class_from_counts(n_aligned)
        logger.debug(
            f"[{tag_prefix}] alignment tally: {n_aligned}  "
            f"neutral={n_neutral}  baseline={baseline_class}"
        )

        # Per-reference texture blocks: each candidate-class
        # reference paired with its own k_local nearest reference-set
        # neighbours and their class labels. The LLM perceives
        # neighbourhood composition (deep-territory vs boundary) by
        # reading these directly — see docs/design.md §3.
        reference_neighbourhoods: list[dict] = []
        X_for_texture = (
            self._X_ref_scaled if self.use_scaled_for_prompt else self._X_ref
        )
        for cls, ri in ref_indices.items():
            sig = (
                self._neighbor_finder.signature(ri)
                if self._neighbor_finder is not None else None
            )
            if sig is None or sig.is_empty:
                continue
            nbr_idx = sig.indices.tolist()
            reference_neighbourhoods.append({
                "ref_label_id": f"{cls}_R{ri}",
                "ref_value": str(self._y_ref[ri]),
                "ref_features": X_for_texture[ri][disc_mask],
                "neighbour_ids": [f"R{ni}" for ni in nbr_idx],
                "neighbour_values": [
                    str(self._y_ref[ni]) for ni in nbr_idx
                ],
                "neighbour_features": [
                    X_for_texture[ni][disc_mask] for ni in nbr_idx
                ],
            })

        # Build the prompt with the discriminative subset only.
        msgs = self._prompt_builder.build_tuple_prompt(
            references=[disc_refs[i] for i in range(disc_refs.shape[0])],
            class_names=candidate_classes,
            test_sample=test_disc,
            feature_names=disc_names,
            n_total_features=n_total,
            n_aligned=n_aligned,
            n_neutral=n_neutral,
            baseline_class=baseline_class,
            reference_neighbourhoods=reference_neighbourhoods,
        )

        last_response = ""
        last_issues: list[str] = ["No response yet"]
        for attempt in range(self.max_check_retries + 1):
            tag = (
                f"{tag_prefix}_ANA"
                if attempt == 0
                else f"{tag_prefix}_ANA_r{attempt}"
            )
            raw = self.llm.chat(
                msgs, tag=tag, max_tokens=2000,
                temperature=self.llm_temperature,
            )
            last_response = raw
            logger.debug(f"[{tag}] raw:\n{raw}")

            parsed = self._safe_parse_tuple_json(raw)
            if parsed is None:
                last_issues = [
                    "Output was not valid JSON. Emit ONLY the JSON "
                    "object described in the schema, with no markdown "
                    "fences and no surrounding prose."
                ]
            else:
                ok, fixed, issues = self._check_tuple(
                    parsed, disc_set, candidate_classes,
                    n_aligned=n_aligned,
                    baseline_class=baseline_class,
                )
                if ok:
                    return self._finalize_tuple(
                        fixed, ref_indices, candidate_classes, disc_names,
                        n_aligned=n_aligned,
                        n_neutral=n_neutral,
                        baseline_class=baseline_class,
                    )
                last_issues = issues

            logger.info(
                f"[{tag_prefix}] CHECK failed (attempt {attempt}): "
                f"{'; '.join(last_issues[:3])}"
            )
            if attempt < self.max_check_retries:
                msgs = self._prompt_builder.build_feedback_messages(
                    msgs, last_response, last_issues,
                )

        # Validation exhausted: still return the deterministic baseline
        # verdict + alignment counts so this tuple contributes useful
        # information instead of falling through to Unknown.
        logger.warning(
            f"[{tag_prefix}] LLM validation exhausted after "
            f"{self.max_check_retries + 1} attempts; "
            f"falling back to deterministic baseline ({baseline_class})"
        )
        total_aligned = sum(n_aligned.values())
        if total_aligned > 0:
            class_scores = {
                cls: round(n / total_aligned, 3)
                for cls, n in n_aligned.items()
            }
        else:
            eq = round(1.0 / len(candidate_classes), 3)
            class_scores = {cls: eq for cls in candidate_classes}
        sorted_scores = sorted(class_scores.values(), reverse=True)
        margin = sorted_scores[0] - (
            sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        )
        confidence = round(0.5 * margin + 0.4, 3)
        return {
            "candidate_classes": candidate_classes,
            "ref_indices": ref_indices,
            "label": baseline_class,
            "confidence": confidence,
            "class_scores": class_scores,
            "n_aligned": n_aligned,
            "n_neutral": int(n_neutral),
            "baseline_class": baseline_class,
            "summary": (
                f"Numerical alignment tally: {n_aligned}. "
                "(LLM rationale unavailable.)"
            ),
            "reason": (
                f"Validation failed {self.max_check_retries + 1} times "
                f"({'; '.join(last_issues[:2])}); used deterministic "
                "baseline."
            ),
            "discriminative_features": disc_names,
            "key_drivers": [],
            "supporting_features": [],
            "opposing_features": [],
            "counter_evidence": "",
            "uncertainty": "LLM rationale unavailable; relied on tally.",
        }

    # ── tuple-level parsing & validation ─────────────────────────────

    @staticmethod
    def _safe_parse_tuple_json(raw: str) -> dict | None:
        """Best-effort JSON extraction from a tuple LLM response."""
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, count=1).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fall back: find the first balanced { ... } block.
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _baseline_class_from_counts(n_aligned: dict[str, int]) -> str:
        """Verdict from the deterministic alignment tally alone.

        The class with the most feature-votes wins. Ties (or all-zero
        counts) collapse to ``"Unknown"`` so the LLM has the freedom to
        break the deadlock with a domain-grounded argument — but if
        there is a unique top class, that is the baseline the LLM is
        expected to match.
        """
        if not n_aligned:
            return "Unknown"
        sorted_counts = sorted(
            n_aligned.items(), key=lambda kv: kv[1], reverse=True,
        )
        if sorted_counts[0][1] == 0:
            return "Unknown"
        if (
            len(sorted_counts) > 1
            and sorted_counts[0][1] == sorted_counts[1][1]
        ):
            return "Unknown"
        return sorted_counts[0][0]

    @staticmethod
    def _check_tuple(
        parsed: dict,
        disc_set: set,
        candidate_classes: list[str],
        n_aligned: dict[str, int] | None = None,
        baseline_class: str = "Unknown",
    ) -> tuple[bool, dict, list[str]]:
        """Rule-based validation of the parsed tuple JSON.

        Returns ``(ok, possibly-mutated-parsed, issues)``. The most
        important check is that the LLM's ``label`` matches the
        deterministic baseline class. Overrides are accepted only
        when the cited drivers clearly and unanimously contradict the
        baseline (handled by clamping the label in
        ``_finalize_tuple``, not by failing here).
        """
        issues: list[str] = []
        n_aligned = n_aligned or {}
        candidate_set = set(candidate_classes)
        valid_labels = candidate_set | {"Unknown"}

        label = parsed.get("label")
        if label not in valid_labels:
            issues.append(
                f"`label` must be one of {sorted(valid_labels)!r}; "
                f"got {label!r}."
            )

        try:
            conf = float(parsed.get("confidence", 0.5))
            parsed["confidence"] = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            parsed["confidence"] = 0.5

        drivers = parsed.get("key_drivers", [])
        if not isinstance(drivers, list) or len(drivers) < 2:
            issues.append(
                "`key_drivers` must be a list of at least 2 entries."
            )
        else:
            seen = set()
            cleaned = []
            for d in drivers:
                if not isinstance(d, dict):
                    continue
                fn = d.get("feature")
                if fn not in disc_set:
                    issues.append(
                        f"`key_drivers` cites '{fn}' which is not a "
                        "discriminative feature in the table."
                    )
                    continue
                if fn in seen:
                    continue
                seen.add(fn)
                fav = d.get("favored_class")
                if fav not in candidate_set:
                    issues.append(
                        f"`key_drivers[{fn}].favored_class` must be one "
                        f"of {sorted(candidate_set)!r}; got {fav!r}."
                    )
                    continue
                try:
                    s = float(d.get("strength", 0.5))
                    d["strength"] = max(0.0, min(1.0, s))
                except (TypeError, ValueError):
                    d["strength"] = 0.5
                d.setdefault("why", "")
                cleaned.append(d)
            parsed["key_drivers"] = cleaned
            if len(cleaned) < 2 and not issues:
                issues.append(
                    "Need at least 2 valid driver entries; please retry "
                    "and only cite features that exist in the table "
                    "with valid favored_class values."
                )

        for k in ("supporting_features", "opposing_features"):
            v = parsed.get(k, [])
            if not isinstance(v, list):
                v = []
            parsed[k] = [x for x in v if isinstance(x, str)]

        for k in ("summary", "counter_evidence", "uncertainty"):
            if not isinstance(parsed.get(k), str):
                parsed[k] = ""

        # Soft baseline check — IMPORTANT: this is an *optimisation
        # gate*, not a permission gate.
        #
        # The witness ALWAYS wins. ``_finalize_tuple`` (called next)
        # unconditionally clamps the per-tuple ``label`` to the
        # alignment-tally baseline whenever they disagree. That clamp
        # is the load-bearing safety net.
        #
        # What the soft check does: when the LLM's label disagrees
        # with the baseline AND its drivers don't decisively favour
        # the LLM's pick (sum of strengths in the LLM's direction
        # ≤ sum of strengths in the baseline's direction + 1.0), we
        # add a ``CHECK FAILED`` issue and ask the LLM to retry. The
        # retry is the LLM's chance to either:
        #   (a) align its label with the baseline, OR
        #   (b) strengthen its drivers in support of its pick.
        # Either outcome leaves the per-tuple drivers consistent with
        # the verdict that ``_finalize_tuple`` will assign.
        #
        # Why we DON'T retry when ``override_strong`` is true: the
        # LLM has clearly committed to its choice with strong
        # rationale; further retries would just re-emit the same
        # output and burn LLM calls. We accept the strong-driver case
        # without retry — the clamp in ``_finalize_tuple`` will still
        # set the verdict to baseline, but at least we save the round
        # trip. The only cost is a per-tuple record where the cited
        # drivers favour a different class than the final clamped
        # ``label``; that is documented and downstream code (notably
        # the driver-attribution aggregation) treats the drivers as
        # the LLM's *raw rationale* rather than as committed votes
        # for the verdict.
        #
        # Threshold rationale (``+ 1.0``): a typical strength is
        # 0.5-0.9 per driver and there are 3-6 drivers per tuple, so
        # ``+ 1.0`` corresponds to "the LLM's drivers outweigh the
        # baseline-favouring drivers by roughly one full
        # high-confidence driver". Below this threshold we judge the
        # LLM's commitment too tentative to skip the retry.
        if (
            not issues
            and baseline_class in candidate_set
            and parsed.get("label") in candidate_set
            and parsed.get("label") != baseline_class
        ):
            drivers_clean = parsed["key_drivers"]
            same_dir = sum(
                d["strength"] for d in drivers_clean
                if d["favored_class"] == baseline_class
            )
            opp_dir = sum(
                d["strength"] for d in drivers_clean
                if d["favored_class"] == parsed["label"]
            )
            override_strong = opp_dir > same_dir + 1.0
            if not override_strong:
                tally_text = ", ".join(
                    f"{cls}: {n}" for cls, n in n_aligned.items()
                )
                issues.append(
                    f"The deterministic alignment tally ({tally_text}) "
                    f"identifies baseline = {baseline_class}, and your "
                    "selected drivers do not provide a decisive "
                    "counter-argument. Set `label` to "
                    f"'{baseline_class}' or substantially strengthen "
                    f"the drivers favouring '{parsed['label']}'."
                )

        return (len(issues) == 0), parsed, issues

    @staticmethod
    def _finalize_tuple(
        parsed: dict,
        ref_indices: dict[str, int],
        candidate_classes: list[str],
        disc_names: list[str],
        n_aligned: dict[str, int] | None = None,
        n_neutral: int = 0,
        baseline_class: str = "Unknown",
    ) -> dict:
        """Build the canonical tuple record.

        Two design commitments are baked in here:

        1. **``class_scores`` come from the deterministic alignment
           tally**, never from the LLM's driver-strength sums. This
           keeps per-tuple scoring reproducible and not biased by
           the LLM's narrative. The LLM's drivers are kept to power
           explanations and feature attribution; they do NOT
           influence which class wins.
        2. **``label`` is the alignment-tally baseline whenever
           there's a real candidate class to clamp to.** The LLM
           gets to PROPOSE a label inside the tuple's evidence
           record, but the per-tuple verdict is anchored by the
           witness. Even when ``_check_tuple`` accepted the LLM's
           "strong override", we clamp here — see the long comment
           in ``_check_tuple`` for why the soft check is an
           optimisation gate, not a permission gate.

           The witness ALWAYS wins. Reading either function's
           comments without the other has historically caused
           confusion; both are documented as one decision so the
           contract is unambiguous.
        """
        n_aligned = n_aligned or {}
        drivers = parsed["key_drivers"]
        total_aligned = sum(n_aligned.values())
        if total_aligned > 0:
            class_scores = {
                cls: round(n / total_aligned, 3)
                for cls, n in n_aligned.items()
            }
        else:
            # No discriminative votes anywhere → uniform per-class
            # score over the candidates. The verdict will be set
            # by the LLM (likely "Unknown") rather than by this
            # tally.
            eq = round(1.0 / len(candidate_classes), 3)
            class_scores = {cls: eq for cls in candidate_classes}

        label = parsed["label"]
        if (
            baseline_class in candidate_classes
            and label in candidate_classes
            and label != baseline_class
        ):
            # The witness wins. See the contract documented above
            # and in the soft-check comment in ``_check_tuple``.
            label = baseline_class

        confidence = float(parsed["confidence"])

        return {
            "candidate_classes": candidate_classes,
            "ref_indices": ref_indices,
            "label": label,
            "confidence": confidence,
            "class_scores": class_scores,
            "n_aligned": dict(n_aligned),
            "n_neutral": int(n_neutral),
            "baseline_class": baseline_class,
            "summary": parsed.get("summary", ""),
            "reason": parsed.get("summary", "")[:200],
            "discriminative_features": disc_names,
            "key_drivers": drivers,
            "supporting_features": parsed.get("supporting_features", []),
            "opposing_features": parsed.get("opposing_features", []),
            "counter_evidence": parsed.get("counter_evidence", ""),
            "uncertainty": parsed.get("uncertainty", ""),
        }

    # ── deterministic aggregation ────────────────────────────────────

    def _aggregate_results(
        self,
        tuple_results: list[dict],
        test_id,
        candidate_classes: list[str],
        n_degenerate: int = 0,
        class_reference_counts: dict[str, int] | None = None,
    ) -> dict:
        """Compute final class, confidence, and driver attribution
        without an extra LLM call.
        """
        class_reference_counts = class_reference_counts or {}
        class_votes = {cls: 0 for cls in candidate_classes}
        class_weights = {cls: 0.0 for cls in candidate_classes}
        n_unk = 0
        for tr in tuple_results:
            lbl = tr.get("label", "Unknown")
            conf = float(tr.get("confidence", 0.5) or 0.5)
            if lbl in class_votes:
                class_votes[lbl] += 1
                class_weights[lbl] += conf
            else:
                n_unk += 1

        n_total = len(tuple_results)
        n_decisive = sum(class_votes.values())

        if n_decisive == 0:
            # No tuple produced a decisive verdict — every tuple
            # returned ``Unknown``. This is the engine's
            # "I genuinely don't know" output; we never paper over
            # it with a coin-flip pick. See docs/design.md §18.1.
            final_label = "Inconclusive"
            margin = 0.0
        else:
            sorted_w = sorted(
                class_weights.items(), key=lambda kv: kv[1], reverse=True,
            )
            winner_cls, winner_w = sorted_w[0]
            runner_w = sorted_w[1][1] if len(sorted_w) > 1 else 0.0
            # ``0.05 * n_decisive`` is the **decisive-margin** rule:
            # the winner must outweigh the runner-up by more than
            # 5% of the decisive-tuple count. Without this bias a
            # single low-confidence vote could decide a tie. The
            # constant is small enough that any genuine majority
            # passes; large enough to refuse near-ties. Documented
            # in docs/design.md §18.1.
            threshold = 0.05 * max(n_decisive, 1)
            if winner_w > runner_w + threshold:
                final_label = winner_cls
                # Margin = relative winner-runner gap. The
                # ``max(..., 1e-9)`` guards a degenerate
                # all-zero-weights case (impossible after the
                # threshold check above, but defensive).
                denom = max(winner_w + runner_w, 1e-9)
                margin = (winner_w - runner_w) / denom
            else:
                # Decisive tuples exist but no candidate class has a
                # decisive lead — refuse to pick rather than break
                # the tie. The aggregator never invents a winner.
                final_label = "Inconclusive"
                margin = 0.0

        confidence, conf_basis = self._calibrate_confidence(
            margin=margin,
            n_decisive=n_decisive,
            n_total=n_total,
            n_unknown=n_unk,
            class_reference_counts=class_reference_counts,
            candidate_classes=candidate_classes,
            n_degenerate=n_degenerate,
            final_label=final_label,
        )

        attribution = self._compute_driver_attribution(
            tuple_results, self.feature_descriptions,
        )

        supporting_evidence, opposing_evidence = self._summarize_evidence(
            tuple_results, attribution, final_label, candidate_classes,
        )

        # Per-class top features — drivers most consistently cited per
        # class, capped at six entries each.
        key_features_per_class: dict[str, list[str]] = {
            cls: [] for cls in candidate_classes
        }
        for fa in attribution:
            dom = fa.get("dominant_class")
            if dom in key_features_per_class and fa.get("n_cited", 0) >= 1:
                key_features_per_class[dom].append(fa["feature"])
        for cls in key_features_per_class:
            key_features_per_class[cls] = key_features_per_class[cls][:6]

        votes_text = ", ".join(
            f"{cls}={n}" for cls, n in class_votes.items()
        )
        weighted_text = ", ".join(
            f"{cls}={w:.2f}" for cls, w in class_weights.items()
        )
        reason = (
            f"Vote tally: {votes_text}, Unknown={n_unk} "
            f"(weighted {weighted_text}); "
            f"{conf_basis['rationale']}"
        )
        uncertainty = (
            "No decisive tuple verdicts."
            if n_decisive == 0
            else f"{n_unk} of {n_total} tuple(s) were inconclusive; "
                 f"{n_degenerate} degenerate tuple(s) excluded."
        )

        logger.info(
            f"[AGG TEST {test_id}] {n_total} tuples  "
            f"votes={class_votes}  weighted={class_weights}  "
            f"Unk={n_unk}  => {final_label} @ {confidence:.0%}"
        )

        vote_summary = {
            "total": n_total,
            "per_class": {
                cls: {
                    "votes": int(class_votes[cls]),
                    "weighted": round(class_weights[cls], 3),
                }
                for cls in candidate_classes
            },
            "num_unknown": n_unk,
            "n_degenerate": n_degenerate,
        }

        return {
            "test_index": test_id,
            "task": "classification",
            "final_label": final_label,
            "confidence": round(confidence, 3),
            "confidence_basis": conf_basis,
            "reason": reason,
            "supporting_evidence": supporting_evidence,
            "opposing_evidence": opposing_evidence,
            "uncertainty": uncertainty,
            "key_features_per_class": key_features_per_class,
            "tuple_results": tuple_results,
            "driver_attribution": attribution,
            "vote_summary": vote_summary,
            "candidate_classes": list(candidate_classes),
        }

    @staticmethod
    def _calibrate_confidence(
        margin: float,
        n_decisive: int,
        n_total: int,
        n_unknown: int,
        class_reference_counts: dict[str, int],
        candidate_classes: list[str],
        n_degenerate: int,
        final_label: str,
    ) -> tuple[float, dict]:
        """Multi-class confidence: top-vs-runner-up margin × sample
        size × coverage. Same multiplicative form as the original
        binary calibration, generalised so the margin compares the
        winner against the strongest runner-up rather than Pos vs Neg.
        """
        if final_label == "Inconclusive":
            confidence = max(0.0, 0.4 - 0.2 * max(0.0, margin))
            return confidence, {
                "margin_score": round(margin, 3),
                "sample_size_score": 0.0,
                "coverage_score": 0.0,
                "rationale": (
                    "verdict is Inconclusive; confidence reflects "
                    "absence of decisive evidence."
                ),
                "basis_text": (
                    f"{n_decisive} decisive of {n_total} tuple "
                    "comparisons; verdict undecided."
                ),
            }

        margin_score = 0.5 + 0.5 * min(1.0, max(0.0, margin))
        cand_sizes = [
            class_reference_counts.get(c, 0) for c in candidate_classes
        ]
        ref_min = min(cand_sizes) if cand_sizes else 0
        sample_size_score = min(1.0, ref_min / 10.0)
        coverage_score = (
            n_decisive / max(n_total, 1) if n_total > 0 else 0.0
        )

        confidence = margin_score * (
            0.5 + 0.25 * sample_size_score + 0.25 * coverage_score
        )
        confidence = max(0.0, min(1.0, confidence))

        counts_text = ", ".join(
            f"{n} {c}" for c, n in class_reference_counts.items()
        )
        basis_text = (
            f"{n_decisive} of {n_total} tuple comparison(s) were "
            f"decisive (reference pool: {counts_text}"
            + (f"; {n_degenerate} degenerate excluded" if n_degenerate else "")
            + ")."
        )
        rationale_bits = [f"weighted margin {margin:.2f}"]
        if sample_size_score < 1.0:
            rationale_bits.append(
                f"smallest candidate-class size {ref_min} discounts "
                "confidence"
            )
        if coverage_score < 1.0:
            rationale_bits.append(
                f"{n_unknown} inconclusive tuple(s) discount confidence"
            )

        return confidence, {
            "margin_score": round(margin_score, 3),
            "sample_size_score": round(sample_size_score, 3),
            "coverage_score": round(coverage_score, 3),
            "rationale": "; ".join(rationale_bits),
            "basis_text": basis_text,
        }

    # ── driver attribution (cited drivers only, weighted) ───────────

    @staticmethod
    def _compute_driver_attribution(
        tuple_results: list[dict],
        feature_descriptions: dict[str, str] | None = None,
    ) -> list[dict]:
        """Aggregate features that were CITED AS DRIVERS by the
        per-tuple analyses, weighted by the LLM's stated strength.

        Crucially, this does NOT count features merely present in the
        discriminative set — only those the LLM picked out as drivers.
        """
        descs = feature_descriptions or {}
        per_feat: dict[str, dict] = defaultdict(
            lambda: {
                "feature": None,
                "n_cited": 0,
                "class_strength": defaultdict(float),
                "rationales": [],
            }
        )

        for tr in tuple_results:
            for d in tr.get("key_drivers", []):
                fn = d.get("feature")
                if not fn:
                    continue
                fav = d.get("favored_class")
                if fav is None:
                    continue
                slot = per_feat[fn]
                slot["feature"] = fn
                slot["n_cited"] += 1
                strength = float(d.get("strength", 0.5))
                slot["class_strength"][fav] += strength
                why = d.get("why", "").strip()
                if why:
                    slot["rationales"].append(why)

        attribution = []
        for fn, slot in per_feat.items():
            cs_dict = dict(slot["class_strength"])
            if cs_dict:
                sorted_cs = sorted(
                    cs_dict.items(), key=lambda kv: kv[1], reverse=True,
                )
                top_cls, top_strength = sorted_cs[0]
                runner_strength = (
                    sorted_cs[1][1] if len(sorted_cs) > 1 else 0.0
                )
                net = top_strength - runner_strength
                total = sum(cs_dict.values())
                dominant_class = top_cls if net > 0.05 else "Mixed"
            else:
                dominant_class = "Mixed"
                net = 0.0
                total = 0.0
            attribution.append({
                "feature": fn,
                "description": descs.get(fn, ""),
                "n_cited": slot["n_cited"],
                "class_strength": {
                    k: round(v, 3) for k, v in cs_dict.items()
                },
                "weighted_score": round(net, 3),
                "abs_strength": round(total, 3),
                "dominant_class": dominant_class,
                "rationales": slot["rationales"][:3],
            })

        attribution.sort(
            key=lambda f: (f["abs_strength"], f["n_cited"]),
            reverse=True,
        )
        return attribution

    @staticmethod
    def _summarize_evidence(
        tuple_results: list[dict],
        attribution: list[dict],
        final_label: str,
        candidate_classes: list[str],
    ) -> tuple[str, str]:
        """Build textual summaries of supporting (favouring final
        label) and opposing (favouring other classes) evidence.
        """
        if final_label not in candidate_classes:
            return (
                "Verdict is Inconclusive; no consistently supporting "
                "evidence emerged.",
                "Per-tuple verdicts were split or undecided; see vote "
                "tally and per-tuple records below.",
            )

        supporting = [
            fa for fa in attribution
            if fa.get("dominant_class") == final_label
        ]
        opposing = [
            fa for fa in attribution
            if fa.get("dominant_class") in candidate_classes
            and fa.get("dominant_class") != final_label
        ]

        def _sup_line(fa):
            r = fa["rationales"][0] if fa["rationales"] else ""
            dom = fa.get("dominant_class")
            return (
                f"{fa['feature']}: cited in {fa['n_cited']} tuple(s), "
                f"weighted strength {fa.get('weighted_score', 0):+.2f} "
                f"toward {dom}"
                + (f" — {r}" if r else "")
            )

        sup_lines = [_sup_line(fa) for fa in supporting[:5]]
        opp_lines = [_sup_line(fa) for fa in opposing[:5]]

        supporting_text = (
            "; ".join(sup_lines) if sup_lines
            else "No drivers were cited consistently in this direction."
        )
        opposing_text = (
            "; ".join(opp_lines) if opp_lines
            else "No drivers were cited that opposed the verdict."
        )
        return supporting_text, opposing_text

    # ── explanation (single Markdown call) ───────────────────────────

    def generate_explanation(
        self,
        result: dict,
        class_labels: list[str] | None = None,
        target_name: str = "target",
    ) -> str:
        """Produce a Markdown analysis report from an aggregated
        result. Branches on the result's ``task`` field; falls back
        to the agent's own ``self.task`` if absent.

        Classification: per-class reference statistics for top driver
        features so the LLM can cite concrete value ranges for every
        candidate class.
        Regression: a numeric report — predicted value, interval,
        per-feature direction (increases/decreases), with an honest
        treatment of refused predictions.
        """
        task = result.get("task", self.task)

        if task == "regression":
            messages = self._prompt_builder.build_regression_explanation_prompt(
                result=result,
                background=self.background,
                target_name=target_name,
            )
        else:
            attribution = result.get("driver_attribution", [])
            cls_list = (
                class_labels
                or result.get("candidate_classes")
                or self.class_labels
            )

            ref_stats: dict[str, dict] = {}
            if self._is_fitted:
                X = self._X_ref
                for fa in attribution[:15]:
                    fname = fa["feature"]
                    if fname not in self.feature_names:
                        continue
                    idx = self.feature_names.index(fname)
                    per_class_stats = {}
                    for cls in cls_list:
                        mask = self._y_ref == cls
                        vals = X[mask, idx]
                        if len(vals) == 0:
                            per_class_stats[cls] = {"mean": None, "range": None}
                            continue
                        per_class_stats[cls] = {
                            "mean": float(np.mean(vals)),
                            "range": (float(np.min(vals)), float(np.max(vals))),
                        }
                    ref_stats[fname] = per_class_stats

            messages = self._prompt_builder.build_explanation_prompt(
                result=result,
                background=self.background,
                class_labels=list(cls_list),
                ref_feature_stats=ref_stats,
            )

        tag = f"t{result.get('test_index', '?')}_EXPLAIN"
        text = self.llm.chat(
            messages, tag=tag, max_tokens=3500,
            temperature=self.llm_temperature,
        )
        logger.debug(f"[{tag}] raw explanation:\n{text}")

        text = self._strip_outer_code_fence(text).strip()
        logger.info(f"[{tag}] explanation generated ({len(text)} chars)")
        return text

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _strip_outer_code_fence(text: str) -> str:
        """Strip a single outer ``` … ``` wrapper, but preserve fenced
        blocks that are part of the content."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    return "\n".join(lines[1:i]).strip()
        return text

    @staticmethod
    def _fallback_tuple(
        ref_indices: dict[str, int],
        candidate_classes: list[str],
        error_msg: str,
    ) -> dict:
        return {
            "candidate_classes": list(candidate_classes),
            "ref_indices": ref_indices,
            "label": "Unknown",
            "confidence": 0.0,
            "class_scores": {
                c: round(1.0 / max(1, len(candidate_classes)), 3)
                for c in candidate_classes
            },
            "n_aligned": {c: 0 for c in candidate_classes},
            "n_neutral": 0,
            "baseline_class": "Unknown",
            "summary": "",
            "reason": f"LLM call failed: {error_msg}",
            "discriminative_features": [],
            "key_drivers": [],
            "supporting_features": [],
            "opposing_features": [],
            "counter_evidence": "",
            "uncertainty": "Tuple failed at the LLM-call layer.",
        }

    @staticmethod
    def _empty_result(test_id, reason) -> dict:
        return {
            "test_index": test_id,
            "task": "classification",
            "final_label": "Inconclusive",
            "confidence": 0.0,
            "confidence_basis": {
                "margin_score": 0.0,
                "sample_size_score": 0.0,
                "coverage_score": 0.0,
                "rationale": reason,
                "basis_text": reason,
            },
            "reason": reason,
            "supporting_evidence": "",
            "opposing_evidence": "",
            "uncertainty": reason,
            "tuple_results": [],
            "driver_attribution": [],
            "key_features_per_class": {},
            "vote_summary": {
                "total": 0,
                "per_class": {},
                "num_unknown": 0,
                "n_degenerate": 0,
            },
            "candidate_classes": [],
            "reference_info": {},
        }

