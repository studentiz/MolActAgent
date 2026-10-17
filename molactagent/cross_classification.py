"""MolActAgent's experimental regression backend — CCR.

This backend remains under development and is not evaluated in the
accompanying manuscript. It is shared for community exploration and
collaboration; reliability and applicability require further validation.
See ``docs/experimental-regression.md``.

CCR reformulates regression as TWO multi-class classification problems
whose predictions are intersected to pick a small set of reference
samples whose true target values are then combined into a numeric
verdict. This is the current experimental regression backend; MolActAgent's
``predict_one`` for ``task='regression'`` delegates here.

Pipeline per test sample:

  Phase 1 ("target view")   : equal-frequency bin the regression
      targets into N classes; classify the test sample with
      ``MolActAgent`` (top-M=2) over those bins.
  Phase 2 ("feature view")  : hierarchical / average-linkage cluster
      the standardised features under cosine distance into M
      clusters; classify the test sample with ``MolActAgent``
      (top-M=2) over those clusters.
  Selection                 : top ``k_per_class`` nearest references
      in each predicted class — both ranked by the project's
      contextual distance (``NeighborFinder``).
  Aggregation (default)     : confidence-weighted mean — each picked
      reference is weighted by its source phase's classification
      confidence; refs picked by both phases get summed weights.

Hyperparameters degrade gracefully with dataset size via the
``floor(sqrt(n)) + 1`` default for N (target bins) and M (feature
clusters). Top-M=2 in the inner classifier means the per-test-sample
LLM cost grows with ``k_per_class`` (default 2), not with N — so we
err on the side of more granular bins/clusters because that helped
on every benchmark we measured (BSLT n=16, penguins n=24). The
classification phases reuse MolActAgent's mature contrastive machinery
— so CCR is fully insulated from the per-tuple regression LLM
prompts the previous bootstrap-tuple backend used.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── tunable defaults ────────────────────────────────────────────────

DEFAULT_AGGREGATOR = "conf"
DEFAULT_K_PER_CLASS = 2
DEFAULT_AGENT_K = 2
DEFAULT_N_MODE = "sqrt"


# ── pure helpers (binning, clustering, hyperparam resolution) ──────

def equal_frequency_bins(
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, list[float]]:
    """Equal-frequency binning of a continuous target.

    Returns ``(bin_labels, bin_edges)`` where ``bin_labels`` is an
    array of strings ``"bin_0"..."bin_{N-1}"`` ordered by ascending
    target value (so ``bin_0`` is the lowest-value bin), and
    ``bin_edges`` is a list of cut points (length ``≤ n_bins+1``;
    duplicate edges are dropped when the target has many ties).
    """
    n_bins_eff = min(int(n_bins), int(np.unique(y).size))
    if n_bins_eff < 2:
        # Degenerate fallback — single class.
        return (
            np.array(["bin_0" for _ in y]),
            [float(np.min(y)), float(np.max(y))],
        )
    cats, edges = pd.qcut(
        y, q=n_bins_eff, labels=False, retbins=True, duplicates="drop",
    )
    if pd.isna(cats).any():
        # Fall back to simple linspace if qcut still left NaNs.
        edges_arr = np.linspace(float(y.min()), float(y.max()), n_bins_eff + 1)
        cats = np.digitize(y, edges_arr[1:-1])
    cats = np.asarray(cats, dtype=int)
    bin_labels = np.array([f"bin_{int(c)}" for c in cats])
    return bin_labels, [float(e) for e in edges]


def cosine_feature_clusters(
    X: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Hierarchical / average-linkage clusters on cosine distance.

    Standardises features first — raw cosine has no useful meaning
    when features live on wildly different scales (e.g. molecular
    descriptors mixed with binary fingerprint bits, or any tabular
    dataset with a mix of large-magnitude and unit-scale columns).
    Returns string labels ``"fc_0"..."fc_{M-1}"``.
    """
    if X.shape[0] < 2:
        return np.array(["fc_0" for _ in range(X.shape[0])])
    n_clusters_eff = max(1, min(int(n_clusters), X.shape[0] - 1))
    Xs = StandardScaler().fit_transform(X)
    cos_dist = pairwise_distances(Xs, metric="cosine")
    np.fill_diagonal(cos_dist, 0.0)
    cos_dist = np.clip(cos_dist, 0.0, None)
    if n_clusters_eff == 1:
        return np.array(["fc_0" for _ in range(X.shape[0])])
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters_eff,
        metric="precomputed",
        linkage="average",
    )
    cluster_ids = clusterer.fit_predict(cos_dist)
    return np.array([f"fc_{int(c)}" for c in cluster_ids])


def resolve_n_classes(
    n_full: int,
    mode: str = DEFAULT_N_MODE,
    fixed_value: int = 5,
) -> int:
    """Resolve N (or M) for a phase given the dataset size and mode.

    ``mode``:
      - ``"sqrt"`` (default): ``max(2, min(floor(sqrt(n_full)) + 1,
        n_full - 1))``. The "``+ 1``" is a small bias toward more
        granular bins/clusters because empirically (BSLT n=16,
        penguins n=24) more clusters consistently helped prediction
        without proportionally hurting per-phase classification
        cost — top-M=2 still caps each multi-class call to a
        contrast between the two closest classes regardless of how
        many classes exist in total.
      - ``"sqrt-floor"``: bare ``floor(sqrt(n_full))``, no "+1"
        adjustment. Slightly more conservative; useful for
        reproducing pre-deployment ablation runs.
      - ``"sqrt-capped-10"``: legacy default — ``floor(sqrt(n_full))``
        clamped to ``[2, 10]``. Kept for backward compatibility
        with any caller that pinned the old behaviour explicitly.
      - ``"fixed"``: return ``fixed_value`` clamped to
        ``[2, n_full-1]``.

    Why no upper cap by default? Even at ``n=10000`` the resolved
    N is ``101`` — a 101-class classifier sounds expensive, but the
    top-M=2 selector means each per-test-sample LLM call still only
    contrasts two classes, so the per-fold LLM cost grows with
    ``k_per_class`` (default 2), not with ``N``.
    """
    upper = max(2, n_full - 1)
    if mode == "fixed":
        return max(2, min(int(fixed_value), upper))
    floor_sqrt = int(n_full ** 0.5)
    if mode == "sqrt":
        return max(2, min(floor_sqrt + 1, upper))
    if mode == "sqrt-floor":
        return max(2, min(floor_sqrt, upper))
    if mode == "sqrt-capped-10":
        return max(2, min(floor_sqrt, 10, upper))
    raise ValueError(
        f"Unknown CCR n-mode {mode!r}; expected one of "
        "'sqrt', 'sqrt-floor', 'sqrt-capped-10', 'fixed'."
    )


# ── result-construction helpers ────────────────────────────────────

def _phase_winner_class(result: dict) -> str | None:
    """Pick the class to draw nearest-neighbours from.

    If MolActAgent returned a concrete ``final_label``, use it. If the
    verdict was ``"Inconclusive"`` / ``"Unknown"``, fall back to the
    candidate class with the largest weighted vote in the per-class
    summary. Returns ``None`` when no usable winner exists at all.
    """
    final = result.get("final_label")
    if final and final not in ("Inconclusive", "Unknown"):
        return str(final)
    vs = result.get("vote_summary", {}) or {}
    per_class = vs.get("per_class", {}) or {}
    best_cls = None
    best_w = -1.0
    for cls, info in per_class.items():
        w = float((info or {}).get("weighted", 0.0) or 0.0)
        if w > best_w:
            best_w = w
            best_cls = cls
    return best_cls


def _top_k_in_class(
    inner_agent: Any,
    x_test: np.ndarray,
    label_array: np.ndarray,
    target_class: str,
    k: int,
) -> list[tuple[int, float]]:
    """Top-``k`` references whose label == ``target_class``, ranked
    by the project's contextual distance.

    Reuses ``inner_agent``'s preprocessor + NeighborFinder so the
    distance is exactly the contextual blend (per docs/design.md).
    Returns a list of ``(ref_index, ctx_distance)`` tuples in
    ascending-distance order, length ``min(k, n_in_class)``. Empty
    when no references in the class survive.
    """
    x = np.asarray(x_test, dtype=np.float64)
    feature_mask = inner_agent._feature_mask
    if feature_mask is not None and x.shape[0] == len(feature_mask):
        x = x[feature_mask]
    x_scaled = inner_agent._preprocessor.transform(x.reshape(1, -1))[0]
    n = len(inner_agent._X_ref_scaled)
    indices, ctx_dists, _ = inner_agent._neighbor_finder.find_top_k_references(
        inner_agent._X_ref_scaled, x_scaled, k=n,
    )
    in_class: list[tuple[int, float]] = [
        (int(i), float(d))
        for i, d in zip(indices, ctx_dists)
        if str(label_array[int(i)]) == str(target_class)
    ]
    if not in_class:
        logger.debug(
            f"[CCR] no references survived the class filter "
            f"({target_class!r}); class may be empty in this fold."
        )
    return in_class[:k]


def _aggregate_simple(
    targets: list[float],
    distances: list[float],
    method: str,
) -> float:
    """Distance-only aggregators — ``mean`` / ``median`` / ``weighted``.

    Caller is responsible for refusing when ``targets`` is empty.
    """
    if not targets:
        return float("nan")
    arr = np.asarray(targets, dtype=float)
    if method == "median":
        return float(np.median(arr))
    if method == "weighted":
        d = np.asarray(distances, dtype=float)
        # 1 / (d + eps) maps short distances to large weights;
        # samples at distance 0 dominate, far samples decay smoothly.
        eps = 1e-6
        w = 1.0 / (d + eps)
        if not np.isfinite(w).any() or w.sum() <= 0:
            return float(np.mean(arr))
        return float(np.sum(arr * w) / np.sum(w))
    # Default: uniform mean.
    return float(np.mean(arr))


def _aggregate_per_phase(
    pt_neighbours: list[dict],
    pf_neighbours: list[dict],
    pt_conf: float,
    pf_conf: float,
    y_train: list[float],
    method: str,
    eps: float = 1e-6,
) -> float | None:
    """Confidence-aware aggregation that needs per-phase metadata.

    Refs picked by both phases get a SUMMED weight from each phase's
    contribution — both views agreeing on a ref counts as two votes
    of confidence rather than one.

    ``method`` in ``{"conf", "conf-distance"}``.
    """
    weights: dict[int, float] = {}

    def _accumulate(neighbours, phase_conf):
        for nb in neighbours or []:
            ri = nb.get("ref_idx")
            if ri is None:
                continue
            ri = int(ri)
            if not (0 <= ri < len(y_train)):
                continue
            d = float(nb.get("ctx_distance", 0.0) or 0.0)
            if method == "conf":
                w = float(phase_conf)
            elif method == "conf-distance":
                w = float(phase_conf) / (d + eps)
            else:
                raise ValueError(
                    f"Unknown CCR per-phase method {method!r}"
                )
            weights[ri] = weights.get(ri, 0.0) + w

    _accumulate(pt_neighbours, pt_conf)
    _accumulate(pf_neighbours, pf_conf)

    total_w = sum(weights.values())
    if total_w <= 0.0:
        return None
    num = sum(float(y_train[ri]) * w for ri, w in weights.items())
    return num / total_w


# ── driver attribution synthesis ───────────────────────────────────

def _synthesize_driver_attribution(
    result_t: dict,
    result_f: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str] | None,
    top_n: int = 12,
) -> list[dict]:
    """Combine the two phases' driver attributions into one
    regression-flavoured list.

    Each Phase produces a classification-style ``driver_attribution``
    list whose entries cite features that discriminated the winning
    class. We merge those, augment each entry with a regression-
    specific ``dominant_direction`` derived from the Pearson
    correlation between the feature and the continuous target across
    the training set, and order by the strength signal that the
    explanation report consumes (``abs_strength`` then ``n_cited``).
    """
    descs = feature_descriptions or {}
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    per_feat: dict[str, dict] = {}

    def _ingest(phase_label: str, phase_result: dict):
        for entry in phase_result.get("driver_attribution") or []:
            fn = entry.get("feature")
            if not fn:
                continue
            slot = per_feat.setdefault(fn, {
                "feature": fn,
                "n_cited": 0,
                "abs_strength": 0.0,
                "phases_citing": [],
                "rationales": [],
            })
            slot["n_cited"] += 1
            slot["abs_strength"] += float(
                entry.get("abs_strength")
                or abs(float(entry.get("weighted_score", 0.0)))
                or 0.0
            )
            slot["phases_citing"].append(phase_label)
            for r in entry.get("rationales") or []:
                if r and r not in slot["rationales"]:
                    slot["rationales"].append(r)

    _ingest("target_bin", result_t)
    _ingest("feature_cluster", result_f)

    if not per_feat:
        return []

    # Pre-compute per-feature Pearson correlation with the target
    # so we can give each cited driver a ``dominant_direction``.
    # This is the regression analogue of the classification
    # ``dominant_class`` field — we tell the LLM (and the report
    # renderer) "higher MolWt → higher predicted IC50" rather than
    # the more abstract "MolWt is a driver". The Pearson is a
    # property of the training set, not of any individual test
    # sample, so it's identical across samples in a batch (cheap to
    # compute once per fit).
    corr_by_feature: dict[str, float] = {}
    yt = np.asarray(y_train, dtype=float)
    if yt.size >= 3 and float(yt.std()) > 1e-9:
        for fn, slot in per_feat.items():
            idx = name_to_idx.get(fn)
            if idx is None or idx >= X_train.shape[1]:
                corr_by_feature[fn] = 0.0
                continue
            col = X_train[:, idx].astype(float)
            std_col = float(col.std())
            if std_col < 1e-9:
                corr_by_feature[fn] = 0.0
            else:
                # Manual Pearson to avoid making scipy a hard
                # dependency on this hot path. Equivalent to
                # ``scipy.stats.pearsonr(col, yt).statistic`` up to
                # FP precision.
                cov = float(np.mean(
                    (col - col.mean()) * (yt - yt.mean())
                ))
                corr_by_feature[fn] = cov / (std_col * float(yt.std()))
    else:
        corr_by_feature = {fn: 0.0 for fn in per_feat}

    attribution: list[dict] = []
    for fn, slot in per_feat.items():
        corr = corr_by_feature.get(fn, 0.0)
        # Direction thresholds at ±0.15:
        #
        # - We want a SMALL conservative floor that ignores
        #   weakly-correlated features instead of claiming a
        #   misleading direction (a near-zero Pearson is just
        #   noise on a small training set).
        # - The cutoff is symmetric around zero so the
        #   "increases" and "decreases" buckets get identical
        #   treatment.
        # - 0.15 is small enough that any real monotone trend
        #   passes; large enough that pure noise on n=20-30
        #   typically does not. Empirically validated on the
        #   penguins n=24 and BSLT n=16 benchmarks.
        if corr > 0.15:
            dominant = "increases"
        elif corr < -0.15:
            dominant = "decreases"
        else:
            dominant = "mixed"
        attribution.append({
            "feature": fn,
            "description": descs.get(fn, ""),
            "n_cited": int(slot["n_cited"]),
            "weighted_score": round(float(corr), 3),
            "abs_strength": round(float(slot["abs_strength"]), 3),
            "dominant_direction": dominant,
            "phases_citing": list(slot["phases_citing"]),
            "feature_target_correlation": round(float(corr), 3),
            "rationales": list(slot["rationales"][:3]),
            # Direction-specific strengths kept for compatibility with
            # the legacy regression explanation-prompt schema; we set
            # only the dominant slot so the report's ranking signal
            # stays meaningful.
            "increases_strength": (
                round(slot["abs_strength"], 3) if dominant == "increases" else 0.0
            ),
            "decreases_strength": (
                round(slot["abs_strength"], 3) if dominant == "decreases" else 0.0
            ),
            "mixed_strength": (
                round(slot["abs_strength"], 3) if dominant == "mixed" else 0.0
            ),
        })

    attribution.sort(
        key=lambda f: (f["abs_strength"], f["n_cited"]),
        reverse=True,
    )
    return attribution[:top_n]


# ── main class ─────────────────────────────────────────────────────

class CrossClassificationRegressor:
    """The experimental MolActAgent regression backend.

    This backend remains under development and is not evaluated in the
    accompanying manuscript. It is shared for community exploration and
    collaboration; reliability and applicability require further
    validation. See ``docs/experimental-regression.md``.

    Composes two ``MolActAgent`` instances internally — one for the
    target-bin view, one for the feature-cluster view — runs them in
    sequence on each test sample, and aggregates their selections
    into a numeric estimate with a heuristic confidence score.

    Use directly when you want fine-grained control:

    .. code-block:: python

       from molactagent import CrossClassificationRegressor, LLMClient
       reg = CrossClassificationRegressor(
           llm_client=llm,
           feature_names=feature_names,
           feature_descriptions=feature_descriptions,
           background=background,
       ).fit(X_train, y_train)
       result = reg.predict_one(X_test[i], test_id=i)

    For the standard end-to-end flow, prefer ``MolActAgent.fit(..., y)``
    with a continuous-numeric ``y`` — ``MolActAgent`` auto-detects
    regression and delegates to this class.
    """

    def __init__(
        self,
        llm_client: Any,
        feature_names: list[str] | None = None,
        feature_descriptions: dict[str, str] | None = None,
        background: str | None = None,
        n_target_bins: int | None = None,
        n_feature_clusters: int | None = None,
        n_mode: str = DEFAULT_N_MODE,
        k_per_class: int = DEFAULT_K_PER_CLASS,
        agent_k: int = DEFAULT_AGENT_K,
        max_workers: int = 100,
        llm_temperature: float = 0.0,
        aggregator: str = DEFAULT_AGGREGATOR,
        random_state: int | None = None,
    ):
        self.llm = llm_client
        self.feature_names = list(feature_names) if feature_names else None
        self.feature_descriptions = (
            dict(feature_descriptions) if feature_descriptions else {}
        )
        self.background = background
        self.n_target_bins = n_target_bins
        self.n_feature_clusters = n_feature_clusters
        self.n_mode = n_mode
        self.k_per_class = int(k_per_class)
        self.agent_k = int(agent_k)
        self.max_workers = int(max_workers)
        self.llm_temperature = float(llm_temperature)
        self.aggregator = aggregator
        self.random_state = random_state

        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.target_bin_labels: np.ndarray | None = None
        self.feature_cluster_labels: np.ndarray | None = None
        self.bin_edges: list[float] | None = None
        self.target_range: tuple[float, float] | None = None
        self.agent_target: Any | None = None
        self.agent_feature: Any | None = None
        self._n_bins_used: int = 0
        self._n_clusters_used: int = 0
        self._is_fitted = False

    # ── internal helpers ─────────────────────────────────────────

    def _bin_summary_string(self) -> str:
        if self.bin_edges is None or self._y is None or self.target_bin_labels is None:
            return ""
        parts = []
        for lbl in sorted(set(self.target_bin_labels.tolist())):
            mask = self.target_bin_labels == lbl
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            lo = float(self._y[mask].min())
            hi = float(self._y[mask].max())
            med = float(np.median(self._y[mask]))
            parts.append(
                f"{lbl}: target ∈ [{lo:.3g}, {hi:.3g}], "
                f"median={med:.3g}, n={n_in}"
            )
        return "; ".join(parts)

    def _cluster_summary_string(self) -> str:
        if self.feature_cluster_labels is None or self._y is None:
            return ""
        parts = []
        for lbl in sorted(set(self.feature_cluster_labels.tolist())):
            mask = self.feature_cluster_labels == lbl
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            lo = float(self._y[mask].min())
            hi = float(self._y[mask].max())
            med = float(np.median(self._y[mask]))
            parts.append(
                f"{lbl}: target ∈ [{lo:.3g}, {hi:.3g}], "
                f"median={med:.3g}, n={n_in}"
            )
        return "; ".join(parts)

    # ── public API ───────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "CrossClassificationRegressor":
        """Fit the two inner MolActAgent classifiers (target-bin view
        + feature-cluster view) on a labelled regression reference
        set."""
        # Lazy import — both this module and agent.py live in the
        # same package and reference each other; the lazy form
        # breaks the circular import at module-load time.
        from .agent import MolActAgent

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]}"
            )
        if X.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 samples to fit CCR; got {X.shape[0]}."
            )

        self._X = X
        self._y = y
        self.target_range = (float(y.min()), float(y.max()))

        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        # Resolve per-phase class counts. The override knobs
        # (``n_target_bins`` / ``n_feature_clusters``) are honoured
        # when set; otherwise ``n_mode`` chooses the default.
        n_full = X.shape[0]
        if self.n_target_bins is None:
            n_bins = resolve_n_classes(n_full, mode=self.n_mode)
        else:
            n_bins = max(2, min(int(self.n_target_bins), max(2, n_full - 1)))
        if self.n_feature_clusters is None:
            n_clusters = resolve_n_classes(n_full, mode=self.n_mode)
        else:
            n_clusters = max(2, min(
                int(self.n_feature_clusters), max(2, n_full - 1),
            ))
        self._n_bins_used = n_bins
        self._n_clusters_used = n_clusters

        self.target_bin_labels, self.bin_edges = equal_frequency_bins(
            y, n_bins,
        )
        self.feature_cluster_labels = cosine_feature_clusters(X, n_clusters)

        n_unique_target = len(set(self.target_bin_labels.tolist()))
        n_unique_feature = len(set(self.feature_cluster_labels.tolist()))

        bg_target = (
            f"{self.background or ''}\n\n"
            f"Auxiliary classification task (CCR Phase 1 / target view): "
            f"the regression target has been split into "
            f"{n_unique_target} equal-frequency bins "
            f"({self._bin_summary_string()}). Predicting the test "
            f"sample's bin is one of two complementary views used to "
            f"triangulate its continuous target value; Phase 2 "
            f"separately classifies it by an unsupervised "
            f"feature-space cluster. After both phases vote, the "
            f"actual target values of the closest references in each "
            f"predicted class are weighted by classification "
            f"confidence and averaged."
        ).strip()
        bg_feature = (
            f"{self.background or ''}\n\n"
            f"Auxiliary classification task (CCR Phase 2 / feature "
            f"view): the reference set has been split into "
            f"{n_unique_feature} unsupervised feature-space clusters "
            f"via hierarchical / average-linkage clustering on "
            f"cosine distance over standardised features "
            f"({self._cluster_summary_string()}). Predicting the "
            f"test sample's cluster is one of two complementary "
            f"views used to triangulate its continuous target value; "
            f"Phase 1 separately classifies it by target-bin. After "
            f"both phases vote, the actual target values of the "
            f"closest references in each predicted class are "
            f"weighted by classification confidence and averaged."
        ).strip()

        agent_kwargs = dict(
            llm_client=self.llm,
            k=self.agent_k,
            metric="euclidean",
            feature_names=list(self.feature_names),
            feature_descriptions=dict(self.feature_descriptions),
            max_workers=self.max_workers,
            llm_temperature=self.llm_temperature,
            random_state=self.random_state,
        )

        self.agent_target = MolActAgent(
            background=bg_target, **agent_kwargs,
        ).fit(X, self.target_bin_labels, task="classification")

        self.agent_feature = MolActAgent(
            background=bg_feature, **agent_kwargs,
        ).fit(X, self.feature_cluster_labels, task="classification")

        self._is_fitted = True
        logger.info(
            f"CCR fitted: {n_full} samples, "
            f"{n_bins} target bins, {n_clusters} feature clusters, "
            f"k_per_class={self.k_per_class}, "
            f"aggregator={self.aggregator!r}"
        )
        return self

    def predict_one(
        self,
        x_test: np.ndarray,
        test_id: Any = None,
    ) -> dict:
        """Predict the continuous target for a single test sample.

        Returns a result dict whose schema is backward-compatible
        with the legacy bootstrap-tuple regression backend (so
        ``molactagent.pipeline``, ``molactagent.report``, and existing
        downstream callers keep working) plus CCR-specific fields:
        ``method``, ``phase_target``, ``phase_feature``,
        ``selected_indices``, ``selected_targets``,
        ``selected_distances``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CrossClassificationRegressor is not fitted; "
                "call fit() first."
            )

        # Phase 1: target-bin classification.
        result_t = self.agent_target.predict_one(x_test, test_id=test_id)
        winner_t = _phase_winner_class(result_t)
        nn_t: list[tuple[int, float]] = (
            _top_k_in_class(
                self.agent_target, x_test, self.target_bin_labels,
                winner_t, self.k_per_class,
            )
            if winner_t is not None else []
        )

        # Phase 2: feature-cluster classification.
        result_f = self.agent_feature.predict_one(x_test, test_id=test_id)
        winner_f = _phase_winner_class(result_f)
        nn_f: list[tuple[int, float]] = (
            _top_k_in_class(
                self.agent_feature, x_test, self.feature_cluster_labels,
                winner_f, self.k_per_class,
            )
            if winner_f is not None else []
        )

        # Dedupe by reference index, keeping the smallest
        # contextual distance.
        #
        # WHY dedupe: a reference picked by BOTH phases is still
        # ONE piece of evidence (the same row of the training
        # set). Counting it twice in the deduped list would
        # double-count its target value when we aggregate.
        #
        # WHY keep the smallest distance: when CCR phases pick
        # the same reference at different distances (one phase
        # ranked it 1st, the other phase ranked it 5th), we keep
        # the closer of the two as the "best" estimate of how
        # similar that reference really is. The distance is only
        # used for the distance-weighted aggregator variants;
        # the default ``conf`` aggregator looks at per-phase
        # confidences instead.
        #
        # The "vote of confidence from both views" effect for
        # double-picked references is preserved separately in the
        # per-phase aggregator (`_aggregate_per_phase`), which
        # SUMS weights across phases without depending on this
        # dedupe step.
        merged: dict[int, float] = {}
        for idx, dist in nn_t + nn_f:
            if idx not in merged or dist < merged[idx]:
                merged[idx] = dist

        if not merged:
            return self._build_refusal(
                test_id, result_t, result_f, winner_t, winner_f,
                reason="Both phases produced no usable winner class.",
            )

        sel_indices = sorted(merged.keys())
        sel_targets = [float(self._y[i]) for i in sel_indices]
        sel_distances = [float(merged[i]) for i in sel_indices]

        # Aggregator dispatch — confidence-aware aggregators need
        # per-phase metadata; simple ones just need the deduped
        # flat list.
        if self.aggregator in ("conf", "conf-distance"):
            pt_nbrs = [
                {"ref_idx": int(i), "ctx_distance": float(d)}
                for i, d in nn_t
            ]
            pf_nbrs = [
                {"ref_idx": int(i), "ctx_distance": float(d)}
                for i, d in nn_f
            ]
            pred_value = _aggregate_per_phase(
                pt_neighbours=pt_nbrs,
                pf_neighbours=pf_nbrs,
                pt_conf=float(result_t.get("confidence", 0.0) or 0.0),
                pf_conf=float(result_f.get("confidence", 0.0) or 0.0),
                y_train=[float(v) for v in self._y.tolist()],
                method=self.aggregator,
            )
            if pred_value is None:
                return self._build_refusal(
                    test_id, result_t, result_f, winner_t, winner_f,
                    reason=(
                        "Confidence-weighted aggregation produced "
                        "no usable weight; refusing."
                    ),
                )
            prediction = float(pred_value)
        else:
            prediction = _aggregate_simple(
                sel_targets, sel_distances, self.aggregator,
            )
            if not np.isfinite(prediction):
                return self._build_refusal(
                    test_id, result_t, result_f, winner_t, winner_f,
                    reason=(
                        f"Aggregator {self.aggregator!r} returned "
                        "non-finite value."
                    ),
                )

        return self._build_result(
            test_id=test_id,
            x_test=x_test,
            prediction=prediction,
            sel_indices=sel_indices,
            sel_targets=sel_targets,
            sel_distances=sel_distances,
            nn_t=nn_t,
            nn_f=nn_f,
            result_t=result_t,
            result_f=result_f,
            winner_t=winner_t,
            winner_f=winner_f,
        )

    # ── result construction (legacy-schema-compatible) ──────────

    def _build_result(
        self,
        *,
        test_id: Any,
        x_test: np.ndarray,
        prediction: float,
        sel_indices: list[int],
        sel_targets: list[float],
        sel_distances: list[float],
        nn_t: list[tuple[int, float]],
        nn_f: list[tuple[int, float]],
        result_t: dict,
        result_f: dict,
        winner_t: str | None,
        winner_f: str | None,
    ) -> dict:
        target_min, target_max = self.target_range or (
            float("nan"), float("nan"),
        )

        # Honest empirical interval: tight bound from the picked refs.
        if sel_targets:
            lo = float(min(sel_targets))
            hi = float(max(sel_targets))
        else:
            lo = hi = float(prediction)

        # Confidence: mean of the phases that produced a winner. If
        # one phase refused, averaging in 0 would artificially halve
        # the surviving phase's confidence.
        present_confs: list[float] = []
        if winner_t is not None:
            present_confs.append(
                float(result_t.get("confidence", 0.0) or 0.0)
            )
        if winner_f is not None:
            present_confs.append(
                float(result_f.get("confidence", 0.0) or 0.0)
            )
        confidence = float(np.mean(present_confs)) if present_confs else 0.0

        # Extrapolation flag: prediction outside the training-target
        # range is a clear signal we're stretching beyond evidence.
        extrap = bool(
            prediction < target_min - 1e-9 or prediction > target_max + 1e-9
        )

        # phase-views-agree: did at least one ref get picked by BOTH
        # phases? Useful diagnostic surfaced in the result.
        agree = bool(
            winner_t is not None and winner_f is not None
            and len(
                {ix for ix, _ in nn_t} & {ix for ix, _ in nn_f}
            ) > 0
        )

        # Synthesize a regression-flavoured driver_attribution from
        # the inner MolActAgents' per-phase driver lists. Direction
        # ("increases" / "decreases" / "mixed") is set by Pearson
        # correlation between the feature and the continuous target
        # across the training set.
        attribution = _synthesize_driver_attribution(
            result_t=result_t,
            result_f=result_f,
            X_train=self._X if self._X is not None else np.zeros((0, 0)),
            y_train=self._y if self._y is not None else np.zeros((0,)),
            feature_names=self.feature_names or [],
            feature_descriptions=self.feature_descriptions,
        )

        # Turn each phase into a one-row "tuple_results" entry so the
        # legacy explanation prompt has something to render under
        # "Per-tuple Reasoning Records". Each row carries the phase's
        # mean target across its picks, the empirical interval over
        # those picks, and the phase's drivers (down-coerced into the
        # legacy regression-driver shape).
        tuple_results = self._build_phase_tuples(
            result_t, result_f, nn_t, nn_f, winner_t, winner_f,
        )

        # Build a confidence_basis dict the legacy explanation prompt
        # can quote verbatim.
        basis_text_parts: list[str] = []
        if winner_t is not None:
            basis_text_parts.append(
                f"Phase 1 (target bin) → {winner_t} "
                f"@ {float(result_t.get('confidence', 0.0)):.0%}"
            )
        if winner_f is not None:
            basis_text_parts.append(
                f"Phase 2 (feature cluster) → {winner_f} "
                f"@ {float(result_f.get('confidence', 0.0)):.0%}"
            )
        confidence_basis = {
            "agreement_score": (
                round(min(1.0, len(sel_indices) / max(2, 2 * self.k_per_class)), 3)
                if sel_indices else 0.0
            ),
            "phase_target_confidence": round(
                float(result_t.get("confidence", 0.0) or 0.0), 3
            ),
            "phase_feature_confidence": round(
                float(result_f.get("confidence", 0.0) or 0.0), 3
            ),
            "n_selected_refs": int(len(sel_indices)),
            "phases_agree_on_a_ref": bool(agree),
            "rationale": (
                f"Confidence-weighted ({self.aggregator}) average "
                f"of {len(sel_indices)} reference target values "
                "selected by the two cross-classification phases; "
                "each phase's classification confidence drives the "
                "weight."
                if self.aggregator in ("conf", "conf-distance")
                else (
                    f"{self.aggregator} of {len(sel_indices)} "
                    "reference target values selected by the two "
                    "cross-classification phases."
                )
            ),
            "basis_text": "; ".join(basis_text_parts) or (
                "No phase produced a winning class."
            ),
        }

        reference_info = {
            "total_reference_samples": int(len(self._y) if self._y is not None else 0),
            "n_comparisons": 2,
            "target_range": [float(target_min), float(target_max)],
            "n_target_bins": int(self._n_bins_used),
            "n_feature_clusters": int(self._n_clusters_used),
            "selected_ref_indices": [int(i) for i in sel_indices],
            "phase_target_winner": winner_t,
            "phase_feature_winner": winner_f,
            "phase_target_candidates": list(
                result_t.get("candidate_classes", []) or []
            ),
            "phase_feature_candidates": list(
                result_f.get("candidate_classes", []) or []
            ),
        }

        # ``test_feature_values`` powers the "Sample Feature Profile"
        # block in the explanation report. Apply the inner agent's
        # variance-filter mask so the names and values line up with
        # what the LLM saw.
        x_arr = np.asarray(x_test, dtype=np.float64)
        feature_mask = (
            self.agent_target._feature_mask
            if self.agent_target is not None else None
        )
        feature_names_for_display = list(self.feature_names or [])
        if (
            feature_mask is not None
            and x_arr.shape[0] == len(feature_mask)
        ):
            x_arr = x_arr[feature_mask]
            feature_names_for_display = [
                n for n, keep in zip(feature_names_for_display, feature_mask)
                if keep
            ]
        test_feature_values = {
            name: float(val)
            for name, val in zip(feature_names_for_display, x_arr.tolist())
        }

        return {
            "test_index": test_id,
            "task": "regression",
            "method": "CCR",
            "predicted_value": float(prediction),
            "interval": (lo, hi),
            "confidence": round(float(confidence), 3),
            "confidence_basis": confidence_basis,
            "extrapolation_flagged": extrap,
            "predictions": [float(t) for t in sel_targets],
            "tuple_results": tuple_results,
            "driver_attribution": attribution,
            "reference_info": reference_info,
            "test_feature_values": test_feature_values,
            "phase_target": {
                "predicted_class": winner_t,
                "confidence": float(result_t.get("confidence", 0.0) or 0.0),
                "candidate_classes": list(
                    result_t.get("candidate_classes", []) or []
                ),
                "final_label": result_t.get("final_label"),
                "neighbours": [
                    {"ref_idx": int(i), "ctx_distance": float(d)}
                    for i, d in nn_t
                ],
            },
            "phase_feature": {
                "predicted_class": winner_f,
                "confidence": float(result_f.get("confidence", 0.0) or 0.0),
                "candidate_classes": list(
                    result_f.get("candidate_classes", []) or []
                ),
                "final_label": result_f.get("final_label"),
                "neighbours": [
                    {"ref_idx": int(i), "ctx_distance": float(d)}
                    for i, d in nn_f
                ],
            },
            "selected_indices": [int(i) for i in sel_indices],
            "selected_targets": [float(t) for t in sel_targets],
            "selected_distances": [float(d) for d in sel_distances],
            "aggregator": self.aggregator,
            "phase_views_agree": agree,
        }

    def _build_phase_tuples(
        self,
        result_t: dict,
        result_f: dict,
        nn_t: list[tuple[int, float]],
        nn_f: list[tuple[int, float]],
        winner_t: str | None,
        winner_f: str | None,
    ) -> list[dict]:
        """One legacy-shaped ``tuple_results`` entry per phase, so the
        existing explanation prompt has per-phase reasoning records
        to render under "Per-tuple Reasoning Records".
        """
        if self._y is None:
            return []
        rows: list[dict] = []

        def _phase_row(
            label: str,
            phase_result: dict,
            nbrs: list[tuple[int, float]],
            winner: str | None,
        ) -> dict | None:
            if not nbrs or winner is None:
                return None
            targets = [float(self._y[i]) for i, _ in nbrs]
            mean_t = float(np.mean(targets))
            lo = float(min(targets))
            hi = float(max(targets))
            drivers_in = phase_result.get("driver_attribution") or []
            drivers_out: list[dict] = []
            for d in drivers_in[:5]:
                fn = d.get("feature")
                if not fn:
                    continue
                why = ""
                rats = d.get("rationales") or []
                if rats:
                    why = str(rats[0])
                # Map dominant_class info to a regression-style
                # direction; lacking a direct signal here, default
                # to "mixed" — the synthesized cross-phase
                # ``driver_attribution`` carries the regression
                # direction proper.
                drivers_out.append({
                    "feature": fn,
                    "direction": "mixed",
                    "strength": float(
                        d.get("abs_strength")
                        or abs(float(d.get("weighted_score", 0.0)))
                        or 0.5
                    ),
                    "test_value": 0.0,
                    "why": why,
                })
            return {
                "phase": label,
                "predicted_value": mean_t,
                "interval": (lo, hi),
                "confidence": float(
                    phase_result.get("confidence", 0.0) or 0.0
                ),
                "extrapolation_flag": False,
                "drivers": drivers_out,
                "rationale": (
                    f"{label}: classified test sample as {winner!r} "
                    f"({len(nbrs)} nearest references in this class "
                    f"have target values {targets}; mean = "
                    f"{mean_t:.4g})."
                ),
                "uncertainty": "",
                "ref_indices": [int(i) for i, _ in nbrs],
            }

        for entry in (
            _phase_row("Phase 1 (target bin)", result_t, nn_t, winner_t),
            _phase_row("Phase 2 (feature cluster)", result_f, nn_f, winner_f),
        ):
            if entry is not None:
                rows.append(entry)
        return rows

    def _build_refusal(
        self,
        test_id: Any,
        result_t: dict,
        result_f: dict,
        winner_t: str | None,
        winner_f: str | None,
        *,
        reason: str,
    ) -> dict:
        """Refusal result with the same field shape as a normal
        result, so downstream code paths handle it uniformly."""
        return {
            "test_index": test_id,
            "task": "regression",
            "method": "CCR",
            "predicted_value": None,
            "interval": (None, None),
            "confidence": 0.0,
            "confidence_basis": {
                "agreement_score": 0.0,
                "phase_target_confidence": float(
                    result_t.get("confidence", 0.0) or 0.0
                ),
                "phase_feature_confidence": float(
                    result_f.get("confidence", 0.0) or 0.0
                ),
                "n_selected_refs": 0,
                "phases_agree_on_a_ref": False,
                "rationale": reason,
                "basis_text": reason,
            },
            "extrapolation_flagged": True,
            "predictions": [],
            "tuple_results": [],
            "driver_attribution": [],
            "reference_info": {
                "total_reference_samples": int(
                    len(self._y) if self._y is not None else 0
                ),
                "n_comparisons": 2,
                "target_range": [
                    float(self.target_range[0]) if self.target_range else 0.0,
                    float(self.target_range[1]) if self.target_range else 0.0,
                ],
                "n_target_bins": int(self._n_bins_used),
                "n_feature_clusters": int(self._n_clusters_used),
                "phase_target_winner": winner_t,
                "phase_feature_winner": winner_f,
            },
            "test_feature_values": {},
            "phase_target": {
                "predicted_class": winner_t,
                "confidence": float(result_t.get("confidence", 0.0) or 0.0),
                "candidate_classes": list(
                    result_t.get("candidate_classes", []) or []
                ),
                "final_label": result_t.get("final_label"),
                "neighbours": [],
            },
            "phase_feature": {
                "predicted_class": winner_f,
                "confidence": float(result_f.get("confidence", 0.0) or 0.0),
                "candidate_classes": list(
                    result_f.get("candidate_classes", []) or []
                ),
                "final_label": result_f.get("final_label"),
                "neighbours": [],
            },
            "selected_indices": [],
            "selected_targets": [],
            "selected_distances": [],
            "aggregator": self.aggregator,
            "phase_views_agree": False,
            "reason": reason,
        }
