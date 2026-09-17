"""LOOCV regression evaluation for BSLT_huigui.csv comparing
MolActAgent against ML baselines and a pure-LLM baseline.

The BSLT (brine-shrimp lethality) dataset is a small (n=16) cyclic
peptide / polyketide regression dataset where each row carries a
SMILES and a continuous ``standard_value`` (lethality readout —
lower = more toxic). For every sample i in the dataset we hold it
out, train on the remaining n-1 and predict on i. We then compare:

  - ML baselines: KNN-1/3/5, Linear, Ridge, Lasso, Random Forest,
    Gradient Boosting, XGBoost  (all on RDKit descriptors + MACCS
    fingerprints, standard-scaled per fold).
  - Pure LLM baseline: each fold passes the n-1 reference
    (descriptors + true value) plus the held-out test descriptors
    in ONE prompt and asks the LLM for a numeric prediction.
  - MolActAgent (regression mode): contextual neighbour selection +
    bootstrap tuple LLM calls, aggregated to a numeric verdict
    with an honest interval.

Reported metrics: MAE, RMSE, R^2, Pearson r, Spearman r,
Median Absolute Error, plus MolActAgent's interval coverage and
refusal counts.

Usage
-----
    conda activate molactagent_env
    python scripts/run_loocv_BSLT_regression.py \
        --csv data/BSLT_huigui.csv --k 5 \
        --llm-profile deepseek --fold-workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from molactagent import MolActAgent, LLMClient, setup_logging  # noqa: E402

from prepare_features import (  # noqa: E402
    DESCRIPTOR_DESCRIPTIONS,
    DESCRIPTOR_FUNCS,
    MACCS_KEY_DESCRIPTIONS,
    compute_descriptors,
    compute_maccs,
    smiles_to_mol,
)
from run_demo import load_llm_config  # noqa: E402
from run_loocv_metrics import (  # noqa: E402
    compute_regression_metrics,
    format_regression_metrics_table,
    run_molactagent_loocv_regression,
    run_ml_loocv_regression,
)

logger = logging.getLogger("loocv_BSLT_reg")


BACKGROUND_BSLT_REG = (
    "Task: regression on the brine-shrimp (Artemia salina) lethality "
    "assay (BSLT) for cyclic peptide and polyketide natural products. "
    "Target = standard_value: a continuous lethality readout where "
    "LOWER values mean MORE toxic / more potent (typical LC50-style "
    "concentration scale; numerically smaller = more bioactive in "
    "the assay). Range observed in the reference set is roughly "
    "0.1–25.\n"
    "Features: RDKit physicochemical descriptors and MACCS structural "
    "keys (binary substructure indicators). For BSLT activity, "
    "drivers tend to involve halogen substitution, polyphenolic "
    "scaffolds, hydrogen-bonding motifs and overall lipophilicity "
    "(MolLogP, TPSA, ring count, MACCS bits for chlorine / hydroxyl "
    "patterns)."
)


# ── feature build (regression: SMILES + standard_value) ────────────

def build_regression_dataset_from_csv(
    csv_path: pathlib.Path,
    smiles_col: str = "smiles",
    target_col: str = "standard_value",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a SMILES + numeric-target CSV and return X, y plus
    a metadata dict (feature_names, feature_descriptions, smiles).

    Mirrors ``prepare_features.build_dataset_from_dataframe`` but
    keeps a continuous-target column instead of a string class
    label, so it can be fed straight to ``MolActAgent.fit(...,
    task='regression')`` and to the ML regressors.
    """
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(
            f"CSV {csv_path.name!r} must contain a {smiles_col!r} column"
        )
    if target_col not in df.columns:
        raise ValueError(
            f"CSV {csv_path.name!r} must contain a {target_col!r} column"
        )

    nan_mask = df[target_col].isna() | df[smiles_col].isna()
    if nan_mask.any():
        bad = df.loc[nan_mask].index.tolist()
        raise ValueError(f"Missing values on rows {bad}")
    y_raw = df[target_col].astype(float).to_numpy()

    desc_rows: list[dict] = []
    maccs_rows: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, smi in enumerate(df[smiles_col]):
        try:
            mol = smiles_to_mol(smi)
        except ValueError as e:
            if verbose:
                print(f"  WARNING: skipping row {i} – {e}")
            continue
        desc_rows.append(compute_descriptors(mol))
        maccs_rows.append(compute_maccs(mol))
        valid_idx.append(i)

    if not valid_idx:
        raise ValueError("No valid SMILES rows after parsing")

    desc_names = list(DESCRIPTOR_FUNCS.keys())
    X_desc = np.array(
        [[row[n] for n in desc_names] for row in desc_rows],
        dtype=np.float64,
    )
    X_maccs = np.vstack(maccs_rows).astype(np.float64)
    y = y_raw[valid_idx]

    maccs_var = X_maccs.var(axis=0)
    keep_maccs = np.where(maccs_var > 0)[0]
    X_maccs = X_maccs[:, keep_maccs]
    maccs_names = [f"MACCS_{k}" for k in keep_maccs]

    X = np.hstack([X_desc, X_maccs])
    feature_names = desc_names + maccs_names

    feature_descriptions: dict[str, str] = {}
    for nm in desc_names:
        feature_descriptions[nm] = DESCRIPTOR_DESCRIPTIONS[nm]
    for k in keep_maccs:
        feature_descriptions[f"MACCS_{k}"] = MACCS_KEY_DESCRIPTIONS.get(
            int(k), f"MACCS structural key {k}"
        )

    smiles_list = [str(df[smiles_col].iloc[i]) for i in valid_idx]

    if verbose:
        print(
            f"  Dataset: {len(y)} rows  "
            f"({X.shape[1]} features = "
            f"{len(desc_names)} descriptors + {len(maccs_names)} MACCS)"
        )
        print(
            f"  Target range: [{y.min():.3g}, {y.max():.3g}], "
            f"mean={y.mean():.3g}, std={y.std():.3g}"
        )

    meta = {
        "feature_names": feature_names,
        "feature_descriptions": feature_descriptions,
        "smiles": smiles_list,
        "n_descriptors": len(desc_names),
        "n_maccs": len(maccs_names),
    }
    return X, y, meta


# ── pure-LLM regression baseline ───────────────────────────────────

_PURE_LLM_NUM_RE = re.compile(
    r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
)


def _extract_pure_llm_numeric(text: str) -> float | None:
    """Pick a numeric prediction from the LLM's free-form reply.

    Strategy:
      1. Prefer a value following ``predicted_value=`` /
         ``prediction:`` / ``answer:`` (case-insensitive).
      2. Fall back to the last number on the last non-empty line.
      3. Return None if no number can be parsed (counted as a refusal).
    """
    if not text:
        return None
    cleaned = text.strip()

    label_pat = re.compile(
        r"(?i)(?:predicted[_ ]?value|prediction|answer|standard[_ ]?value|y[_ ]?hat)"
        r"\s*[:=]\s*"
        r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    m = label_pat.search(cleaned)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    for ln in reversed(lines):
        nums = _PURE_LLM_NUM_RE.findall(ln)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                continue
    return None


def _format_pure_llm_reference_block(
    X_train: np.ndarray,
    y_train: np.ndarray,
    smiles_train: list[str],
    feature_names: list[str],
    n_features_to_show: int = 12,
) -> str:
    """Render the n-1 reference compounds as one short block per row.

    Shows SMILES + the first ``n_features_to_show`` interpretable
    descriptors (the same descriptors that drive ML baselines and
    MolActAgent's similarity computation, so it's a fair playing
    field for the LLM).
    """
    show_idx = list(range(min(n_features_to_show, X_train.shape[1])))
    out_lines: list[str] = []
    for i in range(len(y_train)):
        feats = ", ".join(
            f"{feature_names[j]}={X_train[i, j]:.3g}" for j in show_idx
        )
        out_lines.append(
            f"  ref{i + 1}: y={y_train[i]:.3g}  "
            f"SMILES={smiles_train[i]}  {feats}"
        )
    return "\n".join(out_lines)


def _pure_llm_one_fold(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    smiles_all: list[str],
    feature_names: list[str],
    background: str,
    llm: LLMClient,
    llm_temperature: float,
    progress_cb: Callable[[int, float, float | None, float], None] | None = None,
) -> dict:
    n = len(y)
    train_idx = [j for j in range(n) if j != fold_idx]
    X_train = X[train_idx]
    y_train = y[train_idx]
    smiles_train = [smiles_all[j] for j in train_idx]

    show_n = min(15, len(feature_names))
    show_features = ", ".join(
        f"{nm}={X[fold_idx, j]:.3g}"
        for j, nm in enumerate(feature_names[:show_n])
    )
    ref_block = _format_pure_llm_reference_block(
        X_train, y_train, smiles_train,
        feature_names, n_features_to_show=show_n,
    )

    prompt = (
        f"You are an expert medicinal-chemistry regression model.\n"
        f"Background:\n{background}\n\n"
        f"Reference set ({len(y_train)} compounds, true target values "
        f"shown alongside SMILES and the first {show_n} molecular "
        f"descriptors):\n{ref_block}\n\n"
        f"Reference target stats: min={float(y_train.min()):.3g}, "
        f"max={float(y_train.max()):.3g}, mean={float(y_train.mean()):.3g}, "
        f"median={float(np.median(y_train)):.3g}.\n\n"
        f"NEW test compound:\n"
        f"  SMILES: {smiles_all[fold_idx]}\n"
        f"  Features (first {show_n}): {show_features}\n\n"
        f"Predict the test compound's standard_value (a single real "
        f"number on the same scale as the references). Reason briefly "
        f"about the most chemically similar references and structural "
        f"drivers, then on the FINAL line output exactly:\n"
        f"  predicted_value=<number>\n"
        f"Do not include units or extra symbols on the final line."
    )

    t0 = time.time()
    try:
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=llm_temperature,
            tag=f"pureLLM_reg_{fold_idx}",
        )
    except Exception as e:
        logger.error(f"PureLLM fold {fold_idx} failed: {e}")
        resp = ""
    elapsed = time.time() - t0

    pred = _extract_pure_llm_numeric(resp)
    if progress_cb:
        progress_cb(fold_idx, float(y[fold_idx]), pred, elapsed)

    return {
        "fold": int(fold_idx),
        "true": float(y[fold_idx]),
        "pred": float(pred) if pred is not None and np.isfinite(pred) else None,
        "elapsed_s": round(elapsed, 1),
        "raw_response": resp,
    }


def run_pure_llm_loocv_regression(
    X: np.ndarray,
    y: np.ndarray,
    smiles_all: list[str],
    feature_names: list[str],
    background: str,
    llm: LLMClient,
    llm_temperature: float,
    fold_workers: int,
) -> tuple[dict, list[dict]]:
    n = len(y)
    fold_records: list[dict] = [None] * n  # type: ignore[list-item]
    completed = [0]

    def _progress(fi, true_v, pred_v, elapsed):
        completed[0] += 1
        pred_str = "REFUSED" if pred_v is None else f"{pred_v:.4g}"
        logger.info(
            f"  [PureLLM] fold {fi+1}/{n} done in {elapsed:.1f}s  "
            f"true={true_v:.4g}  pred={pred_str}  "
            f"({completed[0]}/{n} folds finished)"
        )

    if fold_workers > 1:
        with ThreadPoolExecutor(max_workers=fold_workers) as pool:
            futures = {
                pool.submit(
                    _pure_llm_one_fold, i, X, y, smiles_all,
                    feature_names, background, llm, llm_temperature,
                    _progress,
                ): i
                for i in range(n)
            }
            for fut in as_completed(futures):
                rec = fut.result()
                fold_records[rec["fold"]] = rec
    else:
        for i in range(n):
            fold_records[i] = _pure_llm_one_fold(
                i, X, y, smiles_all, feature_names,
                background, llm, llm_temperature, _progress,
            )

    y_true = [r["true"] for r in fold_records]
    y_pred = [r["pred"] for r in fold_records]
    metrics = compute_regression_metrics(y_true, y_pred, intervals=None)
    return metrics, fold_records


# ── Cross-Classification Regressor (CCR) ───────────────────────────
#
# Reformulates regression as TWO multi-class classification problems
# whose predictions are intersected to pick a small set of reference
# samples whose true target values are then averaged.
#
#   Phase 1 ("target view")   : equal-frequency bin the regression
#       targets into N classes; classify the test sample with
#       MolActAgent (top-M=2) over those bins.
#   Phase 2 ("feature view")  : hierarchical / average-linkage cluster
#       the standardised features under cosine distance into M
#       clusters; classify the test sample with MolActAgent (top-M=2)
#       over those clusters.
#   Aggregation               : take the top-2 nearest references in
#       Phase 1's predicted bin AND the top-2 nearest in Phase 2's
#       predicted cluster — both ranked by the project's CONTEXTUAL
#       distance (NeighborFinder) — dedupe by sample index, and
#       collapse to a single value with mean / median / 1/distance-
#       weighted mean.
#
# Hyperparameters are pre-registered (n_target_bins=5,
# n_feature_clusters=5, k_per_class=2) rather than tuned by silhouette;
# at n=16 silhouette is too noisy to trust. They become tunable once
# more data is available.


def _equal_frequency_bins(
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, list[float]]:
    """Equal-frequency target binning.

    Returns ``(bin_labels, bin_edges)`` where ``bin_labels`` is an
    array of strings ``"bin_0"..."bin_{N-1}"`` ordered by ascending
    target value (so ``bin_0`` is the lowest-value bin), and
    ``bin_edges`` is a length-``n_bins+1`` list of cut points.
    """
    n_bins_eff = min(n_bins, int(np.unique(y).size))
    if n_bins_eff < 2:
        # Degenerate fallback — single class.
        return (
            np.array([f"bin_0" for _ in y]),
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


def _cosine_feature_clusters(
    X: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Hierarchical / average-linkage clusters on cosine distance.

    Standardises features first (RDKit descriptors are wildly mixed
    scales, so raw cosine isn't meaningful), then clusters into
    ``n_clusters`` groups. Returns string labels
    ``"fc_0"..."fc_{M-1}"``.
    """
    if X.shape[0] < 2:
        return np.array([f"fc_0" for _ in range(X.shape[0])])
    n_clusters_eff = min(n_clusters, X.shape[0] - 1, X.shape[0])
    n_clusters_eff = max(n_clusters_eff, 1)
    Xs = StandardScaler().fit_transform(X)
    cos_dist = pairwise_distances(Xs, metric="cosine")
    np.fill_diagonal(cos_dist, 0.0)
    cos_dist = np.clip(cos_dist, 0.0, None)
    if n_clusters_eff == 1:
        return np.array([f"fc_0" for _ in range(X.shape[0])])
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters_eff,
        metric="precomputed",
        linkage="average",
    )
    cluster_ids = clusterer.fit_predict(cos_dist)
    return np.array([f"fc_{int(c)}" for c in cluster_ids])


def _leiden_feature_clusters(
    X: np.ndarray,
    knn: int = 4,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Community detection on a k-NN graph built from cosine similarity.

    Builds a symmetric k-NN graph where each node is a sample and an
    edge between i and j carries weight ``cosine_similarity(i, j)``
    (clamped non-negative). Runs Leiden community detection with
    the modularity vertex-partition objective. Resolution parameter
    > 1 yields smaller, more numerous communities; < 1 yields
    fewer, larger ones. Returns string labels
    ``"fc_0"..."fc_{M-1}"`` like the hierarchical baseline so the
    rest of CCR is agnostic to the clustering method.

    Lazy-imports leidenalg + python-igraph so the rest of the script
    runs even when those optional deps are not installed.
    """
    try:
        import igraph as ig  # type: ignore[import-untyped]
        import leidenalg  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "Leiden Phase 2 requires `leidenalg` and `python-igraph`; "
            "install with `pip install leidenalg python-igraph`."
        ) from e

    n = X.shape[0]
    if n < 2:
        return np.array([f"fc_0" for _ in range(n)])
    Xs = StandardScaler().fit_transform(X)
    sim = 1.0 - pairwise_distances(Xs, metric="cosine")
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, 1.0)

    # Symmetric k-NN graph: edge (i, j) exists if j is in i's top-k
    # OR i is in j's top-k. Captures local manifold structure
    # without the all-pairs density that confuses Leiden.
    k_eff = min(knn, n - 1)
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        # argsort descending — pick the top k_eff most similar (excluding self).
        top = np.argsort(-sim[i])
        added = 0
        for j in top:
            if int(j) == i:
                continue
            if added >= k_eff:
                break
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            w = float(sim[i, int(j)])
            if w <= 0.0:
                continue
            edges[(a, b)] = max(edges.get((a, b), 0.0), w)
            added += 1

    g = ig.Graph(n=n)
    if edges:
        g.add_edges(list(edges.keys()))
        g.es["weight"] = list(edges.values())

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if edges else None,
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    cluster_ids = np.asarray(partition.membership, dtype=int)
    return np.array([f"fc_{int(c)}" for c in cluster_ids])


def _phase_winner_class(result: dict) -> str | None:
    """Pick the class to draw nearest-neighbours from.

    If MolActAgent returned a concrete final_label, use it. If the
    verdict was ``"Inconclusive"``, fall back to the candidate class
    with the largest weighted vote in the per-class summary.
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
    agent: MolActAgent,
    x_test: np.ndarray,
    label_array: np.ndarray,
    target_class: str,
    k: int,
) -> list[tuple[int, float]]:
    """Top-k references whose label == ``target_class``, ranked by
    contextual distance to ``x_test``.

    Reuses ``agent``'s preprocessor + NeighborFinder so the distance
    is exactly the project's contextual blend (per docs/design.md).
    Returns a list of ``(ref_index, ctx_distance)`` tuples in
    ascending-distance order, length ``min(k, n_in_class)``.
    """
    x = np.asarray(x_test, dtype=np.float64)
    if agent._feature_mask is not None and x.shape[0] == len(agent._feature_mask):
        x = x[agent._feature_mask]
    x_scaled = agent._preprocessor.transform(x.reshape(1, -1))[0]
    n = len(agent._X_ref_scaled)
    indices, ctx_dists, _ = agent._neighbor_finder.find_top_k_references(
        agent._X_ref_scaled, x_scaled, k=n,
    )
    in_class: list[tuple[int, float]] = [
        (int(i), float(d))
        for i, d in zip(indices, ctx_dists)
        if str(label_array[int(i)]) == str(target_class)
    ]
    return in_class[:k]


def _aggregate_predictions(
    targets: list[float],
    distances: list[float],
    method: str,
) -> float:
    """Collapse a small set of reference targets into a single value.

    ``method`` in ``{"mean", "median", "weighted"}``. ``"weighted"``
    is a 1/(distance+eps)-weighted mean; ``"mean"`` is a uniform
    mean; ``"median"`` ignores distances entirely. Returns a finite
    float; the caller is responsible for refusing if no targets
    were supplied.
    """
    if not targets:
        return float("nan")
    arr = np.asarray(targets, dtype=float)
    if method == "median":
        return float(np.median(arr))
    if method == "weighted":
        d = np.asarray(distances, dtype=float)
        # Map distance -> weight via 1 / (d + eps); samples at
        # distance 0 dominate, far samples decay smoothly.
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
    contribution. ``method`` in ``{"conf", "conf-distance"}``.
    """
    weights: dict[int, float] = {}

    def _accumulate(neighbours, phase_conf):
        for nb in neighbours or []:
            ri = nb.get("ref_idx")
            if ri is None or not (0 <= int(ri) < len(y_train)):
                continue
            ri = int(ri)
            d = float(nb.get("ctx_distance", 0.0) or 0.0)
            if method == "conf":
                w = float(phase_conf)
            elif method == "conf-distance":
                w = float(phase_conf) / (d + eps)
            else:
                raise ValueError(f"Unknown per-phase method {method!r}")
            weights[ri] = weights.get(ri, 0.0) + w

    _accumulate(pt_neighbours, pt_conf)
    _accumulate(pf_neighbours, pf_conf)

    total_w = sum(weights.values())
    if total_w <= 0.0:
        return None
    num = sum(float(y_train[ri]) * w for ri, w in weights.items())
    return num / total_w


def _resolve_n_classes(n_train: int, mode: str, fixed: int) -> int:
    """Resolve N (or M) for a phase given the data size and mode.

    ``mode``:
      - ``"fixed"``: return ``fixed`` clamped to ``[2, n_train-1]``.
      - ``"sqrt"`` : return ``floor(sqrt(n_train))`` clamped to
                     ``[2, min(10, n_train-1)]``. Standard ML default
                     (KNN / RF / histogram binning use the same
                     heuristic) for the same bias-variance reason —
                     at sqrt(n) you balance "too few classes loses
                     signal" against "too many leaves classes too
                     small to estimate stably". The upper cap of 10
                     keeps top-M=2 manageable on large datasets.
      - ``"sqrt-uncapped"`` : same but without the ``min(10, ...)``
                              upper cap (use only when you really
                              want very many classes on big n).
    """
    if mode == "fixed":
        return max(2, min(int(fixed), max(2, n_train - 1)))
    floor_sqrt = int(n_train ** 0.5)
    if mode == "sqrt":
        return max(2, min(floor_sqrt, 10, max(2, n_train - 1)))
    if mode == "sqrt-uncapped":
        return max(2, min(floor_sqrt, max(2, n_train - 1)))
    raise ValueError(f"Unknown n-mode {mode!r}")


class CrossClassificationRegressor:
    """Regression-via-cross-classification on two label views."""

    def __init__(
        self,
        llm_client: LLMClient,
        feature_names: list[str],
        feature_descriptions: dict[str, str],
        background: str,
        n_target_bins: int = 5,
        n_feature_clusters: int = 5,
        k_per_class: int = 2,
        agent_k: int = 2,
        max_workers: int = 20,
        llm_temperature: float = 0.0,
        aggregator: str = "mean",
        cluster_method: str = "hierarchical",
        leiden_knn: int = 4,
        leiden_resolution: float = 1.0,
    ):
        self.llm = llm_client
        self.feature_names = list(feature_names)
        self.feature_descriptions = dict(feature_descriptions)
        self.background = background
        self.n_target_bins = n_target_bins
        self.n_feature_clusters = n_feature_clusters
        self.k_per_class = k_per_class
        self.agent_k = agent_k
        self.max_workers = max_workers
        self.llm_temperature = llm_temperature
        self.aggregator = aggregator
        self.cluster_method = cluster_method
        self.leiden_knn = leiden_knn
        self.leiden_resolution = leiden_resolution

        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.target_bin_labels: np.ndarray | None = None
        self.feature_cluster_labels: np.ndarray | None = None
        self.bin_edges: list[float] | None = None
        self.agent_target: MolActAgent | None = None
        self.agent_feature: MolActAgent | None = None

    def _bin_summary_string(self) -> str:
        """A short human-readable description of the target bins."""
        if self.bin_edges is None or self._y is None or self.target_bin_labels is None:
            return ""
        parts = []
        unique_bins = sorted(set(self.target_bin_labels.tolist()))
        for lbl in unique_bins:
            mask = self.target_bin_labels == lbl
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            lo = float(self._y[mask].min())
            hi = float(self._y[mask].max())
            med = float(np.median(self._y[mask]))
            parts.append(
                f"{lbl}: target ∈ [{lo:.3g}, {hi:.3g}], median={med:.3g}, n={n_in}"
            )
        return "; ".join(parts)

    def _cluster_summary_string(self) -> str:
        if self.feature_cluster_labels is None or self._y is None:
            return ""
        parts = []
        unique = sorted(set(self.feature_cluster_labels.tolist()))
        for lbl in unique:
            mask = self.feature_cluster_labels == lbl
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            lo = float(self._y[mask].min())
            hi = float(self._y[mask].max())
            med = float(np.median(self._y[mask]))
            parts.append(
                f"{lbl}: target ∈ [{lo:.3g}, {hi:.3g}] median={med:.3g}, n={n_in}"
            )
        return "; ".join(parts)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CrossClassificationRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]}"
            )
        self._X = X
        self._y = y

        self.target_bin_labels, self.bin_edges = _equal_frequency_bins(
            y, self.n_target_bins,
        )
        if self.cluster_method == "leiden":
            self.feature_cluster_labels = _leiden_feature_clusters(
                X,
                knn=self.leiden_knn,
                resolution=self.leiden_resolution,
            )
        else:
            self.feature_cluster_labels = _cosine_feature_clusters(
                X, self.n_feature_clusters,
            )

        n_unique_target = len(set(self.target_bin_labels.tolist()))
        n_unique_feature = len(set(self.feature_cluster_labels.tolist()))

        bg_target = (
            f"{self.background}\n\n"
            f"Auxiliary classification task (Phase 1 / target view): "
            f"the regression target has been split into "
            f"{n_unique_target} equal-frequency bins ({self._bin_summary_string()}). "
            f"Predicting the test sample's bin is one of two "
            f"complementary views used to triangulate its continuous "
            f"target value; Phase 2 separately classifies it by "
            f"feature-space cluster. After both phases vote, we "
            f"average the actual target values of the closest "
            f"references in each predicted class."
        )
        # Describe the actual clustering algorithm used so the LLM
        # isn't given a misleading description of how the cluster
        # labels were formed. The previous version unconditionally
        # said "hierarchical / average-linkage" even when Leiden
        # was active.
        if self.cluster_method == "leiden":
            cluster_descr = (
                f"Leiden community detection on a k-NN graph "
                f"(k={self.leiden_knn}, "
                f"resolution={self.leiden_resolution}) "
                "built from cosine similarity over standardised "
                "features"
            )
        else:
            cluster_descr = (
                "hierarchical / average-linkage clustering on cosine "
                "distance over standardised features"
            )
        bg_feature = (
            f"{self.background}\n\n"
            f"Auxiliary classification task (Phase 2 / feature view): "
            f"the reference set has been split into {n_unique_feature} "
            f"unsupervised feature-space clusters via "
            f"{cluster_descr} "
            f"({self._cluster_summary_string()}). Predicting the "
            f"test sample's cluster is one of two complementary "
            f"views used to triangulate its continuous target "
            f"value; Phase 1 separately classifies it by target-bin. "
            f"After both phases vote, we average the actual target "
            f"values of the closest references in each predicted "
            f"class."
        )

        self.agent_target = MolActAgent(
            llm_client=self.llm,
            k=self.agent_k,
            metric="euclidean",
            feature_names=self.feature_names,
            feature_descriptions=self.feature_descriptions,
            background=bg_target,
            max_workers=self.max_workers,
            llm_temperature=self.llm_temperature,
        ).fit(X, self.target_bin_labels, task="classification")

        self.agent_feature = MolActAgent(
            llm_client=self.llm,
            k=self.agent_k,
            metric="euclidean",
            feature_names=self.feature_names,
            feature_descriptions=self.feature_descriptions,
            background=bg_feature,
            max_workers=self.max_workers,
            llm_temperature=self.llm_temperature,
        ).fit(X, self.feature_cluster_labels, task="classification")

        return self

    def predict_one(
        self,
        x_test: np.ndarray,
        test_id=None,
    ) -> dict:
        if (
            self.agent_target is None
            or self.agent_feature is None
            or self.target_bin_labels is None
            or self.feature_cluster_labels is None
            or self._y is None
        ):
            raise RuntimeError("CCR is not fitted. Call fit() first.")

        result_t = self.agent_target.predict_one(x_test, test_id=test_id)
        winner_t = _phase_winner_class(result_t)
        nn_t: list[tuple[int, float]] = (
            _top_k_in_class(
                self.agent_target, x_test, self.target_bin_labels,
                winner_t, self.k_per_class,
            )
            if winner_t is not None else []
        )

        result_f = self.agent_feature.predict_one(x_test, test_id=test_id)
        winner_f = _phase_winner_class(result_f)
        nn_f: list[tuple[int, float]] = (
            _top_k_in_class(
                self.agent_feature, x_test, self.feature_cluster_labels,
                winner_f, self.k_per_class,
            )
            if winner_f is not None else []
        )

        # Dedupe by reference index — a sample picked in BOTH views
        # is one piece of evidence, not two — and keep the smaller
        # contextual distance for weighting purposes.
        merged: dict[int, float] = {}
        for idx, dist in nn_t + nn_f:
            if idx not in merged or dist < merged[idx]:
                merged[idx] = dist

        if not merged:
            # Both phases failed to nominate a usable class — refuse.
            return {
                "test_index": test_id,
                "task": "regression",
                "method": "CCR",
                "predicted_value": None,
                "phase_target": {
                    "predicted_class": winner_t,
                    "confidence": float(result_t.get("confidence", 0.0) or 0.0),
                    "candidate_classes": result_t.get("candidate_classes", []),
                    "final_label": result_t.get("final_label"),
                },
                "phase_feature": {
                    "predicted_class": winner_f,
                    "confidence": float(result_f.get("confidence", 0.0) or 0.0),
                    "candidate_classes": result_f.get("candidate_classes", []),
                    "final_label": result_f.get("final_label"),
                },
                "selected_indices": [],
                "selected_targets": [],
                "selected_distances": [],
                "aggregator": self.aggregator,
                "reason": "Both phases produced no usable winner class.",
            }

        sel_indices = sorted(merged.keys())
        sel_targets = [float(self._y[i]) for i in sel_indices]
        sel_distances = [float(merged[i]) for i in sel_indices]

        # Aggregator dispatch:
        #   * Per-phase aggregators (``conf`` / ``conf-distance``) need
        #     per-phase neighbour metadata + each phase's confidence —
        #     route to ``_aggregate_per_phase``.
        #   * Simple aggregators (``mean`` / ``median`` / ``weighted``)
        #     only need the deduped flat ref list — route to
        #     ``_aggregate_predictions``.
        # The previous version unconditionally called
        # ``_aggregate_predictions``; for ``conf`` / ``conf-distance``
        # that fell through to the default uniform-mean branch and
        # silently produced wrong results when ``predict_one`` was
        # called directly. The recompute helper for cached records
        # was already routing correctly, so the cached LOOCV metrics
        # are unaffected — but production callers going through
        # ``predict_one`` would have been bitten.
        if self.aggregator in ("conf", "conf-distance"):
            pt_nbrs_for_agg = [
                {"ref_idx": int(i), "ctx_distance": float(d)}
                for i, d in nn_t
            ]
            pf_nbrs_for_agg = [
                {"ref_idx": int(i), "ctx_distance": float(d)}
                for i, d in nn_f
            ]
            pred_value = _aggregate_per_phase(
                pt_neighbours=pt_nbrs_for_agg,
                pf_neighbours=pf_nbrs_for_agg,
                pt_conf=float(result_t.get("confidence", 0.0) or 0.0),
                pf_conf=float(result_f.get("confidence", 0.0) or 0.0),
                y_train=[float(v) for v in self._y.tolist()],
                method=self.aggregator,
            )
            prediction = (
                float(pred_value) if pred_value is not None else float("nan")
            )
        else:
            prediction = _aggregate_predictions(
                sel_targets, sel_distances, self.aggregator,
            )

        # Confidence is the mean of phases that actually produced a
        # winning class. Averaging in 0 for a refused phase would
        # artificially halve the surviving phase's confidence.
        present_confs: list[float] = []
        if winner_t is not None:
            present_confs.append(
                float(result_t.get("confidence", 0.0) or 0.0)
            )
        if winner_f is not None:
            present_confs.append(
                float(result_f.get("confidence", 0.0) or 0.0)
            )
        conf = float(np.mean(present_confs)) if present_confs else 0.0

        return {
            "test_index": test_id,
            "task": "regression",
            "method": "CCR",
            "predicted_value": float(prediction),
            "confidence": float(conf),
            "phase_target": {
                "predicted_class": winner_t,
                "confidence": float(result_t.get("confidence", 0.0) or 0.0),
                "candidate_classes": result_t.get("candidate_classes", []),
                "final_label": result_t.get("final_label"),
                "neighbours": [
                    {"ref_idx": int(i), "ctx_distance": float(d)}
                    for i, d in nn_t
                ],
            },
            "phase_feature": {
                "predicted_class": winner_f,
                "confidence": float(result_f.get("confidence", 0.0) or 0.0),
                "candidate_classes": result_f.get("candidate_classes", []),
                "final_label": result_f.get("final_label"),
                "neighbours": [
                    {"ref_idx": int(i), "ctx_distance": float(d)}
                    for i, d in nn_f
                ],
            },
            "selected_indices": [int(i) for i in sel_indices],
            "selected_targets": sel_targets,
            "selected_distances": sel_distances,
            "aggregator": self.aggregator,
            "phase_views_agree": bool(
                winner_t is not None
                and winner_f is not None
                and len(
                    {ix for ix, _ in nn_t} & {ix for ix, _ in nn_f}
                ) > 0
            ),
        }


def _ccr_one_fold(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    llm: LLMClient,
    n_target_bins: int,
    n_feature_clusters: int,
    k_per_class: int,
    agent_k: int,
    max_workers: int,
    llm_temperature: float,
    aggregator: str,
    cluster_method: str = "hierarchical",
    leiden_knn: int = 4,
    leiden_resolution: float = 1.0,
    progress_cb: Callable[[int, float, float | None, float, float], None] | None = None,
) -> dict:
    n = len(y)
    train_idx = [j for j in range(n) if j != fold_idx]
    X_train = X[train_idx]
    y_train = y[train_idx]

    ccr = CrossClassificationRegressor(
        llm_client=llm,
        feature_names=list(feature_names),
        feature_descriptions=dict(feature_descriptions),
        background=background,
        n_target_bins=n_target_bins,
        n_feature_clusters=n_feature_clusters,
        k_per_class=k_per_class,
        agent_k=agent_k,
        max_workers=max_workers,
        llm_temperature=llm_temperature,
        aggregator=aggregator,
        cluster_method=cluster_method,
        leiden_knn=leiden_knn,
        leiden_resolution=leiden_resolution,
    ).fit(X_train, y_train)

    t0 = time.time()
    res = ccr.predict_one(X[fold_idx], test_id=fold_idx)
    elapsed = time.time() - t0

    pv = res.get("predicted_value")
    conf = float(res.get("confidence", 0.0) or 0.0)
    if progress_cb:
        progress_cb(fold_idx, float(y[fold_idx]), pv, conf, elapsed)

    # Map train-relative indices back to global indices.
    train_to_global = {ti: gi for ti, gi in enumerate(train_idx)}
    sel_indices_global = [
        int(train_to_global[i]) for i in res.get("selected_indices", [])
    ]

    return {
        "fold": int(fold_idx),
        "true": float(y[fold_idx]),
        "pred": float(pv) if pv is not None and np.isfinite(pv) else None,
        "confidence": conf,
        "elapsed_s": round(elapsed, 1),
        "phase_target": res.get("phase_target", {}),
        "phase_feature": res.get("phase_feature", {}),
        "selected_indices_global": sel_indices_global,
        "selected_targets": res.get("selected_targets", []),
        "selected_distances": res.get("selected_distances", []),
        "phase_views_agree": bool(res.get("phase_views_agree", False)),
        "aggregator": aggregator,
    }


def run_ccr_loocv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    llm: LLMClient,
    n_target_bins: int,
    n_feature_clusters: int,
    k_per_class: int,
    agent_k: int,
    max_workers: int,
    llm_temperature: float,
    fold_workers: int,
    aggregator: str,
    cluster_method: str = "hierarchical",
    leiden_knn: int = 4,
    leiden_resolution: float = 1.0,
) -> tuple[dict, list[dict]]:
    """Run CCR LOOCV with a single aggregator.

    The fold records returned here also carry ``selected_targets`` /
    ``selected_distances``, so callers wanting to compare alternate
    aggregators can do so without re-running the LLM-driven
    classification phases — see :func:`recompute_ccr_with_aggregator`.
    """
    n = len(y)
    fold_records: list[dict] = [None] * n  # type: ignore[list-item]
    completed = [0]

    def _progress(fi, true_v, pred_v, conf, elapsed):
        completed[0] += 1
        pred_str = "REFUSED" if pred_v is None else f"{pred_v:.4g}"
        logger.info(
            f"  [CCR/{aggregator}] fold {fi+1}/{n} done in {elapsed:.1f}s  "
            f"true={true_v:.4g}  pred={pred_str}  conf={conf:.2f}  "
            f"({completed[0]}/{n} folds finished)"
        )

    if fold_workers > 1:
        with ThreadPoolExecutor(max_workers=fold_workers) as pool:
            futures = {
                pool.submit(
                    _ccr_one_fold, i, X, y, feature_names,
                    feature_descriptions, background, llm,
                    n_target_bins, n_feature_clusters, k_per_class,
                    agent_k, max_workers, llm_temperature, aggregator,
                    cluster_method, leiden_knn, leiden_resolution,
                    _progress,
                ): i
                for i in range(n)
            }
            for fut in as_completed(futures):
                rec = fut.result()
                fold_records[rec["fold"]] = rec
    else:
        for i in range(n):
            fold_records[i] = _ccr_one_fold(
                i, X, y, feature_names, feature_descriptions,
                background, llm, n_target_bins, n_feature_clusters,
                k_per_class, agent_k, max_workers, llm_temperature,
                aggregator,
                cluster_method, leiden_knn, leiden_resolution,
                _progress,
            )

    y_true = [r["true"] for r in fold_records]
    y_pred = [r["pred"] for r in fold_records]
    metrics = compute_regression_metrics(y_true, y_pred, intervals=None)
    return metrics, fold_records


def recompute_ccr_with_aggregator(
    base_fold_records: list[dict],
    aggregator: str,
    y_all: list[float] | None = None,
) -> tuple[dict, list[dict]]:
    """Recompute predictions + metrics from cached CCR fold records
    using a different aggregator. Avoids re-running the underlying
    LLM-driven classification calls.

    Supports both:
      * Simple aggregators (``mean`` / ``median`` / ``weighted``)
        which only need the cached deduped ``selected_targets`` /
        ``selected_distances`` lists.
      * Per-phase aggregators (``conf`` / ``conf-distance``) which
        also need ``y_all`` (the full LOOCV target list) to
        reconstruct the per-fold training targets and to weight
        each phase's neighbours by that phase's classification
        confidence.

    A fold whose contributions are empty stays a refusal.
    """
    per_phase_methods = {"conf", "conf-distance"}
    is_per_phase = aggregator in per_phase_methods
    if is_per_phase and y_all is None:
        raise ValueError(
            f"Aggregator {aggregator!r} needs y_all to reconstruct "
            "per-fold training targets; pass y_all from the dataset."
        )

    new_records: list[dict] = []
    for rec in base_fold_records:
        if rec is None:
            continue

        if is_per_phase:
            fold = int(rec["fold"])
            y_train = [
                float(v) for i, v in enumerate(y_all)  # type: ignore[arg-type]
                if i != fold
            ]
            pt = rec.get("phase_target") or {}
            pf = rec.get("phase_feature") or {}
            v = _aggregate_per_phase(
                pt_neighbours=pt.get("neighbours") or [],
                pf_neighbours=pf.get("neighbours") or [],
                pt_conf=float(pt.get("confidence", 0.0) or 0.0),
                pf_conf=float(pf.get("confidence", 0.0) or 0.0),
                y_train=y_train,
                method=aggregator,
            )
            new_pred = (
                float(v) if v is not None and np.isfinite(v) else None
            )
        else:
            targets = rec.get("selected_targets") or []
            distances = (
                rec.get("selected_distances") or [0.0] * len(targets)
            )
            if not targets:
                new_pred = None
            else:
                v = _aggregate_predictions(
                    [float(t) for t in targets],
                    [float(d) for d in distances],
                    aggregator,
                )
                new_pred = float(v) if np.isfinite(v) else None

        new_rec = dict(rec)
        new_rec["pred"] = new_pred
        new_rec["aggregator"] = aggregator
        new_records.append(new_rec)

    y_true = [r["true"] for r in new_records]
    y_pred = [r["pred"] for r in new_records]
    metrics = compute_regression_metrics(y_true, y_pred, intervals=None)
    return metrics, new_records


# ── main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LOOCV regression on BSLT_huigui.csv: MolActAgent vs "
            "ML baselines vs a pure-LLM baseline."
        )
    )
    parser.add_argument("--csv", type=str,
                        default=str(_PROJECT_ROOT / "data" / "BSLT_huigui.csv"))
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="standard_value")
    parser.add_argument("--k", type=int, default=5,
                        help="MolActAgent neighbours per tuple "
                             "(regression mode, default 5).")
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Per-fold thread pool for MolActAgent's "
                             "tuple LLM calls.")
    parser.add_argument("--fold-workers", type=int, default=4,
                        help="Number of LOOCV folds run in parallel.")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-profile", type=str, default="deepseek")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Run ML baselines only (no LLM calls).")
    parser.add_argument("--skip-pure-llm", action="store_true",
                        help="Skip the pure-LLM baseline.")
    parser.add_argument("--skip-molactagent", action="store_true",
                        help="Skip MolActAgent.")
    parser.add_argument("--skip-ccr", action="store_true",
                        help="Skip the cross-classification regressor.")
    parser.add_argument("--ccr-n-mode",
                        choices=["fixed", "sqrt", "sqrt-uncapped"],
                        default="sqrt",
                        help="How to choose the per-phase class count "
                             "N (target bins) and M (feature clusters). "
                             "'fixed' uses --ccr-bins / --ccr-clusters "
                             "as-is; 'sqrt' (default) uses "
                             "max(2, min(floor(sqrt(n_train)), 10)); "
                             "'sqrt-uncapped' drops the upper cap.")
    parser.add_argument("--ccr-bins", type=int, default=5,
                        help="Number of equal-frequency target bins "
                             "(used only when --ccr-n-mode=fixed; "
                             "default 5).")
    parser.add_argument("--ccr-clusters", type=int, default=5,
                        help="Number of feature-space clusters "
                             "(used only when --ccr-n-mode=fixed; "
                             "default 5).")
    parser.add_argument("--ccr-cluster-method",
                        choices=["hierarchical", "leiden"],
                        default="hierarchical",
                        help="Phase 2 clustering algorithm. "
                             "'hierarchical' uses sklearn "
                             "AgglomerativeClustering with average "
                             "linkage on cosine distance (default). "
                             "'leiden' uses community detection on a "
                             "k-NN graph built from cosine similarity "
                             "(requires leidenalg + python-igraph).")
    parser.add_argument("--ccr-leiden-knn", type=int, default=4,
                        help="k for the k-NN graph fed into Leiden "
                             "(default 4).")
    parser.add_argument("--ccr-leiden-resolution", type=float,
                        default=1.0,
                        help="Resolution parameter for Leiden's "
                             "modularity optimisation (default 1.0; "
                             "higher = more, smaller communities).")
    parser.add_argument("--ccr-k-per-class", type=int, default=2,
                        help="Top-k references per predicted class "
                             "to draw for the aggregation step "
                             "(default 2).")
    parser.add_argument("--ccr-agent-k", type=int, default=2,
                        help="MolActAgent k for each multiclass phase "
                             "(default 2; small because each class "
                             "may only contain ~3 samples on tiny "
                             "datasets).")
    parser.add_argument("--ccr-aggregators", nargs="+",
                        default=["mean", "median", "weighted",
                                 "conf", "conf-distance"],
                        choices=["mean", "median", "weighted",
                                 "conf", "conf-distance"],
                        help="Aggregator variants to evaluate "
                             "(default: all five; reported as "
                             "separate methods CCR-<name>). The "
                             "first one in the list is the primary "
                             "(LLM-driven) run; the rest are "
                             "recomputed from the cached classification "
                             "phases without extra LLM calls.")
    parser.add_argument("--output-dir", type=str,
                        default=str(_PROJECT_ROOT / "data"))
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for output filenames.")
    args = parser.parse_args()

    log_files = setup_logging(log_dir=_PROJECT_ROOT / "logs")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log : {log_files['full']}")
    logger.info(f"Args: {vars(args)}")

    csv_path = pathlib.Path(args.csv).resolve()
    X, y, meta = build_regression_dataset_from_csv(
        csv_path,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        verbose=True,
    )
    n = len(y)
    feature_names = meta["feature_names"]
    feature_descriptions = meta["feature_descriptions"]
    smiles_all = meta["smiles"]

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_json = (
        output_dir
        / f"loocv_regression_{csv_path.stem}_k{args.k}{tag_suffix}.json"
    )
    out_table = (
        output_dir
        / f"loocv_regression_{csv_path.stem}_k{args.k}{tag_suffix}.txt"
    )

    all_metrics: dict[str, dict] = {}
    all_predictions: dict[str, list[dict]] = {}

    # ── ML baselines ─────────────────────────────────────────────
    logger.info(
        "\n" + "=" * 60 +
        "\n  ML regression baselines (LOOCV)\n" + "=" * 60
    )
    t0 = time.time()
    ml_metrics = run_ml_loocv_regression(X, y)
    all_metrics.update(ml_metrics)
    logger.info(f"ML LOOCV finished in {time.time() - t0:.1f}s")

    if args.skip_llm:
        table = format_regression_metrics_table(all_metrics)
        print("\n" + table)
        out_table.write_text(table, encoding="utf-8")
        out_json.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "task": "regression",
                    "n_samples": n,
                    "target_range": [float(y.min()), float(y.max())],
                    "feature_names": feature_names,
                    "metrics": all_metrics,
                },
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        logger.info(f"Saved metrics → {out_json}")
        logger.info(f"Saved table   → {out_table}")
        return

    # ── load LLM once, share with both LLM-driven baselines ────
    llm_cfg = load_llm_config(_PROJECT_ROOT, profile=args.llm_profile)
    logger.info(
        f"LLM profile={args.llm_profile!r} -> "
        f"model={llm_cfg['model_name']!r} "
        f"base_url={llm_cfg['base_url']!r}"
    )
    llm = LLMClient(**llm_cfg, timeout=120.0)

    # ── Pure LLM baseline ──────────────────────────────────────
    if not args.skip_pure_llm:
        logger.info(
            "\n" + "=" * 60 +
            "\n  Pure-LLM regression baseline (LOOCV)\n" + "=" * 60
        )
        t0 = time.time()
        pure_metrics, pure_records = run_pure_llm_loocv_regression(
            X, y, smiles_all, feature_names, BACKGROUND_BSLT_REG, llm,
            llm_temperature=args.llm_temperature,
            fold_workers=args.fold_workers,
        )
        elapsed = time.time() - t0
        logger.info(f"Pure LLM LOOCV finished in {elapsed:.1f}s")
        all_metrics["PureLLM"] = pure_metrics
        all_predictions["PureLLM"] = [
            {k: v for k, v in r.items() if k != "raw_response"}
            for r in pure_records
        ]
        all_predictions["PureLLM_full"] = pure_records
    else:
        logger.info("Pure LLM baseline skipped.")

    # ── MolActAgent ─────────────────────────────────────────────
    if not args.skip_molactagent:
        logger.info(
            "\n" + "=" * 60 +
            "\n  MolActAgent regression (LOOCV)\n" + "=" * 60
        )
        t0 = time.time()
        ma_metrics, ma_records = run_molactagent_loocv_regression(
            X, y,
            feature_names, feature_descriptions,
            BACKGROUND_BSLT_REG, llm,
            k=args.k, metric=args.metric,
            max_workers=args.max_workers,
            llm_temperature=args.llm_temperature,
            fold_workers=args.fold_workers,
        )
        elapsed = time.time() - t0
        logger.info(f"MolActAgent LOOCV finished in {elapsed:.1f}s")
        all_metrics["MolActAgent"] = ma_metrics

        ma_records_lite: list[dict] = []
        for r in ma_records:
            if r is None:
                continue
            raw = r["raw"] or {}
            ma_records_lite.append({
                "fold": r["fold"],
                "true": r["true"],
                "pred": r["pred"],
                "interval": r["interval"],
                "confidence": r["confidence"],
                "elapsed_s": r["elapsed_s"],
                "extrapolation_flagged": raw.get(
                    "extrapolation_flagged", False
                ),
                "predictions": raw.get("predictions", []),
                "driver_attribution": (raw.get("driver_attribution") or [])[:5],
            })
        all_predictions["MolActAgent"] = ma_records_lite
    else:
        logger.info("MolActAgent skipped.")

    # ── Cross-Classification Regressor ─────────────────────────
    if not args.skip_ccr:
        # Resolve N (target bins) and M (feature clusters) from
        # --ccr-n-mode once for the whole run. We size from the
        # FULL dataset size n (not n_train = n-1) because:
        #   * N/M is a hyperparameter chosen ahead of time, not a
        #     per-fold quantity — pre-registered defaults per the
        #     experimental discipline we've been holding.
        #   * floor(sqrt(n)) is the canonical ML default and what
        #     "the dataset size" naturally refers to.
        n_target_bins = _resolve_n_classes(
            n, args.ccr_n_mode, args.ccr_bins,
        )
        n_feature_clusters = _resolve_n_classes(
            n, args.ccr_n_mode, args.ccr_clusters,
        )
        logger.info(
            f"CCR cluster sizing: n-mode={args.ccr_n_mode!r}, "
            f"n_full={n} -> N={n_target_bins} bins, "
            f"M={n_feature_clusters} clusters; "
            f"clustering={args.ccr_cluster_method!r}"
        )

        # The two CCR phases (target-bin classification + feature-
        # cluster classification) only depend on the data and LLM,
        # not on the aggregator. So we run the LLM-driven part ONCE
        # using the first aggregator and recompute metrics for the
        # remaining aggregators from the cached fold records — saves
        # the bulk of the LLM cost when multiple aggregators are
        # evaluated.
        if not args.ccr_aggregators:
            args.ccr_aggregators = ["mean"]
        primary_agg = args.ccr_aggregators[0]
        primary_method = f"CCR-{primary_agg}"
        method_tag = (
            "" if args.ccr_cluster_method == "hierarchical"
            else f"-{args.ccr_cluster_method}"
        )
        logger.info(
            "\n" + "=" * 60 +
            f"\n  {primary_method}{method_tag} (LOOCV, "
            f"N={n_target_bins} bins, M={n_feature_clusters} "
            f"clusters, k_per_class={args.ccr_k_per_class})\n"
            + "=" * 60
        )
        t0 = time.time()
        ccr_metrics, ccr_records = run_ccr_loocv(
            X, y,
            feature_names, feature_descriptions,
            BACKGROUND_BSLT_REG, llm,
            n_target_bins=n_target_bins,
            n_feature_clusters=n_feature_clusters,
            k_per_class=args.ccr_k_per_class,
            agent_k=args.ccr_agent_k,
            max_workers=args.max_workers,
            llm_temperature=args.llm_temperature,
            fold_workers=args.fold_workers,
            aggregator=primary_agg,
            cluster_method=args.ccr_cluster_method,
            leiden_knn=args.ccr_leiden_knn,
            leiden_resolution=args.ccr_leiden_resolution,
        )
        elapsed = time.time() - t0
        logger.info(
            f"{primary_method}{method_tag} LOOCV finished in "
            f"{elapsed:.1f}s (LLM-driven phases evaluated)"
        )
        primary_label = f"{primary_method}{method_tag}"
        all_metrics[primary_label] = ccr_metrics
        all_predictions[primary_label] = ccr_records

        for agg in args.ccr_aggregators[1:]:
            method_name = f"CCR-{agg}{method_tag}"
            logger.info(
                f"Recomputing {method_name} from cached "
                "classifications (no extra LLM calls)..."
            )
            agg_metrics, agg_records = recompute_ccr_with_aggregator(
                ccr_records, agg, y_all=[float(v) for v in y.tolist()],
            )
            all_metrics[method_name] = agg_metrics
            all_predictions[method_name] = agg_records
    else:
        logger.info("CCR skipped.")

    # ── output ─────────────────────────────────────────────────
    table = format_regression_metrics_table(all_metrics)
    print("\n" + table)
    out_table.write_text(table, encoding="utf-8")

    out_json.write_text(
        json.dumps(
            {
                "args": vars(args),
                "task": "regression",
                "n_samples": n,
                "target_range": [float(y.min()), float(y.max())],
                "target_mean": float(y.mean()),
                "target_std": float(y.std()),
                "feature_names": feature_names,
                "background": BACKGROUND_BSLT_REG,
                "smiles": smiles_all,
                "y_true": [float(v) for v in y.tolist()],
                "metrics": all_metrics,
                "predictions": all_predictions,
                "llm_stats": llm.stats,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )

    logger.info(f"Saved metrics  -> {out_json}")
    logger.info(f"Saved table    -> {out_table}")
    llm.print_stats()


if __name__ == "__main__":
    main()
