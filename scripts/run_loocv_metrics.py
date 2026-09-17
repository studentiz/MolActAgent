"""Comprehensive LOOCV evaluation for MolActAgent — both classification
and regression.

For every sample i in the dataset, we hold it out, fit on the
remaining n-1 samples, predict on i, and aggregate predictions and
continuous scores across all n folds.

Classification mode (default for label-string datasets like
``data/antiTB_evidence_table.csv``):
  - Accuracy, Balanced Accuracy, MCC, Macro F1
  - Per-class precision / recall / F1
  - Specificity (negative-class recall), NPV
  - ROC AUC, PR AUC (Average Precision)
  - Confusion matrix (TP / FP / FN / TN), Inconclusive count
  - ML baselines: KNN, NB, LR, SVM, RF, GBC, optionally XGB.
  - Continuous score for MolActAgent comes from the deterministic
    weighted-vote tally:
       score_pos = weighted_votes[positive_class] / sum(weighted)
    which captures evidence direction and strength, not just the
    discrete label.

Regression mode (auto-detected for continuous-target datasets like
``examples/penguins/penguins_30.csv``):
  - MAE, RMSE, R^2, Pearson r, Spearman r
  - Median absolute error, fraction-within-interval
  - Refusal count
  - ML baselines: KNN-1/3/5 regressor, Linear, Ridge, Lasso, RF,
    GBR, optionally XGBRegressor.

Multi-class refactor compatibility note: MolActAgent's predictions are
now real domain labels (e.g. ``"active"`` / ``"inactive"``) rather
than the legacy ``"Pos"`` / ``"Neg"`` sentinels. This script accepts
``--positive-class`` to designate which user label is treated as the
"positive" class for binary metric computation; it auto-detects from
the dataset when there are exactly two classes.

Usage:
    conda activate molactagent_env
    # Classification (binary):
    python scripts/run_loocv_metrics.py \
        --csv data/antiTB_evidence_table.csv \
        --positive-class active --k 3 --llm-profile deepseek
    # Regression (Penguins demo):
    python scripts/run_loocv_metrics.py \
        --csv examples/penguins/reference.csv \
        --task regression \
        --target-col body_mass_g \
        --id-col sample_id \
        --feature-cols bill_length_mm bill_depth_mm flipper_length_mm \
        --background examples/penguins/background.txt \
        --feature-descriptions examples/penguins/feature_descriptions.json \
        --k 5 --llm-profile deepseek
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr, spearmanr

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None  # type: ignore[misc, assignment]
    XGBRegressor = None  # type: ignore[misc, assignment]

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from molactagent import MolActAgent, LLMClient, setup_logging  # noqa: E402

from prepare_features import build_dataset_from_dataframe  # noqa: E402
from run_demo import load_llm_config  # noqa: E402

logger = logging.getLogger("loocv_metrics")

BACKGROUND_ANTITB = (
    "Task: binary classification of cyclic peptide compounds for "
    "anti-tuberculosis (Mycobacterium tuberculosis) activity.\n"
    "Label semantics: Pos = exhibits anti-TB activity (active in "
    "literature/experiment); Neg = no significant activity (inactive).\n"
    "Features: RDKit molecular descriptors and MACCS structural "
    "fingerprints. For peptide-like compounds, drivers of anti-TB "
    "activity tend to involve membrane permeability (TPSA, MolLogP, "
    "MolWt), conformational rigidity (rotatable bonds, ring count), "
    "and specific structural motifs (MACCS bits)."
)

BACKGROUND_BSLT = (
    "Task: binary classification of cyclic peptide and polyketide "
    "compounds for brine-shrimp (Artemia salina) lethality, a "
    "general toxicity screen.\n"
    "Label semantics: Pos = positive in the lethality assay (toxic); "
    "Neg = negative or low toxicity.\n"
    "Features: RDKit molecular descriptors and MACCS fingerprints."
)


# ── metrics ──────────────────────────────────────────────────────────

def compute_full_metrics(
    y_true: list[str],
    y_pred: list[str],
    score_pos: list[float] | None = None,
    positive_class: str = "Pos",
    negative_class: str = "Neg",
) -> dict:
    """Compute the full evaluation suite for binary classification.

    ``y_true`` and ``y_pred`` carry the actual domain class names
    (e.g. ``"active"`` / ``"inactive"``). Anything outside
    ``{positive_class, negative_class}`` (e.g. ``"Inconclusive"`` or
    ``"Unknown"``) is treated as a wrong answer for confusion-matrix
    and accuracy purposes; for AUC computations the score is left
    untouched so that an inconclusive verdict near 0.5 neither
    boosts nor punishes ranking.

    ``score_pos`` is a probability-like score for the positive class
    (higher = more confidence in positive). Optional; if None, AUC
    metrics are returned as None.
    """
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    y_true_bin = np.array(
        [1 if l == positive_class else 0 for l in y_true], dtype=int,
    )

    # Confusion matrix counts inconclusive / unknown as wrong.
    tp = sum(
        1 for t, p in zip(y_true, y_pred)
        if t == positive_class and p == positive_class
    )
    tn = sum(
        1 for t, p in zip(y_true, y_pred)
        if t == negative_class and p == negative_class
    )
    fp = sum(
        1 for t, p in zip(y_true, y_pred)
        if t == negative_class and p == positive_class
    )
    fn = sum(
        1 for t, p in zip(y_true, y_pred)
        if t == positive_class and p == negative_class
    )
    n_unk = sum(
        1 for p in y_pred
        if p not in (positive_class, negative_class)
    )

    accuracy = (tp + tn) / n if n else 0.0

    # For sklearn metrics, treat inconclusive predictions as the
    # negative class (a forced binary decision) so the labels stay
    # binary; we ALSO show the inconclusive count separately so the
    # reader sees both.
    y_pred_bin = np.array(
        [1 if p == positive_class else 0 for p in y_pred], dtype=int,
    )
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    f1_pos = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_neg = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    prec_pos = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    prec_neg = precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    rec_neg = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    out = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "macro_f1": float(macro_f1),
        "precision_pos": float(prec_pos),
        "recall_pos": float(rec_pos),
        "f1_pos": float(f1_pos),
        "precision_neg": float(prec_neg),
        "recall_neg": float(rec_neg),
        "f1_neg": float(f1_neg),
        "specificity_neg": float(specificity),
        "unknown_rate": n_unk / n if n else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "unknown": int(n_unk),
        "n_samples": int(n),
    }

    if score_pos is not None and len(score_pos) == n:
        # Replace any NaN scores with 0.5 so AUC metrics still compute.
        s = np.array(score_pos, dtype=float)
        s = np.where(np.isnan(s), 0.5, s)
        try:
            out["roc_auc"] = float(roc_auc_score(y_true_bin, s))
        except ValueError:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true_bin, s))
        except ValueError:
            out["pr_auc"] = None
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


def format_metrics_table(results: dict[str, dict]) -> str:
    """Render a comparison table for all methods."""
    methods = list(results.keys())
    metric_keys = [
        "balanced_accuracy", "accuracy", "mcc", "macro_f1",
        "f1_pos", "f1_neg", "precision_pos", "recall_pos",
        "precision_neg", "recall_neg", "specificity_neg",
        "roc_auc", "pr_auc", "unknown_rate",
    ]
    labels = [
        "Bal Acc", "Accuracy", "MCC", "Macro F1",
        "F1 (Pos)", "F1 (Neg)", "Prec (Pos)", "Recall (Pos)",
        "Prec (Neg)", "Recall (Neg)", "Specificity",
        "ROC AUC", "PR AUC", "Unknown rate",
    ]

    col_w = max(11, max(len(m) for m in methods) + 2)
    label_w = 16

    def _fmt(v):
        if v is None:
            return f"{'—':>{col_w}s}"
        return f"{v:>{col_w}.3f}"

    sep = "-" * (label_w + col_w * len(methods))
    lines = [sep]
    lines.append(
        f"{'Metric':<{label_w}s}"
        + "".join(f"{m:>{col_w}s}" for m in methods)
    )
    lines.append(sep)
    for key, label in zip(metric_keys, labels):
        row = f"{label:<{label_w}s}"
        for m in methods:
            row += _fmt(results[m].get(key))
        lines.append(row)
    lines.append(sep)

    # confusion matrix line
    lines.append("Confusion matrix (TP / FP / FN / TN  + Unknown):")
    for m in methods:
        r = results[m]
        lines.append(
            f"  {m:<14s}  TP={r['tp']:>2d}  FP={r['fp']:>2d}  "
            f"FN={r['fn']:>2d}  TN={r['tn']:>2d}  Unk={r['unknown']:>2d}"
            f"  (n={r['n_samples']})"
        )
    return "\n".join(lines)


# ── ML baselines (LOOCV with predict_proba where available) ─────────

def _make_ml_classifiers(y_train: np.ndarray, positive_class: str = "Pos"):
    y_tr_bin = np.array([1 if l == positive_class else 0 for l in y_train])
    pos_count = int(y_tr_bin.sum())
    neg_count = len(y_tr_bin) - pos_count
    xgb_scale = neg_count / pos_count if pos_count > 0 else 1.0

    out: dict = {
        "KNN1": KNeighborsClassifier(n_neighbors=1),
        "KNN3": KNeighborsClassifier(n_neighbors=3),
        "KNN5": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42,
        ),
        "LR": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "GBC": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    if XGBClassifier is not None:
        out["XGB"] = XGBClassifier(
            n_estimators=200,
            scale_pos_weight=xgb_scale,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    return out


def run_ml_loocv(
    X: np.ndarray,
    y: np.ndarray,
    positive_class: str,
    negative_class: str,
) -> dict[str, dict]:
    n = len(y)
    label_map = {positive_class: 1, negative_class: 0}
    inv_map = {1: positive_class, 0: negative_class}
    needs_numeric = {"XGB"}

    name_seed = list(_make_ml_classifiers(y[1:], positive_class).keys())
    all_preds: dict[str, list[str]] = {nm: [] for nm in name_seed}
    all_scores: dict[str, list[float]] = {nm: [] for nm in name_seed}

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i], dtype=int)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[i:i + 1]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        classifiers = _make_ml_classifiers(y_train, positive_class)
        y_tr_enc = np.array([label_map[l] for l in y_train])

        for name, clf in classifiers.items():
            try:
                if name in needs_numeric:
                    clf.fit(X_tr, y_tr_enc)
                    pred_raw = int(clf.predict(X_te)[0])
                    pred = inv_map[pred_raw]
                    if hasattr(clf, "predict_proba"):
                        score = float(clf.predict_proba(X_te)[0, 1])
                    else:
                        score = float(pred_raw)
                else:
                    clf.fit(X_tr, y_train)
                    pred = str(clf.predict(X_te)[0])
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(X_te)[0]
                        pos_idx = int(
                            np.where(clf.classes_ == positive_class)[0][0]
                        )
                        score = float(proba[pos_idx])
                    else:
                        score = 1.0 if pred == positive_class else 0.0
            except Exception as e:
                logger.error(f"LOOCV fold {i} ML {name} failed: {e}")
                pred = "Inconclusive"
                score = 0.5
            all_preds[name].append(pred)
            all_scores[name].append(score)

    y_list = y.tolist()
    return {
        name: compute_full_metrics(
            y_list, all_preds[name], all_scores[name],
            positive_class=positive_class,
            negative_class=negative_class,
        )
        for name in name_seed
    }


# ── MolActAgent LOOCV ─────────────────────────────────────────────────

def _molactagent_score(result: dict, positive_class: str) -> float:
    """Continuous positive-class score in [0, 1] from the agent's
    per-class vote summary. Returns 0.5 when no decisive evidence
    was found.
    """
    vs = result.get("vote_summary", {}) or {}
    per_class = vs.get("per_class", {}) or {}
    if not per_class:
        return 0.5
    weighted_pos = float(
        (per_class.get(positive_class) or {}).get("weighted", 0.0) or 0.0
    )
    weighted_total = sum(
        float((info or {}).get("weighted", 0.0) or 0.0)
        for info in per_class.values()
    )
    if weighted_total <= 1e-12:
        return 0.5
    return weighted_pos / weighted_total


def _molactagent_one_fold(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    llm: LLMClient,
    k: int,
    metric: str,
    max_workers: int,
    llm_temperature: float,
    positive_class: str,
    progress_cb: Callable[[int, str, str, float, float], None] | None = None,
) -> dict:
    n = len(y)
    train_idx = np.array([j for j in range(n) if j != fold_idx], dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    if len(np.unique(y_train)) < 2:
        logger.warning(
            f"LOOCV fold {fold_idx}: train set is single-class; "
            "MolActAgent will return Inconclusive."
        )

    agent = MolActAgent(
        llm_client=llm,
        k=k,
        metric=metric,
        feature_names=list(feature_names),
        feature_descriptions=dict(feature_descriptions),
        background=background,
        max_workers=max_workers,
        llm_temperature=llm_temperature,
    )
    agent.fit(X_train, y_train)

    t0 = time.time()
    res = agent.predict_one(X[fold_idx], test_id=fold_idx)
    elapsed = time.time() - t0

    pred = res.get("final_label", "Inconclusive")
    conf = float(res.get("confidence", 0.0) or 0.0)
    score = _molactagent_score(res, positive_class)

    if progress_cb:
        progress_cb(fold_idx, str(y[fold_idx]), pred, conf, elapsed)

    return {
        "fold": int(fold_idx),
        "true": str(y[fold_idx]),
        "pred": pred,
        "score_pos": score,
        "confidence": conf,
        "elapsed_s": round(elapsed, 1),
        "raw": res,
    }


def run_molactagent_loocv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    llm: LLMClient,
    k: int,
    metric: str,
    max_workers: int,
    llm_temperature: float,
    fold_workers: int,
    positive_class: str,
    negative_class: str,
) -> tuple[dict, list[dict]]:
    n = len(y)
    fold_records: list[dict] = [None] * n  # type: ignore[list-item]

    completed = [0]

    def _progress(fi, true, pred, conf, elapsed):
        completed[0] += 1
        logger.info(
            f"  fold {fi+1}/{n} done in {elapsed:.1f}s  "
            f"true={true}  pred={pred}  conf={conf:.2f}  "
            f"({completed[0]}/{n} folds finished)"
        )

    if fold_workers > 1:
        logger.info(
            f"Running MolActAgent LOOCV with fold_workers={fold_workers}, "
            f"max_workers per fold={max_workers}"
        )
        with ThreadPoolExecutor(max_workers=fold_workers) as pool:
            futures = {
                pool.submit(
                    _molactagent_one_fold, i, X, y, feature_names,
                    feature_descriptions, background, llm, k, metric,
                    max_workers, llm_temperature, positive_class,
                    _progress,
                ): i
                for i in range(n)
            }
            for fut in as_completed(futures):
                rec = fut.result()
                fold_records[rec["fold"]] = rec
    else:
        for i in range(n):
            fold_records[i] = _molactagent_one_fold(
                i, X, y, feature_names, feature_descriptions,
                background, llm, k, metric, max_workers, llm_temperature,
                positive_class, _progress,
            )

    y_true = [r["true"] for r in fold_records]
    y_pred = [r["pred"] for r in fold_records]
    scores = [r["score_pos"] for r in fold_records]
    metrics = compute_full_metrics(
        y_true, y_pred, scores,
        positive_class=positive_class,
        negative_class=negative_class,
    )

    return metrics, fold_records


# ── regression metrics + baselines ─────────────────────────────────


def compute_regression_metrics(
    y_true: list[float],
    y_pred: list[float | None],
    intervals: list[tuple[float, float] | None] | None = None,
) -> dict:
    """Compute MAE / RMSE / R^2 / Pearson r / Spearman r plus a few
    "honesty" metrics (refusal count, fraction-within-interval).

    Refused predictions (``y_pred[i] is None``) are excluded from the
    point-error metrics but counted in ``n_refused``.
    """
    yt = np.array(y_true, dtype=float)
    paired = [
        (t, p, iv)
        for t, p, iv in zip(
            yt,
            y_pred,
            intervals if intervals is not None else [None] * len(yt),
        )
        if p is not None and np.isfinite(p)
    ]
    n_refused = len(yt) - len(paired)

    if not paired:
        return {
            "n": int(len(yt)),
            "n_refused": n_refused,
            "n_predicted": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "pearson_r": None,
            "spearman_r": None,
            "median_abs_err": None,
            "fraction_within_interval": None,
        }

    yt_arr = np.array([t for t, _, _ in paired], dtype=float)
    yp_arr = np.array([p for _, p, _ in paired], dtype=float)

    mae = float(mean_absolute_error(yt_arr, yp_arr))
    rmse = float(np.sqrt(mean_squared_error(yt_arr, yp_arr)))
    r2 = float(r2_score(yt_arr, yp_arr)) if yt_arr.size >= 2 else None
    median_abs = float(median_absolute_error(yt_arr, yp_arr))

    if yt_arr.size >= 3 and float(yt_arr.std()) > 0 and float(yp_arr.std()) > 0:
        pr = float(pearsonr(yt_arr, yp_arr).statistic)
        sr = float(spearmanr(yt_arr, yp_arr).statistic)
    else:
        pr = sr = None

    if intervals is not None:
        within = []
        for t, _, iv in paired:
            if iv is None:
                continue
            try:
                lo, hi = float(iv[0]), float(iv[1])
            except (TypeError, ValueError):
                continue
            within.append(1 if lo <= float(t) <= hi else 0)
        frac_within = (
            float(sum(within) / len(within)) if within else None
        )
    else:
        frac_within = None

    return {
        "n": int(len(yt)),
        "n_refused": int(n_refused),
        "n_predicted": int(len(paired)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pr,
        "spearman_r": sr,
        "median_abs_err": median_abs,
        "fraction_within_interval": frac_within,
    }


def _make_ml_regressors() -> dict:
    out: dict = {
        "KNN1": KNeighborsRegressor(n_neighbors=1),
        "KNN3": KNeighborsRegressor(n_neighbors=3),
        "KNN5": KNeighborsRegressor(n_neighbors=5),
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
        "RF": RandomForestRegressor(n_estimators=200, random_state=42),
        "GBR": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    if XGBRegressor is not None:
        out["XGB"] = XGBRegressor(
            n_estimators=200, random_state=42,
            verbosity=0,
        )
    return out


def run_ml_loocv_regression(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    n = len(y)
    name_seed = list(_make_ml_regressors().keys())
    all_preds: dict[str, list[float | None]] = {nm: [] for nm in name_seed}

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i], dtype=int)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[i:i + 1]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        regressors = _make_ml_regressors()
        for name, reg in regressors.items():
            try:
                reg.fit(X_tr, y_train)
                pred = float(reg.predict(X_te)[0])
            except Exception as e:
                logger.error(f"LOOCV fold {i} ML {name} regressor failed: {e}")
                pred = None
            all_preds[name].append(pred)

    y_list = y.tolist()
    return {
        name: compute_regression_metrics(y_list, all_preds[name], intervals=None)
        for name in name_seed
    }


def _molactagent_one_fold_regression(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    llm: LLMClient,
    k: int,
    metric: str,
    max_workers: int,
    llm_temperature: float,
    progress_cb: Callable[[int, float, float | None, float, float], None] | None = None,
) -> dict:
    n = len(y)
    train_idx = np.array([j for j in range(n) if j != fold_idx], dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]

    agent = MolActAgent(
        llm_client=llm,
        k=k,
        metric=metric,
        feature_names=list(feature_names),
        feature_descriptions=dict(feature_descriptions),
        background=background,
        max_workers=max_workers,
        llm_temperature=llm_temperature,
    )
    agent.fit(X_train, y_train, task="regression")

    t0 = time.time()
    res = agent.predict_one(X[fold_idx], test_id=fold_idx)
    elapsed = time.time() - t0

    pv = res.get("predicted_value")
    iv = res.get("interval") or (None, None)
    conf = float(res.get("confidence", 0.0) or 0.0)

    if progress_cb:
        progress_cb(fold_idx, float(y[fold_idx]), pv, conf, elapsed)

    return {
        "fold": int(fold_idx),
        "true": float(y[fold_idx]),
        "pred": pv,
        "interval": (
            (float(iv[0]), float(iv[1]))
            if iv[0] is not None and iv[1] is not None
            else None
        ),
        "confidence": conf,
        "elapsed_s": round(elapsed, 1),
        "raw": res,
    }


def run_molactagent_loocv_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    llm: LLMClient,
    k: int,
    metric: str,
    max_workers: int,
    llm_temperature: float,
    fold_workers: int,
) -> tuple[dict, list[dict]]:
    n = len(y)
    fold_records: list[dict] = [None] * n  # type: ignore[list-item]
    completed = [0]

    def _progress(fi, true, pred, conf, elapsed):
        completed[0] += 1
        pred_str = "REFUSED" if pred is None else f"{pred:.4g}"
        logger.info(
            f"  fold {fi+1}/{n} done in {elapsed:.1f}s  "
            f"true={true:.4g}  pred={pred_str}  conf={conf:.2f}  "
            f"({completed[0]}/{n} folds finished)"
        )

    if fold_workers > 1:
        logger.info(
            f"Running MolActAgent regression LOOCV with "
            f"fold_workers={fold_workers}, "
            f"max_workers per fold={max_workers}"
        )
        with ThreadPoolExecutor(max_workers=fold_workers) as pool:
            futures = {
                pool.submit(
                    _molactagent_one_fold_regression, i, X, y, feature_names,
                    feature_descriptions, background, llm, k, metric,
                    max_workers, llm_temperature, _progress,
                ): i
                for i in range(n)
            }
            for fut in as_completed(futures):
                rec = fut.result()
                fold_records[rec["fold"]] = rec
    else:
        for i in range(n):
            fold_records[i] = _molactagent_one_fold_regression(
                i, X, y, feature_names, feature_descriptions,
                background, llm, k, metric, max_workers, llm_temperature,
                _progress,
            )

    y_true = [r["true"] for r in fold_records]
    y_pred = [r["pred"] for r in fold_records]
    intervals = [r["interval"] for r in fold_records]
    metrics = compute_regression_metrics(y_true, y_pred, intervals)
    return metrics, fold_records


def format_regression_metrics_table(metrics: dict[str, dict]) -> str:
    """Pretty fixed-width table for regression metric comparisons."""
    cols = [
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        ("R^2", "r2"),
        ("Pearson", "pearson_r"),
        ("Spearman", "spearman_r"),
        ("MedianAE", "median_abs_err"),
        ("Within%", "fraction_within_interval"),
        ("Refused", "n_refused"),
    ]
    name_w = max((len(name) for name in metrics.keys()), default=10)
    name_w = max(name_w, 12)
    header = f"{'Method':<{name_w}}  " + "  ".join(
        f"{label:>9}" for label, _ in cols
    )
    sep = "-" * len(header)

    rows = [header, sep]
    for name, m in metrics.items():
        cells = []
        for label, key in cols:
            v = m.get(key)
            if v is None:
                cells.append(f"{'—':>9}")
            elif key in ("n_refused",):
                cells.append(f"{int(v):>9d}")
            elif key == "fraction_within_interval":
                cells.append(f"{v * 100:>8.1f}%")
            else:
                cells.append(f"{v:>9.4f}")
        rows.append(f"{name:<{name_w}}  " + "  ".join(cells))
    return "\n".join(rows)


def load_regression_csv(
    csv_path: pathlib.Path,
    target_col: str,
    feature_cols: list[str] | None,
    id_col: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str] | None]:
    """Load a numeric regression CSV (no SMILES processing).

    Returns ``(X, y, feature_names, sample_ids)``. When
    ``feature_cols`` is ``None``, picks every numeric column other
    than ``target_col`` and ``id_col``.
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(
            f"target column {target_col!r} not in CSV "
            f"(columns: {list(df.columns)[:20]})"
        )
    if feature_cols is None:
        excluded = {target_col}
        if id_col:
            excluded.add(id_col)
        feature_cols = [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature columns missing from CSV: {missing}"
        )

    sample_ids = (
        df[id_col].astype(str).tolist() if id_col and id_col in df.columns
        else None
    )
    keep = df[feature_cols + [target_col]].dropna()
    X = keep[feature_cols].to_numpy(dtype=float)
    y = keep[target_col].to_numpy(dtype=float)
    return X, y, list(feature_cols), sample_ids


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive LOOCV (MolActAgent + ML baselines) — "
            "supports both classification and regression."
        )
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the dataset CSV.")
    parser.add_argument(
        "--task", choices=["auto", "classification", "regression"],
        default="auto",
        help="Task type. Auto-detects from the label column dtype "
             "(continuous numeric -> regression).",
    )
    parser.add_argument("--background",
                        default="antitb",
                        help="For classification SMILES datasets: "
                             "'antitb' or 'bslt' (legacy keys). "
                             "For regression: path to a background "
                             "text file, e.g. "
                             "examples/penguins/background.txt.")
    parser.add_argument("--feature-descriptions", type=str, default=None,
                        help="Path to a JSON file with feature "
                             "descriptions (regression mode only).")
    parser.add_argument("--target-col", type=str, default=None,
                        help="Continuous target column (regression "
                             "mode).")
    parser.add_argument("--feature-cols", nargs="+", default=None,
                        help="Numeric feature columns (regression "
                             "mode). Defaults to every non-id, "
                             "non-target numeric column.")
    parser.add_argument("--id-col", type=str, default=None,
                        help="Identifier column (regression mode).")
    parser.add_argument("--k", type=int, default=3,
                        help="MolActAgent neighbors per class (default: 3).")
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Per-fold thread pool for triplet LLM calls.")
    parser.add_argument("--fold-workers", type=int, default=4,
                        help="Number of LOOCV folds to run in parallel "
                             "(each fold itself fans out triplet calls).")
    parser.add_argument("--llm-temperature", type=float, default=0.0,
                        help="Sampling temperature for evaluation runs "
                             "(default 0.0 for reproducibility).")
    parser.add_argument("--llm-profile", type=str, default="deepseek")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Run ML baselines only.")
    parser.add_argument("--output-dir", type=str,
                        default=str(_PROJECT_ROOT / "data"))
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for output filenames.")
    parser.add_argument(
        "--positive-class", type=str, default=None,
        help="Which user label is the 'positive' class for binary "
             "metrics (TP/FN/AUC). Auto-detected when the dataset has "
             "exactly two classes (defaults to 'active' if present, "
             "otherwise the lexicographically larger class).",
    )
    args = parser.parse_args()

    log_files = setup_logging(log_dir=_PROJECT_ROOT / "logs")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log:  {log_files['full']}")
    logger.info(f"Args: {vars(args)}")

    csv_path = pathlib.Path(args.csv).resolve()

    # ── decide task ──────────────────────────────────────────────
    if args.task == "regression" or (
        args.task == "auto" and args.target_col is not None
    ):
        return _main_regression(args, csv_path)
    # else: classification (existing flow below)

    df = pd.read_csv(csv_path)
    if args.background in ("antitb", "bslt"):
        background = (
            BACKGROUND_ANTITB if args.background == "antitb"
            else BACKGROUND_BSLT
        )
    else:
        # Path to a background text file.
        bg_path = pathlib.Path(args.background)
        background = (
            bg_path.read_text(encoding="utf-8") if bg_path.exists()
            else args.background
        )

    X, y, meta = build_dataset_from_dataframe(df, background, verbose=True)
    n = len(y)
    classes = sorted(np.unique(y).tolist())
    if len(classes) != 2:
        raise ValueError(
            f"This LOOCV evaluator computes binary metrics; the dataset "
            f"has {len(classes)} distinct classes ({classes}). Filter "
            "the data to two classes before running."
        )
    if args.positive_class is not None:
        if args.positive_class not in classes:
            raise ValueError(
                f"--positive-class {args.positive_class!r} is not in "
                f"the dataset; available classes are {classes}."
            )
        positive_class = args.positive_class
    elif "active" in classes:
        positive_class = "active"
    else:
        positive_class = classes[-1]
    negative_class = [c for c in classes if c != positive_class][0]

    n_pos = int((y == positive_class).sum())
    n_neg = int((y == negative_class).sum())
    logger.info(
        f"Loaded {n} samples ({n_pos} {positive_class}, "
        f"{n_neg} {negative_class}) and {X.shape[1]} features from "
        f"{csv_path.name}"
    )
    logger.info(
        f"Binary metrics will use positive_class={positive_class!r}, "
        f"negative_class={negative_class!r}"
    )

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_json = (
        output_dir / f"loocv_metrics_{csv_path.stem}_k{args.k}{tag_suffix}.json"
    )
    out_table = (
        output_dir / f"loocv_metrics_{csv_path.stem}_k{args.k}{tag_suffix}.txt"
    )

    all_metrics: dict[str, dict] = {}
    all_predictions: dict[str, list[dict]] = {}

    # ── ML baselines ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60 + "\n  ML baselines (LOOCV)\n" + "=" * 60)
    t0 = time.time()
    ml_metrics = run_ml_loocv(X, y, positive_class, negative_class)
    all_metrics.update(ml_metrics)
    logger.info(f"ML LOOCV finished in {time.time() - t0:.1f}s")

    if args.skip_llm:
        table = format_metrics_table(all_metrics)
        print("\n" + table)
        out_table.write_text(table, encoding="utf-8")
        out_json.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "metrics": all_metrics,
                    "n_samples": n,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "positive_class": positive_class,
                    "negative_class": negative_class,
                },
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        logger.info(f"Saved metrics → {out_json}")
        logger.info(f"Saved table   → {out_table}")
        return

    # ── MolActAgent LOOCV ─────────────────────────────────────────
    llm_cfg = load_llm_config(_PROJECT_ROOT, profile=args.llm_profile)
    logger.info(
        f"LLM profile={args.llm_profile!r} → model={llm_cfg['model_name']!r} "
        f"base_url={llm_cfg['base_url']!r}"
    )
    llm = LLMClient(**llm_cfg, timeout=120.0)

    logger.info("\n" + "=" * 60 + "\n  MolActAgent (LOOCV)\n" + "=" * 60)
    t0 = time.time()
    ma_metrics, ma_records = run_molactagent_loocv(
        X, y,
        meta["feature_names"], meta["feature_descriptions"],
        background, llm,
        k=args.k, metric=args.metric,
        max_workers=args.max_workers,
        llm_temperature=args.llm_temperature,
        fold_workers=args.fold_workers,
        positive_class=positive_class,
        negative_class=negative_class,
    )
    elapsed = time.time() - t0
    logger.info(f"MolActAgent LOOCV finished in {elapsed:.1f}s")
    all_metrics["MolActAgent"] = ma_metrics

    # Strip the heavy raw tuple_results from per-fold records when
    # writing the summary file (still keep label, score, conf, time
    # and the per-class top-feature attribution).
    ma_records_lite: list[dict] = []
    for r in ma_records:
        if r is None:
            continue
        raw = r["raw"] or {}
        rec_copy = {
            "fold": r["fold"],
            "true": r["true"],
            "pred": r["pred"],
            "score_pos": r["score_pos"],
            "confidence": r["confidence"],
            "elapsed_s": r["elapsed_s"],
            "vote_summary": raw.get("vote_summary", {}),
            "candidate_classes": raw.get("candidate_classes", []),
            "key_features_per_class": raw.get("key_features_per_class", {}),
        }
        ma_records_lite.append(rec_copy)
    all_predictions["MolActAgent"] = ma_records_lite

    # ── output ───────────────────────────────────────────────────
    table = format_metrics_table(all_metrics)
    print("\n" + table)
    out_table.write_text(table, encoding="utf-8")

    out_json.write_text(
        json.dumps(
            {
                "args": vars(args),
                "n_samples": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "positive_class": positive_class,
                "negative_class": negative_class,
                "background": background,
                "metrics": all_metrics,
                "molactagent_records": ma_records_lite,
                "llm_stats": llm.stats,
                "elapsed_molactagent_s": round(elapsed, 1),
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )

    logger.info(f"Saved metrics  → {out_json}")
    logger.info(f"Saved table    → {out_table}")
    llm.print_stats()


def _main_regression(args, csv_path: pathlib.Path) -> None:
    """Regression LOOCV branch: numeric CSV in, MAE/RMSE/R^2 out."""
    if args.target_col is None:
        raise ValueError(
            "Regression mode requires --target-col to identify the "
            "continuous numeric column to predict."
        )

    bg_path = pathlib.Path(args.background) if args.background else None
    if bg_path is not None and bg_path.exists():
        background = bg_path.read_text(encoding="utf-8")
    else:
        background = (
            args.background or
            "Continuous-target regression task. Use the supplied "
            "feature descriptions to interpret the references."
        )

    feature_descriptions: dict[str, str] = {}
    if args.feature_descriptions:
        fd_path = pathlib.Path(args.feature_descriptions)
        if fd_path.exists():
            feature_descriptions = json.loads(
                fd_path.read_text(encoding="utf-8")
            )

    X, y, feature_names, sample_ids = load_regression_csv(
        csv_path,
        target_col=args.target_col,
        feature_cols=args.feature_cols,
        id_col=args.id_col,
    )
    n = len(y)
    logger.info(
        f"Loaded {n} regression samples and {X.shape[1]} features "
        f"from {csv_path.name}; target range "
        f"[{y.min():.4g}, {y.max():.4g}], distinct values "
        f"{int(np.unique(y).size)}"
    )

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

    # ── ML regression baselines ──────────────────────────────────
    logger.info(
        "\n" + "=" * 60 +
        "\n  ML regression baselines (LOOCV)\n" + "=" * 60
    )
    t0 = time.time()
    ml_metrics = run_ml_loocv_regression(X, y)
    all_metrics.update(ml_metrics)
    logger.info(f"ML regression LOOCV finished in {time.time() - t0:.1f}s")

    if args.skip_llm:
        table = format_regression_metrics_table(all_metrics)
        print("\n" + table)
        out_table.write_text(table, encoding="utf-8")
        out_json.write_text(
            json.dumps({
                "args": vars(args),
                "task": "regression",
                "n_samples": n,
                "target_range": [float(y.min()), float(y.max())],
                "feature_names": feature_names,
                "metrics": all_metrics,
            }, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"Saved metrics → {out_json}")
        logger.info(f"Saved table   → {out_table}")
        return

    # ── MolActAgent regression LOOCV ──────────────────────────────
    llm_cfg = load_llm_config(_PROJECT_ROOT, profile=args.llm_profile)
    logger.info(
        f"LLM profile={args.llm_profile!r} -> "
        f"model={llm_cfg['model_name']!r} "
        f"base_url={llm_cfg['base_url']!r}"
    )
    llm = LLMClient(**llm_cfg, timeout=120.0)

    logger.info(
        "\n" + "=" * 60 +
        "\n  MolActAgent regression (LOOCV)\n" + "=" * 60
    )
    t0 = time.time()
    ma_metrics, ma_records = run_molactagent_loocv_regression(
        X, y,
        feature_names, feature_descriptions, background, llm,
        k=args.k, metric=args.metric,
        max_workers=args.max_workers,
        llm_temperature=args.llm_temperature,
        fold_workers=args.fold_workers,
    )
    elapsed = time.time() - t0
    logger.info(f"MolActAgent regression LOOCV finished in {elapsed:.1f}s")
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
            "extrapolation_flagged": raw.get("extrapolation_flagged", False),
            "predictions": raw.get("predictions", []),
            "driver_attribution": (raw.get("driver_attribution") or [])[:5],
        })

    table = format_regression_metrics_table(all_metrics)
    print("\n" + table)
    out_table.write_text(table, encoding="utf-8")

    out_json.write_text(
        json.dumps({
            "args": vars(args),
            "task": "regression",
            "n_samples": n,
            "target_range": [float(y.min()), float(y.max())],
            "feature_names": feature_names,
            "background": background,
            "metrics": all_metrics,
            "molactagent_records": ma_records_lite,
            "llm_stats": llm.stats,
            "elapsed_molactagent_s": round(elapsed, 1),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    logger.info(f"Saved metrics  -> {out_json}")
    logger.info(f"Saved table    -> {out_table}")
    llm.print_stats()


if __name__ == "__main__":
    main()
