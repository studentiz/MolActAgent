"""Cross-validation runners for MolActAgent estimators.

This module ports the per-fold orchestration from
``scripts/run_loocv_metrics.py`` into a generic, estimator-agnostic
form. The estimator-agnostic part matters because the same runner
should work for both:

- :class:`molactagent.estimators.MolActAgentClassifier` /
  :class:`molactagent.estimators.MolActAgentRegressor` — the LLM-backed
  estimators.
- ``sklearn.base.ClassifierMixin`` / ``sklearn.base.RegressorMixin``
  subclasses — for head-to-head baseline comparisons.

The estimator's ``fit(X, y)`` -> ``predict(X)`` shape is everything
the runner needs; whether the underlying engine is an LLM or a
random forest is opaque.

Parallelism: the runner deliberately does NOT add another layer of
parallelism on top of MolActAgent's per-tuple thread pool. For
LLM-backed estimators we expose ``fold_workers`` to run multiple
folds concurrently (each fold itself fans out tuple calls); for
sklearn baselines parallelism is whatever the underlying classifier
provides (e.g. ``n_jobs`` on RF). This keeps token budgets
predictable and avoids fighting between layers of thread pools.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..metrics import (
    ClassificationReport,
    RegressionReport,
    classification_metrics,
    regression_metrics,
)

logger = logging.getLogger(__name__)


# ── result container ──────────────────────────────────────────────


@dataclass
class LOOCVResult:
    """Carries the output of a LOOCV (or k-fold CV) run.

    Holds both the aggregated metrics and the per-fold records so
    downstream consumers can compute their own variants
    (calibration plots, fold-grouped confidence histograms, etc.)
    without re-running the loop.

    Attributes
    ----------
    task : "classification" or "regression"
    n_folds : int
    metrics : ClassificationReport | RegressionReport
        Aggregate metrics over all folds.
    fold_records : list[dict]
        One record per fold: ``{fold, true, pred, confidence,
        elapsed_s, score, raw}``. ``raw`` is the engine's full
        per-sample dict (when the estimator exposes
        ``predict_full``); otherwise an empty dict.
    predictions : np.ndarray
        Convenience alias for the per-fold predicted values.
    confidences : np.ndarray
        Aligned per-fold confidence scores.
    elapsed_s : float
        Wall-clock time for the whole CV run.
    """

    task: str
    n_folds: int
    metrics: ClassificationReport | RegressionReport
    fold_records: list[dict[str, Any]] = field(default_factory=list)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    confidences: np.ndarray = field(default_factory=lambda: np.array([]))
    elapsed_s: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Per-fold records as a tidy DataFrame.

        Drops the heavyweight ``raw`` column so the resulting frame
        is small enough to log / print without flooding the screen.
        """
        rows = []
        for r in self.fold_records:
            rows.append({
                k: v for k, v in r.items() if k != "raw"
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary for the CLI / notebook."""
        lines = [
            f"LOOCV ({self.task}) — {self.n_folds} folds, "
            f"{self.elapsed_s:.1f}s total",
            "",
            self.metrics.to_table(),
        ]
        return "\n".join(lines)


# ── leave-one-out CV ──────────────────────────────────────────────


def loocv(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    fold_workers: int = 1,
    progress: bool = True,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    positive_class: str | None = None,
    negative_class: str | None = None,
) -> LOOCVResult:
    """Leave-one-out cross-validation.

    Parameters
    ----------
    estimator :
        A MolActAgent estimator (or any sklearn estimator with the
        same ``fit`` / ``predict`` shape). The estimator is
        ``copy()``-ed for each fold so per-fold ``fit`` calls do
        not mutate the caller's instance. MolActAgent estimators
        only carry hyperparameters and a reference to the LLM
        client at construction time — both copy cleanly.
    X : (n, d) array
    y : (n,) array
    fold_workers :
        How many folds to run in parallel. For LLM-backed
        estimators each fold fans out its own tuple calls; the
        product ``fold_workers * estimator.max_workers`` controls
        the total live LLM calls. Defaults to 1 (sequential) so
        token spend stays predictable.
    progress :
        When True, log per-fold progress at INFO level.
    progress_callback :
        Optional ``fn(i_done, n_total, record)`` invoked after
        each fold completes.
    positive_class, negative_class :
        Forwarded to :func:`molactagent.metrics.classification_metrics`
        when the underlying task is binary classification.

    Returns
    -------
    :class:`LOOCVResult`

    Notes
    -----
    The task (classification vs. regression) is auto-detected from
    the y dtype:
    - numeric with > 10 distinct values OR non-integer values →
      regression.
    - everything else → classification.
    The detection mirrors :func:`molactagent.io.detect_task` so
    behaviour is consistent across the package.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]}."
        )

    task = _detect_task(y)
    n = len(y)
    fold_records: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]

    t_start = time.time()
    completed = [0]

    def _on_done(rec: dict[str, Any]) -> None:
        completed[0] += 1
        if progress:
            pred = rec.get("pred")
            if isinstance(pred, float):
                pred_str = f"{pred:.4g}" if np.isfinite(pred) else "REFUSED"
            else:
                pred_str = str(pred)
            logger.info(
                f"  fold {rec['fold'] + 1}/{n} done in "
                f"{rec['elapsed_s']:.1f}s  true={rec['true']!r}  "
                f"pred={pred_str}  conf={rec.get('confidence', 0):.2f}  "
                f"({completed[0]}/{n} folds finished)"
            )
        if progress_callback is not None:
            try:
                progress_callback(completed[0], n, rec)
            except Exception:
                logger.debug(
                    "progress_callback raised", exc_info=True,
                )

    if fold_workers > 1:
        with ThreadPoolExecutor(max_workers=fold_workers) as pool:
            futures = {
                pool.submit(_run_one_fold, estimator, X, y, i, task): i
                for i in range(n)
            }
            for fut in as_completed(futures):
                rec = fut.result()
                fold_records[rec["fold"]] = rec
                _on_done(rec)
    else:
        for i in range(n):
            rec = _run_one_fold(estimator, X, y, i, task)
            fold_records[i] = rec
            _on_done(rec)

    # Aggregate.
    if task == "classification":
        y_true = [r["true"] for r in fold_records]
        y_pred = [r["pred"] for r in fold_records]
        proba_pos: list[float] | None = None
        # For binary classification, the per-fold record carries a
        # ``score_pos`` field that we can wire into ROC AUC / PR AUC.
        if all(r.get("score_pos") is not None for r in fold_records):
            proba_pos = [r["score_pos"] for r in fold_records]
        report: ClassificationReport | RegressionReport = classification_metrics(
            y_true,
            y_pred,
            proba=np.array(proba_pos) if proba_pos is not None else None,
            positive_class=positive_class,
            negative_class=negative_class,
        )
        predictions = np.array(y_pred, dtype=object)
    else:
        y_true_f = [r["true"] for r in fold_records]
        y_pred_f = [r["pred"] for r in fold_records]
        intervals = [r.get("interval") for r in fold_records]
        report = regression_metrics(y_true_f, y_pred_f, intervals=intervals)
        predictions = np.array(
            [np.nan if p is None else float(p) for p in y_pred_f],
            dtype=float,
        )

    confidences = np.array(
        [r.get("confidence", 0.0) for r in fold_records], dtype=float,
    )

    return LOOCVResult(
        task=task,
        n_folds=n,
        metrics=report,
        fold_records=fold_records,
        predictions=predictions,
        confidences=confidences,
        elapsed_s=time.time() - t_start,
    )


def kfold_cv(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
    fold_workers: int = 1,
    progress: bool = True,
    positive_class: str | None = None,
    negative_class: str | None = None,
) -> LOOCVResult:
    """Stratified k-fold CV (classification) or k-fold CV (regression).

    Same return contract as :func:`loocv`; ``n_folds`` reflects the
    actual number of splits the underlying CV iterator produced.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y)
    task = _detect_task(y)

    if task == "classification":
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state,
        )
        splits = list(cv.split(X, y))
    else:
        cv = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state,
        )
        splits = list(cv.split(X))

    n_folds = len(splits)
    t_start = time.time()
    fold_records: list[dict[str, Any]] = []

    completed = [0]
    for fi, (train_idx, test_idx) in enumerate(splits):
        if fold_workers > 1:
            with ThreadPoolExecutor(max_workers=fold_workers) as pool:
                # Within a fold we run each test sample as its own
                # virtual "sub-fold" to reuse _run_one_fold's
                # bookkeeping. For sklearn estimators this is
                # cheaper to do as a single ``fit`` + batch
                # ``predict``; we take the simpler-but-slower path
                # so the runner stays uniform across estimator
                # families.
                futures = [
                    pool.submit(
                        _run_one_test_in_fold,
                        estimator, X, y, train_idx, ti, fi, task,
                    )
                    for ti in test_idx
                ]
                for fut in as_completed(futures):
                    fold_records.append(fut.result())
                    completed[0] += 1
        else:
            for ti in test_idx:
                rec = _run_one_test_in_fold(
                    estimator, X, y, train_idx, ti, fi, task,
                )
                fold_records.append(rec)
                completed[0] += 1
        if progress:
            logger.info(
                f"k-fold CV: fold {fi + 1}/{n_folds} done "
                f"({completed[0]} test samples processed)"
            )

    # Sort by ``fold`` then by original ``test_index`` for stable order.
    fold_records.sort(key=lambda r: (r["fold"], r.get("test_index", 0)))

    if task == "classification":
        y_true = [r["true"] for r in fold_records]
        y_pred = [r["pred"] for r in fold_records]
        report: ClassificationReport | RegressionReport = classification_metrics(
            y_true, y_pred,
            positive_class=positive_class,
            negative_class=negative_class,
        )
        predictions = np.array(y_pred, dtype=object)
    else:
        y_true_f = [r["true"] for r in fold_records]
        y_pred_f = [r["pred"] for r in fold_records]
        intervals = [r.get("interval") for r in fold_records]
        report = regression_metrics(y_true_f, y_pred_f, intervals=intervals)
        predictions = np.array(
            [np.nan if p is None else float(p) for p in y_pred_f],
            dtype=float,
        )

    confidences = np.array(
        [r.get("confidence", 0.0) for r in fold_records], dtype=float,
    )

    return LOOCVResult(
        task=task,
        n_folds=n_folds,
        metrics=report,
        fold_records=fold_records,
        predictions=predictions,
        confidences=confidences,
        elapsed_s=time.time() - t_start,
    )


# ── per-fold workers ──────────────────────────────────────────────


def _run_one_fold(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    fold_idx: int,
    task: str,
) -> dict[str, Any]:
    """Standard LOOCV fold: hold out one row, fit on the rest, predict."""
    n = len(y)
    train_idx = np.array([j for j in range(n) if j != fold_idx], dtype=int)
    return _fit_and_predict_one(
        estimator, X, y, train_idx, fold_idx, fold_idx, task,
    )


def _run_one_test_in_fold(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: int,
    fold_idx: int,
    task: str,
) -> dict[str, Any]:
    """k-fold CV inner worker — fit on ``train_idx``, predict on
    ``test_idx``."""
    return _fit_and_predict_one(
        estimator, X, y, train_idx, test_idx, fold_idx, task,
    )


def _fit_and_predict_one(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: int,
    fold_idx: int,
    task: str,
) -> dict[str, Any]:
    """Shared per-(fold, test-sample) worker.

    ``copy(estimator)`` makes a shallow copy so each fold sees its
    own freshly-fitted estimator. MolActAgent estimators carry only
    hyperparameters (and a reference to the LLM client, which is
    thread-safe), so a shallow copy is sufficient.
    """
    est = _shallow_copy_estimator(estimator)
    X_train = X[train_idx]
    y_train = y[train_idx]

    t0 = time.time()
    est.fit(X_train, y_train)
    x_test = X[test_idx:test_idx + 1]

    rec: dict[str, Any] = {
        "fold": int(fold_idx),
        "test_index": int(test_idx),
        "true": _to_python(y[test_idx]),
        "elapsed_s": 0.0,
    }

    if task == "classification":
        # Try to extract per-class probability if the estimator
        # supports predict_proba — we'll wire it into the binary
        # AUC computations downstream.
        pred_label = str(est.predict(x_test)[0])
        score_pos = None
        confidence = None
        if hasattr(est, "predict_with_confidence"):
            _, conf_arr = est.predict_with_confidence(x_test)
            confidence = float(conf_arr[0])
        if hasattr(est, "predict_proba"):
            try:
                proba = est.predict_proba(x_test)[0]
                # Binary positive-class score: pick the column for
                # the lexicographically larger class (or "active"
                # if present, matching the script convention).
                classes = list(getattr(est, "classes_", []))
                if classes:
                    if "active" in classes:
                        pos_idx = classes.index("active")
                    elif len(classes) == 2:
                        pos_idx = 1  # second of two sorted classes
                    else:
                        pos_idx = None
                    if pos_idx is not None:
                        score_pos = float(proba[pos_idx])
            except Exception:
                score_pos = None

        # ``predict_full`` is a MolActAgent extension; sklearn
        # baselines won't have it.
        raw = {}
        if hasattr(est, "predict_full"):
            raw_results = est.predict_full(x_test)
            if raw_results:
                raw = raw_results[0].to_dict()

        rec.update({
            "pred": pred_label,
            "score_pos": score_pos,
            "confidence": (
                confidence if confidence is not None else
                (score_pos if score_pos is not None else 0.5)
            ),
            "raw": raw,
        })
    else:
        if hasattr(est, "predict_with_interval"):
            preds, lo, hi = est.predict_with_interval(x_test)
            pv = float(preds[0])
            rec["pred"] = None if not np.isfinite(pv) else pv
            if np.isfinite(lo[0]) and np.isfinite(hi[0]):
                rec["interval"] = (float(lo[0]), float(hi[0]))
            else:
                rec["interval"] = None
        else:
            pv = float(est.predict(x_test)[0])
            rec["pred"] = None if not np.isfinite(pv) else pv
            rec["interval"] = None

        if hasattr(est, "predict_with_confidence"):
            _, conf_arr = est.predict_with_confidence(x_test)
            rec["confidence"] = float(conf_arr[0])
        else:
            rec["confidence"] = 0.5

        raw = {}
        if hasattr(est, "predict_full"):
            raw_results = est.predict_full(x_test)
            if raw_results:
                raw = raw_results[0].to_dict()
        rec["raw"] = raw

    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


# ── helpers ────────────────────────────────────────────────────────


def _detect_task(y: np.ndarray) -> str:
    """Mirror :func:`molactagent.io.detect_task` for arrays.

    We re-implement instead of importing because the script-side
    helper takes a pandas Series and we want to avoid forcing pandas
    on this code path.
    """
    arr = np.asarray(y)
    if arr.dtype == bool:
        return "classification"
    if not np.issubdtype(arr.dtype, np.number):
        return "classification"
    finite = arr[~np.isnan(arr.astype(float, copy=False))] if arr.dtype.kind == "f" else arr
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


def _shallow_copy_estimator(estimator: Any) -> Any:
    """Make a fresh estimator instance with the same hyperparameters.

    For MolActAgent estimators we use the public ``get_params`` /
    ``set_params`` round-trip — same machinery sklearn's clone uses.
    For arbitrary sklearn estimators we fall back to ``sklearn.base.clone``.
    """
    if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
        # Build a new instance of the same class with the same
        # hyperparameters. Avoid sklearn.base.clone here for
        # MolActAgent estimators because their ``llm_client`` /
        # ``llm_config`` are not "estimator parameters" in the
        # sklearn sense — clone would clone them too.
        try:
            new = type(estimator)()
            new.set_params(**estimator.get_params())
            return new
        except TypeError:
            # The estimator's __init__ requires positional args we
            # don't know — fall through to sklearn.clone.
            pass
    try:
        from sklearn.base import clone
        return clone(estimator)
    except Exception:
        # Last resort: shallow copy.
        return copy(estimator)


def _to_python(v: Any) -> Any:
    """Coerce numpy scalar to native Python type for clean records."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v
