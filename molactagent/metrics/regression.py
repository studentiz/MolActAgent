"""Regression metrics — MAE, RMSE, R², Pearson r, Spearman r,
median AE, refusal-aware fraction-within-interval.

Direct port of :func:`compute_regression_metrics` from
``scripts/run_loocv_metrics.py`` packaged into a typed
:class:`RegressionReport`. Refusal-aware: rows where the predictor
returned ``None`` (or ``np.nan``) are excluded from the point-error
metrics but counted separately as ``n_refused`` so the user can
see how often the system declined to answer.

The Pearson and Spearman correlations are computed via SciPy when
SciPy is available; otherwise we fall back to a numpy-only
implementation so importing :mod:`molactagent.metrics.regression`
never fails on a SciPy-less install. (SciPy is already a transitive
dependency of scikit-learn, so the fallback is rarely hit, but
keeping the door open matters for deployments that pin minimal
dependency sets.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class RegressionReport:
    """Typed wrapper around a regression-metrics dict.

    Mirrors the legacy schema produced by
    ``compute_regression_metrics`` in ``scripts/run_loocv_metrics.py``
    so JSON files written by either path remain interchangeable.
    """

    n: int
    n_predicted: int
    n_refused: int
    mae: float | None
    rmse: float | None
    r2: float | None
    pearson_r: float | None
    spearman_r: float | None
    median_abs_err: float | None
    fraction_within_interval: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "n_predicted": int(self.n_predicted),
            "n_refused": int(self.n_refused),
            "mae": _maybe_float(self.mae),
            "rmse": _maybe_float(self.rmse),
            "r2": _maybe_float(self.r2),
            "pearson_r": _maybe_float(self.pearson_r),
            "spearman_r": _maybe_float(self.spearman_r),
            "median_abs_err": _maybe_float(self.median_abs_err),
            "fraction_within_interval": _maybe_float(
                self.fraction_within_interval
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"metric": k, "value": v} for k, v in self.to_dict().items()]
        )

    def to_table(self) -> str:
        lines = []
        title = (
            f"Regression metrics (n={self.n}, predicted="
            f"{self.n_predicted}, refused={self.n_refused})"
        )
        lines.append(title)
        lines.append("-" * len(title))

        def _row(label, value, fmt=".4g"):
            if value is None:
                v = "—"
            elif isinstance(value, float):
                v = format(value, fmt)
            else:
                v = str(value)
            lines.append(f"  {label:<22}  {v}")

        _row("MAE", self.mae)
        _row("RMSE", self.rmse)
        _row("R²", self.r2)
        _row("Pearson r", self.pearson_r)
        _row("Spearman r", self.spearman_r)
        _row("Median absolute err", self.median_abs_err)
        _row("Fraction within interval", self.fraction_within_interval)
        return "\n".join(lines)


def _maybe_float(v):
    return None if v is None else float(v)


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float | None] | np.ndarray,
    intervals: Sequence[tuple[float, float] | None] | None = None,
) -> RegressionReport:
    """Compute the full regression metric suite.

    Parameters
    ----------
    y_true :
        Ground-truth target values, finite floats.
    y_pred :
        Predicted target values. Refused predictions are encoded as
        ``None`` or ``np.nan`` and are excluded from the point-error
        metrics.
    intervals :
        Optional per-prediction ``(lo, hi)`` empirical intervals.
        When provided, ``fraction_within_interval`` is the fraction
        of predicted samples whose true value falls inside the
        bracket (refusals are excluded from the denominator).

    Notes
    -----
    Pearson and Spearman are returned as ``None`` when there are
    fewer than 3 paired finite values OR when either side has zero
    variance (a constant series has no meaningful correlation).
    """
    yt_arr = np.asarray(list(y_true), dtype=float)
    yp_list = list(y_pred)
    iv_list = (
        list(intervals) if intervals is not None
        else [None] * len(yt_arr)
    )

    paired = []
    for t, p, iv in zip(yt_arr, yp_list, iv_list):
        # ``p is None`` covers the explicit refusal sentinel; the
        # ``np.isfinite`` check covers NaN-encoded refusals from the
        # estimator wrapper.
        if p is None:
            continue
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pf):
            continue
        paired.append((float(t), pf, iv))

    n_total = len(yt_arr)
    n_predicted = len(paired)
    n_refused = n_total - n_predicted

    if n_predicted == 0:
        return RegressionReport(
            n=int(n_total), n_predicted=0, n_refused=int(n_refused),
            mae=None, rmse=None, r2=None,
            pearson_r=None, spearman_r=None,
            median_abs_err=None,
            fraction_within_interval=None,
        )

    yt = np.array([t for t, _, _ in paired], dtype=float)
    yp = np.array([p for _, p, _ in paired], dtype=float)

    # Lazy sklearn import — avoids paying the cost on simple
    # ``import molactagent.metrics.regression`` callers.
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    median_abs = float(median_absolute_error(yt, yp))

    r2 = float(r2_score(yt, yp)) if n_predicted >= 2 else None

    if (
        n_predicted >= 3
        and float(yt.std()) > 0
        and float(yp.std()) > 0
    ):
        pr = _pearson(yt, yp)
        sr = _spearman(yt, yp)
    else:
        pr = sr = None

    if intervals is not None:
        within: list[int] = []
        for t, _, iv in paired:
            if iv is None:
                continue
            try:
                lo, hi = float(iv[0]), float(iv[1])
            except (TypeError, ValueError, IndexError):
                continue
            within.append(1 if lo <= float(t) <= hi else 0)
        frac_within = (
            float(sum(within) / len(within)) if within else None
        )
    else:
        frac_within = None

    return RegressionReport(
        n=int(n_total),
        n_predicted=int(n_predicted),
        n_refused=int(n_refused),
        mae=mae,
        rmse=rmse,
        r2=r2,
        pearson_r=pr,
        spearman_r=sr,
        median_abs_err=median_abs,
        fraction_within_interval=frac_within,
    )


# ── correlation helpers (SciPy if available, numpy fallback) ────────


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. Try SciPy first; numpy fallback otherwise."""
    try:
        from scipy.stats import pearsonr
        return float(pearsonr(x, y).statistic)
    except Exception:
        # Manual Pearson — same formula the engine uses inside
        # ``cross_classification._synthesize_driver_attribution``.
        cov = float(np.mean((x - x.mean()) * (y - y.mean())))
        denom = float(x.std() * y.std())
        if denom == 0:
            return 0.0
        return cov / denom


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. SciPy first, then numpy fallback
    via Pearson on ranks."""
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)
    except Exception:
        # ``argsort.argsort`` produces 0-indexed ranks; we don't
        # bother tie-breaking because Spearman tolerates the
        # average-of-ranks approximation in practice and the SciPy
        # path is the documented default.
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return _pearson(rx, ry)
