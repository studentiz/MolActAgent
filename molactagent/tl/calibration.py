"""Confidence calibration diagnostics.

Inspired by the per-phase confidence-vs-error analysis in
``scripts/ccr_diagnose_and_recompute.py``. The two questions this
module answers:

1. Is the per-sample confidence well-calibrated against
   correctness? — :func:`expected_calibration_error`,
   :func:`confidence_calibration` (returning a reliability curve).
2. For binary tasks, what's the Brier score? — included in the
   :class:`CalibrationReport` output.

The functions are deliberately generic: they accept either a fold-
record list (the LOOCV output shape) or three parallel arrays
``(y_true, y_pred, confidence)`` so they slot into both the new
``ca.tl.loocv`` flow and any user-rolled prediction loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CalibrationReport:
    """Calibration metrics for a (predictions, confidences) pair.

    Attributes
    ----------
    n : int
        Number of samples used (refusals excluded for regression
        tasks; ``Inconclusive`` predictions are kept and treated as
        wrong for accuracy purposes — same convention as
        :mod:`molactagent.metrics`).
    expected_calibration_error : float
        Weighted mean of ``|accuracy(bin) - mean_confidence(bin)|``
        across confidence bins. Lower is better. Range [0, 1].
    brier_score : float | None
        Mean squared error between confidence and correctness
        (1/0). Lower is better. Range [0, 1]. ``None`` for
        regression tasks where "correctness" is not binary.
    bin_centers : np.ndarray
        Midpoint of each confidence bin (length ``n_bins``).
    bin_accuracy : np.ndarray
        Accuracy within each bin.
    bin_confidence : np.ndarray
        Mean confidence within each bin.
    bin_count : np.ndarray
        Number of samples in each bin.
    """

    n: int
    expected_calibration_error: float
    brier_score: float | None
    bin_centers: np.ndarray
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray
    bin_count: np.ndarray

    def reliability_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(bin_centers, bin_accuracy, bin_confidence)``
        with empty bins masked out — handy for plotting."""
        mask = self.bin_count > 0
        return (
            self.bin_centers[mask],
            self.bin_accuracy[mask],
            self.bin_confidence[mask],
        )

    def summary(self) -> str:
        lines = [
            f"Calibration report (n={self.n})",
            "-" * 40,
            f"  ECE          : {self.expected_calibration_error:.4f}",
        ]
        if self.brier_score is not None:
            lines.append(f"  Brier score  : {self.brier_score:.4f}")
        lines.append("")
        lines.append(
            f"  Reliability curve ({int((self.bin_count > 0).sum())} "
            "non-empty bins):"
        )
        for c, acc, conf, n in zip(
            self.bin_centers, self.bin_accuracy,
            self.bin_confidence, self.bin_count,
        ):
            if n == 0:
                continue
            lines.append(
                f"    bin@{c:.2f}  n={int(n):>3d}  "
                f"acc={acc:.3f}  conf_mean={conf:.3f}  "
                f"|gap|={abs(acc - conf):.3f}"
            )
        return "\n".join(lines)


def confidence_calibration(
    y_true: list | np.ndarray | None = None,
    y_pred: list | np.ndarray | None = None,
    confidence: list[float] | np.ndarray | None = None,
    *,
    fold_records: list[dict[str, Any]] | None = None,
    bins: int = 10,
) -> CalibrationReport:
    """Compute a calibration report.

    Either pass three parallel arrays ``(y_true, y_pred, confidence)``
    or pass ``fold_records=`` (the per-fold list from
    :func:`molactagent.tl.loocv` / :func:`molactagent.tl.kfold_cv`).

    For regression tasks (numeric ``y_true``), correctness is
    defined as "the prediction's absolute error is below the
    median absolute error across all predictions". This is a coarse
    binarisation but it lets the same machinery score regression
    confidence; for fine-grained regression calibration users
    should plot ``confidence`` vs ``|error|`` directly using the
    ``fold_records``.
    """
    if fold_records is not None:
        y_true = np.array(
            [r["true"] for r in fold_records], dtype=object,
        )
        y_pred = np.array(
            [r["pred"] for r in fold_records], dtype=object,
        )
        confidence = np.array(
            [r.get("confidence", 0.5) for r in fold_records], dtype=float,
        )
    else:
        if y_true is None or y_pred is None or confidence is None:
            raise ValueError(
                "Either pass fold_records= OR all three of "
                "y_true / y_pred / confidence."
            )
        y_true = np.asarray(y_true, dtype=object)
        y_pred = np.asarray(y_pred, dtype=object)
        confidence = np.asarray(confidence, dtype=float)

    n = len(y_true)
    if n == 0:
        return CalibrationReport(
            n=0,
            expected_calibration_error=0.0,
            brier_score=None,
            bin_centers=np.linspace(0.5 / bins, 1.0 - 0.5 / bins, bins),
            bin_accuracy=np.zeros(bins),
            bin_confidence=np.zeros(bins),
            bin_count=np.zeros(bins, dtype=int),
        )

    correct = _correctness_array(y_true, y_pred)

    # Mask out NaN confidences and refused predictions before
    # binning — they would warp the calibration metric.
    finite_mask = np.isfinite(confidence)
    if finite_mask.sum() == 0:
        return CalibrationReport(
            n=0,
            expected_calibration_error=0.0,
            brier_score=None,
            bin_centers=np.linspace(0.5 / bins, 1.0 - 0.5 / bins, bins),
            bin_accuracy=np.zeros(bins),
            bin_confidence=np.zeros(bins),
            bin_count=np.zeros(bins, dtype=int),
        )

    confidence = confidence[finite_mask]
    correct = correct[finite_mask]

    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    bin_accuracy = np.zeros(bins)
    bin_confidence = np.zeros(bins)
    bin_count = np.zeros(bins, dtype=int)
    ece = 0.0
    for i in range(bins):
        # Half-open intervals; the last bin includes its right edge
        # so confidence==1.0 lands in the top bin instead of nowhere.
        if i < bins - 1:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        bin_count[i] = int(mask.sum())
        if bin_count[i] == 0:
            continue
        bin_accuracy[i] = float(correct[mask].mean())
        bin_confidence[i] = float(confidence[mask].mean())
        ece += (bin_count[i] / len(confidence)) * abs(
            bin_accuracy[i] - bin_confidence[i]
        )

    # Brier score: only meaningful when the underlying labels are
    # binary (or interpretable as binary correctness, which here
    # they always are because we just binarised regression too).
    brier = float(np.mean((confidence - correct) ** 2))

    return CalibrationReport(
        n=int(len(confidence)),
        expected_calibration_error=float(ece),
        brier_score=brier,
        bin_centers=centers,
        bin_accuracy=bin_accuracy,
        bin_confidence=bin_confidence,
        bin_count=bin_count,
    )


def expected_calibration_error(
    y_true: list | np.ndarray | None = None,
    y_pred: list | np.ndarray | None = None,
    confidence: list[float] | np.ndarray | None = None,
    *,
    fold_records: list[dict[str, Any]] | None = None,
    bins: int = 10,
) -> float:
    """Convenience scalar — just the ECE number."""
    return confidence_calibration(
        y_true, y_pred, confidence,
        fold_records=fold_records, bins=bins,
    ).expected_calibration_error


def _correctness_array(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> np.ndarray:
    """Build a 1/0 correctness array.

    Classification: ``y_pred[i] == y_true[i]`` (string equality;
    ``Inconclusive`` predictions are wrong by construction).

    Regression: ``|y_pred[i] - y_true[i]| < median_error``. Refused
    predictions (``None`` / ``NaN``) count as wrong. This
    binarisation is coarse but lets the same calibration machinery
    serve both tasks.

    Task detection uses the same heuristic as
    :func:`molactagent.tl.evaluation._detect_task`:

    - Numeric ``y_true`` with > 10 distinct values OR any
      non-integer values → regression.
    - Everything else (strings, booleans, low-cardinality integer
      labels) → classification.

    The previous implementation tried to coerce every input to
    float and treated 0/1 binary integer labels as regression —
    which silently produced nonsensical "correct if abs(error) <
    median(abs(error))" results for binary tasks. We now route to
    the same ``_detect_task`` rule the rest of the package uses.
    """
    from .evaluation import _detect_task

    n = len(y_true)
    correct = np.zeros(n, dtype=float)

    # Use the package-wide task detector so calibration agrees
    # with the rest of the system on what counts as "regression".
    try:
        task = _detect_task(np.asarray(y_true))
    except Exception:
        # If detection itself fails (rare; usually mixed dtypes),
        # fall back to classification semantics — string equality
        # is well-defined for any input.
        task = "classification"

    if task == "regression":
        try:
            yt_f = np.array([float(t) for t in y_true], dtype=float)
            yp_f = np.array(
                [float(p) if p is not None else np.nan for p in y_pred],
                dtype=float,
            )
        except (TypeError, ValueError):
            # Couldn't coerce — fall through to classification.
            yt_f = yp_f = None

        if yt_f is not None and yp_f is not None:
            finite = np.isfinite(yt_f) & np.isfinite(yp_f)
            if finite.sum() >= 3:
                errs = np.abs(yt_f[finite] - yp_f[finite])
                median_err = float(np.median(errs))
                for i in range(n):
                    if finite[i]:
                        correct[i] = 1.0 if abs(yt_f[i] - yp_f[i]) < median_err else 0.0
                    else:
                        correct[i] = 0.0
                return correct
            # Fewer than 3 finite paired values — degenerate;
            # everything counts as wrong because we have no
            # baseline error scale.
            return correct

    # Classification interpretation (the default and the fallback
    # path for every input that isn't unambiguously regression).
    for i in range(n):
        correct[i] = 1.0 if str(y_pred[i]) == str(y_true[i]) else 0.0
    return correct
