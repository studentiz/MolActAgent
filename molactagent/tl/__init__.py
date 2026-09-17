"""Downstream analysis tools — the Scanpy-style ``ca.tl`` namespace.

Every function here operates on top of fitted estimators or raw
predictions; none of them touch the engine internals. The split is:

- :mod:`evaluation` — leave-one-out and k-fold cross-validation
  runners that work uniformly across classification and regression.
- :mod:`baselines` — sklearn baseline classifiers and regressors,
  with the same LOOCV / CV plumbing for head-to-head comparisons.
- :mod:`calibration` — confidence-vs-correctness diagnostics
  (port of the analyses in
  ``scripts/ccr_diagnose_and_recompute.py``).
- :mod:`attribution` — cohort-level driver attribution synthesis
  across many predictions (the cross-test-set version of the
  per-sample driver list returned by the engine).

These tools are import-light: they pull in scikit-learn lazily inside
function bodies so that ``import molactagent`` itself stays fast.
"""

from .attribution import aggregate_drivers
from .baselines import compare_baselines
from .calibration import (
    confidence_calibration,
    expected_calibration_error,
)
from .evaluation import LOOCVResult, kfold_cv, loocv

__all__ = [
    "loocv",
    "kfold_cv",
    "LOOCVResult",
    "compare_baselines",
    "confidence_calibration",
    "expected_calibration_error",
    "aggregate_drivers",
]
