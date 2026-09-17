"""Comprehensive metric suite for classification and regression.

These metric helpers are the package-internal counterpart of the
metric functions previously stranded inside
``scripts/run_loocv_metrics.py``. Promoting them to a real
sub-package buys two things:

1. They become importable from inside the package without
   ``sys.path`` games — the engine and the new estimators no
   longer have to depend on the scripts directory.
2. They get the same testing discipline as the rest of the
   codebase, so the metric formulas stop drifting silently.

The module is split into ``classification`` and ``regression``
submodules so a regression-only user does not have to import
sklearn classifiers transitively, and vice versa. Both submodules
expose a single small ``Report`` dataclass with named fields,
``to_table()`` for fixed-width text output, ``to_dict()`` for JSON
serialisation, and ``to_dataframe()`` for analysis pipelines.

For multi-method comparison (e.g. MolActAgent vs sklearn baselines),
see :func:`format_comparison_table`.
"""

from .classification import (
    ClassificationReport,
    binary_classification_metrics,
    classification_metrics,
)
from .regression import (
    RegressionReport,
    regression_metrics,
)
from .tables import format_comparison_table

__all__ = [
    "ClassificationReport",
    "RegressionReport",
    "classification_metrics",
    "binary_classification_metrics",
    "regression_metrics",
    "format_comparison_table",
]
