"""ML baseline comparisons.

Direct port of ``_make_ml_classifiers`` / ``_make_ml_regressors``
and the per-fold loops from ``scripts/run_loocv_metrics.py``,
restructured so the user can:

- pick a curated list (``baselines=["KNN3", "RF", "LR"]``),
- ask for everything (``baselines="all"`` / ``"all-clf"`` /
  ``"all-reg"``),
- pass their own sklearn estimators
  (``baselines={"MyTree": DecisionTreeClassifier()}``),
- omit MolActAgent entirely (``include_molactagent=False``) for
  ML-only sanity checks that don't burn LLM tokens.

Returns a typed :class:`BaselineComparison` whose
``metrics_table()`` reproduces the legacy fixed-width comparison
output exactly. Per-method per-fold predictions are kept so the
caller can do their own analyses (calibration curves, fold-level
error breakdowns, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import (
    ClassificationReport,
    RegressionReport,
    format_comparison_table,
)
from .evaluation import LOOCVResult, _detect_task, kfold_cv, loocv

logger = logging.getLogger(__name__)


# ── curated baseline catalogues ────────────────────────────────────
#
# Defaults match the script's catalogue. Importable so callers can
# hand-pick a subset.

_DEFAULT_CLF_BASELINES = ["KNN3", "RF", "LR", "SVM"]
_ALL_CLF_BASELINES = ["KNN1", "KNN3", "KNN5", "NB", "RF", "SVM", "LR", "GBC"]

_DEFAULT_REG_BASELINES = ["KNN3", "Ridge", "RF"]
_ALL_REG_BASELINES = [
    "KNN1", "KNN3", "KNN5", "Linear", "Ridge", "Lasso", "RF", "GBR",
]


def _build_classifier(name: str, y_train: np.ndarray) -> Any:
    """Return a fresh classifier instance.

    The factories match those in ``scripts/run_loocv_metrics.py`` so
    LOOCV numbers stay comparable with previously-cached results.
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    factories: dict[str, Any] = {
        "KNN1": lambda: KNeighborsClassifier(n_neighbors=1),
        "KNN3": lambda: KNeighborsClassifier(n_neighbors=3),
        "KNN5": lambda: KNeighborsClassifier(n_neighbors=5),
        "NB": lambda: GaussianNB(),
        "RF": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "SVM": lambda: SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42,
        ),
        "LR": lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "GBC": lambda: GradientBoostingClassifier(
            n_estimators=200, random_state=42,
        ),
    }
    # XGBoost is optional — match the original script's policy.
    try:
        from xgboost import XGBClassifier
        y_bin = np.array(
            [1 if l == "Pos" or l == "active" else 0 for l in y_train]
        )
        pos = int(y_bin.sum())
        neg = len(y_bin) - pos
        scale = neg / pos if pos > 0 else 1.0
        factories["XGB"] = lambda: XGBClassifier(
            n_estimators=200, scale_pos_weight=scale,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
    except ImportError:  # pragma: no cover - depends on env
        pass

    if name not in factories:
        raise KeyError(
            f"Unknown classifier baseline {name!r}; choose from "
            f"{sorted(factories.keys())}."
        )
    return factories[name]()


def _build_regressor(name: str) -> Any:
    """Return a fresh regressor instance."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor

    factories: dict[str, Any] = {
        "KNN1": lambda: KNeighborsRegressor(n_neighbors=1),
        "KNN3": lambda: KNeighborsRegressor(n_neighbors=3),
        "KNN5": lambda: KNeighborsRegressor(n_neighbors=5),
        "Linear": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=1.0),
        "Lasso": lambda: Lasso(alpha=0.1, max_iter=10000),
        "RF": lambda: RandomForestRegressor(
            n_estimators=200, random_state=42,
        ),
        "GBR": lambda: GradientBoostingRegressor(
            n_estimators=200, random_state=42,
        ),
    }
    try:
        from xgboost import XGBRegressor
        factories["XGB"] = lambda: XGBRegressor(
            n_estimators=200, random_state=42, verbosity=0,
        )
    except ImportError:  # pragma: no cover - depends on env
        pass

    if name not in factories:
        raise KeyError(
            f"Unknown regressor baseline {name!r}; choose from "
            f"{sorted(factories.keys())}."
        )
    return factories[name]()


# ── result container ──────────────────────────────────────────────


@dataclass
class BaselineComparison:
    """Per-method LOOCV/CV results bundled together."""

    task: str
    results: dict[str, LOOCVResult] = field(default_factory=dict)

    @property
    def methods(self) -> list[str]:
        return list(self.results.keys())

    def metrics_table(self) -> str:
        """Render the multi-method comparison table."""
        return format_comparison_table(
            {name: r.metrics for name, r in self.results.items()},
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy DataFrame: one row per (method, metric) pair."""
        rows = []
        for name, r in self.results.items():
            d = r.metrics.to_dict()
            for metric, value in d.items():
                if isinstance(value, (list, dict)):
                    continue
                rows.append({"method": name, "metric": metric, "value": value})
        return pd.DataFrame(rows)

    def best(self, metric: str, *, higher_is_better: bool = True) -> str:
        """Return the method name with the best score on ``metric``."""
        ranked: list[tuple[str, float]] = []
        for name, r in self.results.items():
            v = r.metrics.to_dict().get(metric)
            if v is None:
                continue
            ranked.append((name, float(v)))
        if not ranked:
            raise KeyError(f"No method produced a value for {metric!r}.")
        ranked.sort(key=lambda kv: kv[1], reverse=bool(higher_is_better))
        return ranked[0][0]


# ── main entry point ──────────────────────────────────────────────


def compare_baselines(
    molactagent_estimator: Any | None,
    X: np.ndarray,
    y: np.ndarray,
    *,
    baselines: list[str] | str | dict[str, Any] = "default",
    cv: str = "loocv",
    n_splits: int = 5,
    fold_workers: int = 1,
    include_molactagent: bool = True,
    progress: bool = True,
    positive_class: str | None = None,
    negative_class: str | None = None,
) -> BaselineComparison:
    """Run LOOCV (or k-fold CV) for MolActAgent + ML baselines.

    Parameters
    ----------
    molactagent_estimator :
        A fitted-or-unfitted :class:`molactagent.MolActAgentClassifier`
        / :class:`molactagent.MolActAgentRegressor`. Pass ``None``
        with ``include_molactagent=False`` for ML-only runs.
    X, y :
        The reference dataset.
    baselines :
        Either ``"default"`` (curated subset for the task),
        ``"all"`` / ``"all-clf"`` / ``"all-reg"`` (everything),
        a list of name strings, or a custom
        ``{name: estimator_factory_or_instance}`` dict. Custom
        estimators must implement the sklearn ``fit`` / ``predict``
        contract.
    cv :
        ``"loocv"`` or ``"kfold"``.
    n_splits :
        Number of folds for ``cv="kfold"``.
    fold_workers :
        Forwarded to the underlying CV runner. For ML baselines
        this should be 1 — sklearn estimators already use any
        ``n_jobs`` they were configured with.
    include_molactagent :
        Whether to also run the MolActAgent estimator. Set to
        ``False`` for quick ML-only sanity checks.
    progress :
        Per-fold INFO logging.
    positive_class, negative_class :
        Forwarded to the metrics functions for binary tasks.

    Returns
    -------
    :class:`BaselineComparison`
    """
    X = np.asarray(X)
    y = np.asarray(y)
    task = _detect_task(y)

    # Resolve the requested baseline set.
    if isinstance(baselines, dict):
        # Explicit dict of {name: estimator} — use as-is.
        baseline_estimators = {
            name: _ensure_instance(name, est)
            for name, est in baselines.items()
        }
    elif isinstance(baselines, str):
        names = _resolve_baseline_keyword(baselines, task)
        baseline_estimators = {
            name: _build_baseline(name, task, y) for name in names
        }
    else:
        # List of names.
        baseline_estimators = {
            name: _build_baseline(name, task, y) for name in baselines
        }

    cv = cv.lower()
    if cv not in ("loocv", "kfold"):
        raise ValueError(f"cv must be 'loocv' or 'kfold'; got {cv!r}.")

    out_results: dict[str, LOOCVResult] = {}

    def _run_one(name: str, estimator: Any) -> LOOCVResult:
        if cv == "loocv":
            return loocv(
                estimator, X, y,
                fold_workers=fold_workers,
                progress=progress,
                positive_class=positive_class,
                negative_class=negative_class,
            )
        return kfold_cv(
            estimator, X, y,
            n_splits=n_splits,
            fold_workers=fold_workers,
            progress=progress,
            positive_class=positive_class,
            negative_class=negative_class,
        )

    if include_molactagent:
        if molactagent_estimator is None:
            raise ValueError(
                "include_molactagent=True but molactagent_estimator is None. "
                "Pass an estimator or set include_molactagent=False."
            )
        logger.info("Running MolActAgent CV...")
        # MolActAgent failures should not be silenced — they almost
        # always indicate a configuration problem the user wants to
        # see (bad LLM credentials, missing background, etc.).
        out_results["MolActAgent"] = _run_one(
            "MolActAgent", molactagent_estimator,
        )

    # Baselines are wrapped in try/except: a single broken baseline
    # (e.g. XGBoost choking on string labels in a multi-class
    # context) should not bring down a five-method comparison.
    # Failures are logged at WARNING level and the method is
    # silently dropped from the result.
    for name, est in baseline_estimators.items():
        logger.info(f"Running baseline {name!r} CV...")
        try:
            out_results[name] = _run_one(name, est)
        except Exception as exc:
            logger.warning(
                f"Baseline {name!r} failed during CV and was "
                f"dropped from the comparison: {exc}"
            )

    return BaselineComparison(task=task, results=out_results)


# ── helpers ────────────────────────────────────────────────────────


def _resolve_baseline_keyword(keyword: str, task: str) -> list[str]:
    keyword = keyword.lower()
    if keyword == "default":
        return list(
            _DEFAULT_CLF_BASELINES if task == "classification"
            else _DEFAULT_REG_BASELINES
        )
    if keyword in ("all", "all-clf"):
        if task == "regression" and keyword == "all":
            return list(_ALL_REG_BASELINES)
        return list(_ALL_CLF_BASELINES)
    if keyword == "all-reg":
        return list(_ALL_REG_BASELINES)
    raise ValueError(
        f"Unknown baseline keyword {keyword!r}. Use 'default', "
        f"'all', 'all-clf', 'all-reg', a list, or a dict."
    )


def _build_baseline(name: str, task: str, y: np.ndarray) -> Any:
    return (
        _build_classifier(name, y) if task == "classification"
        else _build_regressor(name)
    )


def _ensure_instance(name: str, est_or_factory: Any) -> Any:
    """Allow callers to pass either an estimator instance or a
    zero-arg factory."""
    if hasattr(est_or_factory, "fit"):
        return est_or_factory
    if callable(est_or_factory):
        return est_or_factory()
    raise TypeError(
        f"Baseline {name!r} must be an estimator instance or a "
        "zero-arg factory; got "
        f"{type(est_or_factory).__name__}."
    )
