import numpy as np
import warnings
import logging

logger = logging.getLogger(__name__)


def _is_regression_target(y_ref: np.ndarray) -> bool:
    """Decide whether ``y_ref`` should be treated as a regression target.

    Heuristic: numeric dtype with either non-integer values or more
    than 10 distinct values. Boolean and string targets are always
    classification. Integer-valued targets with few levels (≤10) are
    classification by default — including ordinal classification —
    because the LLM treats the label set as discrete categories.
    Callers who want to force regression on a small-cardinality
    integer target can pass ``task="regression"`` explicitly at the
    pipeline level.
    """
    if y_ref.dtype == bool:
        return False
    if not np.issubdtype(y_ref.dtype, np.number):
        return False
    finite = y_ref[~_isnan_safe(y_ref)] if y_ref.dtype.kind == "f" else y_ref
    if len(finite) == 0:
        return False
    n_distinct = int(np.unique(finite).size)
    try:
        all_integer = bool(np.all(finite == finite.astype(int)))
    except (TypeError, ValueError):
        all_integer = False
    if all_integer and n_distinct <= 10:
        return False
    return True


def _isnan_safe(arr: np.ndarray) -> np.ndarray:
    """``np.isnan`` that returns all-False for non-float dtypes."""
    if arr.dtype.kind in ("f", "c"):
        return np.isnan(arr)
    return np.zeros(arr.shape, dtype=bool)


def validate_inputs(X_ref, y_ref, X_test, feature_names=None):
    """Inner validator used by ``MolActAgent.fit``.

    Branches on ``y_ref`` dtype: numeric continuous targets are
    treated as a regression task (no class-count requirements;
    instead, no missing values, ≥3 distinct values, and a
    non-degenerate range); everything else is treated as a
    classification task with the existing per-class checks.
    """
    if X_ref.ndim != 2:
        raise ValueError(f"X_ref must be 2D, got shape {X_ref.shape}")
    if y_ref.ndim != 1:
        raise ValueError(f"y_ref must be 1D, got shape {y_ref.shape}")
    if X_ref.shape[0] != y_ref.shape[0]:
        raise ValueError(
            f"X_ref rows ({X_ref.shape[0]}) != y_ref length ({y_ref.shape[0]})"
        )

    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")
    if X_ref.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: X_ref has {X_ref.shape[1]}, "
            f"X_test has {X_test.shape[1]}"
        )

    if feature_names is not None and len(feature_names) != X_ref.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != "
            f"n_features ({X_ref.shape[1]})"
        )

    n_features = X_ref.shape[1]
    if n_features > 1000:
        warnings.warn(
            f"Very high-dimensional input ({n_features} features). "
            "This may significantly reduce LLM reasoning stability. "
            "Consider feature selection or providing feature descriptions.",
            UserWarning,
        )
    elif n_features > 200:
        warnings.warn(
            f"High-dimensional input ({n_features} features). "
            "LLM reasoning may be less stable. "
            "Consider providing feature descriptions.",
            UserWarning,
        )

    is_regression = _is_regression_target(y_ref)

    if is_regression:
        nan_mask = _isnan_safe(y_ref)
        if nan_mask.any():
            raise ValueError(
                f"y_ref (regression target) contains "
                f"{int(nan_mask.sum())} NaN value(s); every reference "
                "row must have a finite numeric target."
            )
        finite = y_ref.astype(float, copy=False)
        n_distinct = int(np.unique(finite).size)
        if n_distinct < 3:
            raise ValueError(
                f"y_ref (regression target) has only {n_distinct} "
                "distinct value(s). Regression needs at least 3 "
                "distinct values; consider treating this as a "
                "classification task instead."
            )
        target_min, target_max = float(finite.min()), float(finite.max())
        if target_max - target_min <= 0:
            raise ValueError(
                "y_ref (regression target) has a degenerate range "
                f"(min={target_min}, max={target_max}); cannot perform "
                "regression on a constant target."
            )
        logger.info(
            f"Input validated [regression]: {X_ref.shape[0]} ref samples, "
            f"{X_test.shape[0]} test samples, {n_features} features"
        )
        logger.info(
            f"Target stats: min={target_min:.4g}, max={target_max:.4g}, "
            f"mean={float(finite.mean()):.4g}, distinct={n_distinct}"
        )
        return

    # Classification path: any string-coercible class names are
    # acceptable; we only require at least 2 distinct classes and a
    # complete (non-missing) label per reference row. The agent treats
    # the actual class strings as first-class identifiers everywhere
    # downstream, so there is no internal "Pos"/"Neg" remapping.
    unique_labels, counts = np.unique(y_ref, return_counts=True)
    if any(
        lbl is None or (isinstance(lbl, float) and np.isnan(lbl))
        for lbl in unique_labels
    ):
        raise ValueError(
            "y_ref contains missing values; every reference row must "
            "have a non-empty class label."
        )
    if len(unique_labels) < 2:
        raise ValueError(
            f"y_ref must contain at least 2 distinct classes; got "
            f"{unique_labels.tolist()!r}. MolActAgent needs at least two "
            "classes to perform contrastive comparisons."
        )

    n_classes = len(unique_labels)
    if n_classes > 10:
        warnings.warn(
            f"Reference set contains {n_classes} distinct classes. "
            "Multi-class MolActAgent works best with up to ~5 classes; "
            "larger label sets force the LLM to weigh many candidates "
            "per call and may reduce reasoning stability.",
            UserWarning,
        )

    per_class = ", ".join(
        f"{lbl}={int(c)}" for lbl, c in zip(unique_labels, counts)
    )
    logger.info(
        f"Input validated [classification]: {X_ref.shape[0]} ref samples, "
        f"{X_test.shape[0]} test samples, {n_features} features, "
        f"{n_classes} classes"
    )
    logger.info(f"Per-class counts: {per_class}")
