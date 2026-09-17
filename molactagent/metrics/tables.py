"""Multi-method comparison tables.

Direct port of :func:`format_metrics_table` and
:func:`format_regression_metrics_table` from
``scripts/run_loocv_metrics.py`` into a single polymorphic helper.
The output exactly matches the legacy table format so existing
``.txt`` artefacts in ``data/`` remain visually consistent.
"""

from __future__ import annotations

from typing import Any

from .classification import ClassificationReport
from .regression import RegressionReport


def format_comparison_table(
    results: dict[str, ClassificationReport | RegressionReport | dict[str, Any]],
) -> str:
    """Render a multi-method comparison table.

    Accepts a dict ``{method_name: report}`` where ``report`` may be:

    - a :class:`ClassificationReport` or :class:`RegressionReport`,
    - a raw dict matching their schemas (legacy path — what the
      old LOOCV scripts produced).

    Auto-detects whether every entry is a regression or
    classification report and dispatches to the matching formatter.
    Mixed dicts raise ``ValueError`` — comparing classification and
    regression methods in the same table would produce nonsensical
    rows.
    """
    if not results:
        return "(no methods to compare)"

    # Coerce raw dicts into reports for shape consistency.
    coerced: dict[str, ClassificationReport | RegressionReport] = {}
    for name, r in results.items():
        if isinstance(r, (ClassificationReport, RegressionReport)):
            coerced[name] = r
        elif isinstance(r, dict):
            # Heuristic: regression schema has ``mae``;
            # classification has ``balanced_accuracy``.
            if "mae" in r:
                coerced[name] = _regression_report_from_dict(r)
            elif "balanced_accuracy" in r:
                coerced[name] = _classification_report_from_dict(r)
            else:
                raise ValueError(
                    f"Cannot infer report kind for method "
                    f"{name!r}; supply a typed report."
                )
        else:
            raise TypeError(
                f"Unsupported report type for {name!r}: {type(r)}."
            )

    types = {type(r) for r in coerced.values()}
    if len(types) > 1:
        raise ValueError(
            "format_comparison_table received a mix of classification "
            "and regression reports; render them separately."
        )

    if isinstance(next(iter(coerced.values())), ClassificationReport):
        return _format_classification_table(coerced)
    return _format_regression_table(coerced)


def _classification_report_from_dict(d: dict[str, Any]) -> ClassificationReport:
    return ClassificationReport(
        n_samples=int(d.get("n_samples", 0)),
        n_classes=int(d.get("n_classes", 2)),
        accuracy=float(d.get("accuracy", 0.0)),
        balanced_accuracy=float(d.get("balanced_accuracy", 0.0)),
        mcc=float(d.get("mcc", 0.0)),
        macro_f1=float(d.get("macro_f1", 0.0)),
        unknown_rate=float(d.get("unknown_rate", 0.0)),
        confusion_matrix=d.get("confusion_matrix") or [],
        class_labels=list(d.get("class_labels") or []),
        positive_class=d.get("positive_class"),
        negative_class=d.get("negative_class"),
        precision_pos=d.get("precision_pos"),
        recall_pos=d.get("recall_pos"),
        f1_pos=d.get("f1_pos"),
        precision_neg=d.get("precision_neg"),
        recall_neg=d.get("recall_neg"),
        f1_neg=d.get("f1_neg"),
        specificity_neg=d.get("specificity_neg"),
        roc_auc=d.get("roc_auc"),
        pr_auc=d.get("pr_auc"),
        tp=d.get("tp"), tn=d.get("tn"),
        fp=d.get("fp"), fn=d.get("fn"),
        unknown=d.get("unknown"),
    )


def _regression_report_from_dict(d: dict[str, Any]) -> RegressionReport:
    return RegressionReport(
        n=int(d.get("n", 0)),
        n_predicted=int(d.get("n_predicted", 0)),
        n_refused=int(d.get("n_refused", 0)),
        mae=d.get("mae"),
        rmse=d.get("rmse"),
        r2=d.get("r2"),
        pearson_r=d.get("pearson_r"),
        spearman_r=d.get("spearman_r"),
        median_abs_err=d.get("median_abs_err"),
        fraction_within_interval=d.get("fraction_within_interval"),
    )


# ── classification table (binary-flavoured) ─────────────────────────


def _format_classification_table(
    results: dict[str, ClassificationReport],
) -> str:
    """Fixed-width comparison table — exactly matches the legacy
    ``format_metrics_table`` output."""
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
            row += _fmt(getattr(results[m], key, None))
        lines.append(row)
    lines.append(sep)

    # Confusion-matrix breakdown only emitted if at least one method
    # is binary (i.e. has populated TP/TN/FP/FN).
    have_binary = any(r.tp is not None for r in results.values())
    if have_binary:
        lines.append("Confusion matrix (TP / FP / FN / TN  + Unknown):")
        for m in methods:
            r = results[m]
            if r.tp is None:
                continue
            lines.append(
                f"  {m:<14s}  TP={r.tp:>2d}  FP={r.fp:>2d}  "
                f"FN={r.fn:>2d}  TN={r.tn:>2d}  Unk={r.unknown:>2d}"
                f"  (n={r.n_samples})"
            )
    return "\n".join(lines)


# ── regression table ───────────────────────────────────────────────


def _format_regression_table(
    results: dict[str, RegressionReport],
) -> str:
    """Fixed-width regression comparison table — exactly matches the
    legacy ``format_regression_metrics_table`` output."""
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
    name_w = max((len(n) for n in results.keys()), default=10)
    name_w = max(name_w, 12)
    header = f"{'Method':<{name_w}}  " + "  ".join(
        f"{label:>9}" for label, _ in cols
    )
    sep = "-" * len(header)

    rows = [header, sep]
    for name, m in results.items():
        cells = []
        for label, key in cols:
            v = getattr(m, key, None)
            if v is None:
                cells.append(f"{'—':>9}")
            elif key == "n_refused":
                cells.append(f"{int(v):>9d}")
            elif key == "fraction_within_interval":
                cells.append(f"{v * 100:>8.1f}%")
            else:
                cells.append(f"{v:>9.4f}")
        rows.append(f"{name:<{name_w}}  " + "  ".join(cells))
    return "\n".join(rows)
