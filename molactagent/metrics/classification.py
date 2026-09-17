"""Classification metrics — binary and multi-class.

Direct port of :func:`compute_full_metrics` from
``scripts/run_loocv_metrics.py`` with two structural changes:

1. The function is split in two: :func:`binary_classification_metrics`
   keeps the original binary-specialised behaviour (positive/negative
   classes, ROC AUC, PR AUC, full confusion-matrix breakdown);
   :func:`classification_metrics` is a polymorphic entry point that
   auto-detects binary vs multi-class and dispatches accordingly.
2. The output is a typed :class:`ClassificationReport` dataclass
   with ``to_table`` / ``to_dict`` / ``to_dataframe`` helpers,
   instead of a bare dict. The dataclass is designed so existing
   dict-keyed code (e.g. the LOOCV scripts) keeps working —
   ``ClassificationReport.to_dict()`` returns a dict whose keys
   match the legacy schema bit-for-bit.

The design follows the §29.2 instruction in ``docs/design.md``:
"new numerical features belong in the witness layer". These
metrics are pure post-hoc computations that do not influence
prediction; they belong here, not in :mod:`molactagent.agent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ── lazy sklearn imports ───────────────────────────────────────────
#
# Importing sklearn at module load time would push it into the
# critical path of ``import molactagent``. Defer.

def _sklearn_metrics():
    """Return the sklearn metric functions we use, imported lazily."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    return {
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "matthews_corrcoef": matthews_corrcoef,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


# ── report dataclass ────────────────────────────────────────────────


@dataclass
class ClassificationReport:
    """Typed wrapper around a metric dict.

    Field meanings match the legacy ``compute_full_metrics`` schema
    so existing JSON files and downstream consumers keep working.
    Multi-class reports leave binary-only fields as ``None``.

    Notes
    -----
    - For binary tasks, the ``positive_class`` and
      ``negative_class`` attributes record which user label was
      treated as 1 / 0; AUCs and per-class metrics use the same
      convention.
    - For multi-class tasks, ``per_class_f1`` / ``per_class_precision``
      / ``per_class_recall`` carry the per-class numbers and the
      macro/weighted averages are stored separately.
    - ``unknown_rate`` is the fraction of test rows where the
      prediction was outside ``{positive_class, negative_class}``
      (binary) or outside the set of fit-time classes
      (multi-class). For MolActAgent, this is non-zero whenever the
      verdict is ``"Inconclusive"``.
    """

    n_samples: int
    n_classes: int
    accuracy: float
    balanced_accuracy: float
    mcc: float
    macro_f1: float
    unknown_rate: float
    confusion_matrix: list[list[int]]  # nested list for JSON-friendliness
    class_labels: list[str]

    # Binary-only.
    positive_class: str | None = None
    negative_class: str | None = None
    precision_pos: float | None = None
    recall_pos: float | None = None
    f1_pos: float | None = None
    precision_neg: float | None = None
    recall_neg: float | None = None
    f1_neg: float | None = None
    specificity_neg: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    tp: int | None = None
    tn: int | None = None
    fp: int | None = None
    fn: int | None = None
    unknown: int | None = None

    # Multi-class extras.
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    weighted_f1: float | None = None

    # ── interop ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a flat dict.

        For binary reports the keys match the legacy schema
        (``compute_full_metrics`` in the original LOOCV script) so
        consumers such as ``scripts/aggregate_loocv_results.py``
        keep parsing existing files unchanged.
        """
        out = {
            "n_samples": int(self.n_samples),
            "n_classes": int(self.n_classes),
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "mcc": float(self.mcc),
            "macro_f1": float(self.macro_f1),
            "unknown_rate": float(self.unknown_rate),
            "confusion_matrix": [list(map(int, row)) for row in self.confusion_matrix],
            "class_labels": list(self.class_labels),
        }
        # Binary fields — included only when populated to match the
        # legacy "either binary or multi-class, not both" schema.
        if self.positive_class is not None:
            out.update({
                "positive_class": self.positive_class,
                "negative_class": self.negative_class,
                "precision_pos": _maybe_float(self.precision_pos),
                "recall_pos": _maybe_float(self.recall_pos),
                "f1_pos": _maybe_float(self.f1_pos),
                "precision_neg": _maybe_float(self.precision_neg),
                "recall_neg": _maybe_float(self.recall_neg),
                "f1_neg": _maybe_float(self.f1_neg),
                "specificity_neg": _maybe_float(self.specificity_neg),
                "roc_auc": _maybe_float(self.roc_auc),
                "pr_auc": _maybe_float(self.pr_auc),
                "tp": _maybe_int(self.tp),
                "tn": _maybe_int(self.tn),
                "fp": _maybe_int(self.fp),
                "fn": _maybe_int(self.fn),
                "unknown": _maybe_int(self.unknown),
            })
        if self.per_class_f1:
            out["per_class_f1"] = dict(self.per_class_f1)
            out["per_class_precision"] = dict(self.per_class_precision)
            out["per_class_recall"] = dict(self.per_class_recall)
            out["weighted_f1"] = _maybe_float(self.weighted_f1)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Return a long-form DataFrame with one row per metric."""
        d = self.to_dict()
        rows = []
        for k, v in d.items():
            if isinstance(v, (list, dict)):
                # Skip nested fields; consumers can extract them
                # explicitly.
                continue
            rows.append({"metric": k, "value": v})
        return pd.DataFrame(rows)

    def to_table(self) -> str:
        """Render a single-method, fixed-width text table."""
        lines = []
        title = (
            f"Classification metrics (n={self.n_samples}, "
            f"classes={self.n_classes}"
            + (
                f", positive={self.positive_class!r}"
                if self.positive_class is not None else ""
            )
            + ")"
        )
        lines.append(title)
        lines.append("-" * len(title))

        def _row(label: str, value: float | None, fmt: str = ".4f"):
            if value is None:
                v = "—"
            elif isinstance(value, float):
                v = format(value, fmt)
            else:
                v = str(value)
            lines.append(f"  {label:<22}  {v}")

        _row("Accuracy", self.accuracy)
        _row("Balanced accuracy", self.balanced_accuracy)
        _row("Matthews CC (MCC)", self.mcc)
        _row("Macro F1", self.macro_f1)
        if self.weighted_f1 is not None:
            _row("Weighted F1", self.weighted_f1)
        _row("Unknown rate", self.unknown_rate)

        if self.positive_class is not None:
            lines.append("")
            lines.append(
                f"  Per-class (positive={self.positive_class!r}, "
                f"negative={self.negative_class!r})"
            )
            _row("F1 (Pos)", self.f1_pos)
            _row("F1 (Neg)", self.f1_neg)
            _row("Precision (Pos)", self.precision_pos)
            _row("Recall (Pos)", self.recall_pos)
            _row("Precision (Neg)", self.precision_neg)
            _row("Recall (Neg)", self.recall_neg)
            _row("Specificity (Neg)", self.specificity_neg)
            _row("ROC AUC", self.roc_auc)
            _row("PR AUC", self.pr_auc)
            lines.append("")
            lines.append(
                f"  Confusion matrix  TP={self.tp} FP={self.fp} "
                f"FN={self.fn} TN={self.tn}  Unk={self.unknown}"
            )
        elif self.per_class_f1:
            lines.append("")
            lines.append("  Per-class F1:")
            for cls, f1 in self.per_class_f1.items():
                lines.append(f"    {cls:<14}  {f1:.4f}")

        return "\n".join(lines)


def _maybe_float(v):
    return None if v is None else float(v)


def _maybe_int(v):
    return None if v is None else int(v)


# ── binary classification (the original LOOCV path) ────────────────


def binary_classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    proba_pos: list[float] | np.ndarray | None = None,
    *,
    positive_class: str = "Pos",
    negative_class: str = "Neg",
) -> ClassificationReport:
    """Compute the full binary metric suite.

    Anything outside ``{positive_class, negative_class}`` (typically
    ``"Inconclusive"`` from MolActAgent or ``"Unknown"`` from a
    legacy run) is treated as a wrong answer for confusion-matrix
    and accuracy purposes; for the ranking metrics (ROC AUC, PR
    AUC) the corresponding score is left untouched (NaN gets
    replaced with 0.5 so AUC computations still complete).

    This matches the policy used by every benchmark in this
    repository — see the comment block in
    ``scripts/run_loocv_metrics.compute_full_metrics``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length; "
            f"got {len(y_true)} vs {len(y_pred)}."
        )
    n = len(y_true)
    skl = _sklearn_metrics()

    y_true_bin = np.array(
        [1 if l == positive_class else 0 for l in y_true], dtype=int,
    )

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

    # Inconclusive predictions get bucketed into "negative" for the
    # binary-encoded sklearn metrics so the labels stay 0/1; we
    # still report the inconclusive count separately above.
    y_pred_bin = np.array(
        [1 if p == positive_class else 0 for p in y_pred], dtype=int,
    )
    bal_acc = skl["balanced_accuracy_score"](y_true_bin, y_pred_bin)
    mcc = skl["matthews_corrcoef"](y_true_bin, y_pred_bin)
    macro_f1 = skl["f1_score"](
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    f1_pos = skl["f1_score"](y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_neg = skl["f1_score"](y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    prec_pos = skl["precision_score"](y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    rec_pos = skl["recall_score"](y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    prec_neg = skl["precision_score"](y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    rec_neg = skl["recall_score"](y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc: float | None = None
    pr_auc: float | None = None
    if proba_pos is not None and len(proba_pos) == n:
        s = np.array(proba_pos, dtype=float)
        s = np.where(np.isnan(s), 0.5, s)
        try:
            roc_auc = float(skl["roc_auc_score"](y_true_bin, s))
        except ValueError:
            roc_auc = None
        try:
            pr_auc = float(skl["average_precision_score"](y_true_bin, s))
        except ValueError:
            pr_auc = None

    return ClassificationReport(
        n_samples=int(n),
        n_classes=2,
        accuracy=float(accuracy),
        balanced_accuracy=float(bal_acc),
        mcc=float(mcc),
        macro_f1=float(macro_f1),
        unknown_rate=n_unk / n if n else 0.0,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        class_labels=[negative_class, positive_class],
        positive_class=positive_class,
        negative_class=negative_class,
        precision_pos=float(prec_pos),
        recall_pos=float(rec_pos),
        f1_pos=float(f1_pos),
        precision_neg=float(prec_neg),
        recall_neg=float(rec_neg),
        f1_neg=float(f1_neg),
        specificity_neg=float(specificity),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        unknown=int(n_unk),
    )


# ── multi-class entry point ────────────────────────────────────────


def classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    *,
    proba: np.ndarray | None = None,
    classes: list[str] | None = None,
    positive_class: str | None = None,
    negative_class: str | None = None,
) -> ClassificationReport:
    """Polymorphic entry point — binary or multi-class.

    When the dataset has exactly two distinct classes (after
    normalising for unknowns), this dispatches to
    :func:`binary_classification_metrics` so users get the full
    binary suite (ROC AUC, PR AUC, per-class precision/recall/F1).

    When the dataset has more than two classes, it computes the
    multi-class equivalents (macro F1, per-class F1, weighted F1,
    multi-class confusion matrix). Multi-class AUC is intentionally
    NOT computed because there is no agreed-upon scalar definition
    that would fit on the legacy schema.

    ``proba``, when given, must be either:

    - 1-D shaped ``(n,)`` — interpreted as the positive-class
      probability for binary tasks.
    - 2-D shaped ``(n, n_classes)`` — interpreted as a full
      probability matrix; the column for ``positive_class`` is
      extracted and used as ``proba_pos`` for the binary branch.

    ``classes`` controls the column order of the multi-class
    confusion matrix; defaults to the sorted union of true and
    predicted labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine the effective class set — y_true is authoritative
    # (predictions can include "Inconclusive" / "Unknown" that we
    # don't want polluting the class list).
    distinct_true = sorted(set(map(str, y_true.tolist())))
    if len(distinct_true) == 2:
        if positive_class is None:
            # Auto-detect: prefer "active" if present (matches the
            # script's convention), otherwise the lexicographically
            # larger label.
            if "active" in distinct_true:
                positive_class = "active"
            else:
                positive_class = distinct_true[-1]
        if negative_class is None:
            negative_class = next(c for c in distinct_true if c != positive_class)

        # Slice positive-class probability if a 2-D matrix was given.
        proba_pos = None
        if proba is not None:
            arr = np.asarray(proba)
            if arr.ndim == 1:
                proba_pos = arr
            elif arr.ndim == 2:
                if classes is None:
                    raise ValueError(
                        "When passing a 2-D `proba` matrix you must "
                        "also pass `classes` so we can find the "
                        "positive-class column."
                    )
                if positive_class not in classes:
                    raise ValueError(
                        f"`positive_class={positive_class!r}` not in "
                        f"`classes={list(classes)!r}`."
                    )
                pos_idx = classes.index(positive_class)
                proba_pos = arr[:, pos_idx]
            else:
                raise ValueError(
                    f"`proba` must be 1-D or 2-D; got shape {arr.shape}."
                )

        return binary_classification_metrics(
            y_true.tolist(),
            y_pred.tolist(),
            proba_pos=proba_pos,
            positive_class=positive_class,
            negative_class=negative_class,
        )

    # Multi-class path.
    return _multiclass_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
    )


def _multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str] | None,
) -> ClassificationReport:
    """Multi-class metrics: macro F1, per-class metrics, multi-class
    confusion matrix, MCC (multi-class generalisation), accuracy,
    balanced accuracy.

    Predictions outside the fit-time class set (typically
    ``"Inconclusive"``) get bucketed as a separate "Unknown" row
    in the confusion matrix and counted in ``unknown_rate``.
    """
    skl = _sklearn_metrics()

    if classes is None:
        # Sort true classes for a stable column order.
        classes = sorted(set(map(str, y_true.tolist())))

    n = len(y_true)
    n_unk = sum(1 for p in y_pred if str(p) not in classes)

    # Build the confusion matrix manually so we can include an
    # "Unknown" pseudo-prediction column without sklearn's
    # ``labels=`` complaining.
    label_to_idx = {c: i for i, c in enumerate(classes)}
    cm_size = len(classes)
    cm = np.zeros((cm_size, cm_size + 1), dtype=int)  # +1 for Unknown col
    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(str(t))
        if ti is None:
            # Truth outside the class set — should never happen,
            # but skip rather than crash if it does.
            continue
        pi = label_to_idx.get(str(p), cm_size)  # last col = Unknown
        cm[ti, pi] += 1

    # Accuracy: rows where t == p.
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0

    # Bucket predictions outside ``classes`` to the LAST class so
    # sklearn metrics that expect the same label set on both sides
    # do not crash. This still penalises the prediction (it is now
    # "the wrong class" rather than "an unknown class") which
    # matches the binary path's behaviour.
    sentinel_class = classes[-1]
    y_pred_buck = np.array([
        str(p) if str(p) in classes else sentinel_class
        for p in y_pred
    ])
    y_true_str = np.array([str(t) for t in y_true])

    bal_acc = skl["balanced_accuracy_score"](y_true_str, y_pred_buck)
    mcc = skl["matthews_corrcoef"](y_true_str, y_pred_buck)
    macro_f1 = skl["f1_score"](
        y_true_str, y_pred_buck, average="macro", labels=classes,
        zero_division=0,
    )
    weighted_f1 = skl["f1_score"](
        y_true_str, y_pred_buck, average="weighted", labels=classes,
        zero_division=0,
    )

    per_class_f1 = {
        cls: float(skl["f1_score"](
            y_true_str, y_pred_buck, labels=[cls], average="macro",
            zero_division=0,
        ))
        for cls in classes
    }
    per_class_precision = {
        cls: float(skl["precision_score"](
            y_true_str, y_pred_buck, labels=[cls], average="macro",
            zero_division=0,
        ))
        for cls in classes
    }
    per_class_recall = {
        cls: float(skl["recall_score"](
            y_true_str, y_pred_buck, labels=[cls], average="macro",
            zero_division=0,
        ))
        for cls in classes
    }

    return ClassificationReport(
        n_samples=int(n),
        n_classes=len(classes),
        accuracy=float(accuracy),
        balanced_accuracy=float(bal_acc),
        mcc=float(mcc),
        macro_f1=float(macro_f1),
        unknown_rate=n_unk / n if n else 0.0,
        confusion_matrix=cm.tolist(),
        class_labels=list(classes) + ["__Unknown__"],
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        weighted_f1=float(weighted_f1),
    )
