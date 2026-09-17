"""Tabular preprocessing helpers.

The package's existing :mod:`molactagent.io` already does the heavy
lifting (loading, validation, preparation) for arbitrary tabular
datasets. This module wraps those primitives in a single
:func:`load_dataset` call so the most common path — "I have a
reference CSV and a test CSV, give me arrays I can fit on" —
becomes a one-liner.

This is intentionally a thin Scanpy-style facade. For complex flows
(custom validation hooks, multi-format input, JSONL streaming)
users should keep calling the low-level :mod:`molactagent.io`
helpers directly. We do not hide them.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import io as _io


@dataclass
class TabularBundle:
    """One-call output of :func:`load_dataset`.

    Holds the validation result *plus* the prepared numpy arrays
    so a single ``ma.pp.load_dataset(...)`` call is enough to feed
    either the high-level pipeline or the new estimator wrappers.

    The validation result is exposed because surfaces like
    ``run_classification`` already accept a separate ``validate``
    pre-pass; this lets users opt into a "see all warnings up
    front" flow without having to call :func:`validate_inputs`
    themselves.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray | None
    feature_names: list[str]
    feature_descriptions: dict[str, str] | None
    background: str | None
    class_labels: list[str]
    target_range: tuple[float, float] | None
    test_index: pd.Index
    task: str
    validation: _io.ValidationResult


def load_dataset(
    reference: str | pathlib.Path | pd.DataFrame,
    test: str | pathlib.Path | pd.DataFrame,
    label_column: str,
    *,
    id_column: str | None = None,
    feature_descriptions: str | pathlib.Path | dict[str, str] | None = None,
    background: str | None = None,
    raise_on_validation_error: bool = True,
) -> TabularBundle:
    """Load + validate + prepare a (reference, test) CSV pair in one call.

    Parameters
    ----------
    reference, test :
        Either DataFrames or paths to a CSV / TSV / Excel / Parquet
        / JSON file. Strings prefixed with ``"@"`` are treated as
        file paths for ``background``.
    label_column :
        Name of the label column. Forwarded to the engine's
        validators.
    id_column :
        Optional row-identifier column.
    feature_descriptions :
        Either a dict ``{column: description}`` or a path to a
        JSON / YAML file. Optional but recommended.
    background :
        Free-text domain context. ``"@notes.txt"`` reads from a
        file.
    raise_on_validation_error :
        Default ``True`` — if validation fails, raise
        :class:`ValueError` with the full report. Set ``False`` to
        get a :class:`TabularBundle` whose ``validation`` carries
        the error list (useful for notebook diagnostics).

    Returns
    -------
    :class:`TabularBundle`
    """
    ref_df = _to_dataframe(reference)
    test_df = _to_dataframe(test)

    bg_text = (
        background if isinstance(background, str)
        and not background.startswith("@")
        else _io.load_background(background)
    )
    fdescs = _coerce_descriptions(feature_descriptions)

    validation = _io.validate_inputs(
        ref_df, test_df, label_column,
        feature_descriptions=fdescs,
        background=bg_text,
        id_column=id_column,
    )

    if not validation.is_valid:
        msg = "Input validation failed:\n" + validation.format_report()
        if raise_on_validation_error:
            raise ValueError(msg)
        # Even on failure we still try to return a bundle with
        # whatever the preparator can salvage; this matches the
        # high-level pipeline's "return early" semantics.
        return TabularBundle(
            X_train=np.empty((0, 0)),
            y_train=np.empty((0,)),
            X_test=np.empty((0, 0)),
            y_test=None,
            feature_names=[],
            feature_descriptions=fdescs,
            background=bg_text,
            class_labels=[],
            target_range=None,
            test_index=pd.Index([]),
            task=validation.summary.get("task", "classification"),
            validation=validation,
        )

    prepared = _io.prepare_for_agent(
        ref_df, test_df, label_column,
        feature_descriptions=fdescs,
        background=bg_text,
        id_column=id_column,
    )

    return TabularBundle(
        X_train=prepared.X_ref,
        y_train=prepared.y_ref,
        X_test=prepared.X_test,
        y_test=prepared.test_true_labels,
        feature_names=list(prepared.feature_names),
        feature_descriptions=prepared.feature_descriptions,
        background=prepared.background,
        class_labels=list(prepared.class_labels),
        target_range=prepared.target_range,
        test_index=prepared.test_index,
        task=prepared.task,
        validation=validation,
    )


# ── helpers ────────────────────────────────────────────────────────


def _to_dataframe(
    data: str | pathlib.Path | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return _io.load_table(data)


def _coerce_descriptions(
    descriptions: str | pathlib.Path | dict[str, str] | None,
) -> dict[str, str] | None:
    if descriptions is None:
        return None
    if isinstance(descriptions, dict):
        return {str(k): "" if v is None else str(v) for k, v in descriptions.items()}
    return _io.load_feature_descriptions(descriptions)
