"""Data loading and input validation for MolActAgent.

This module sits one layer above ``agent.py`` and is the entry point used
by the CLI and the high-level ``run_classification`` pipeline.

Responsibilities
----------------
1. Load tabular data from a variety of formats (CSV, TSV, Excel, Parquet,
   JSON) using a uniform interface.
2. Load optional artefacts (feature descriptions, background notes) from
   JSON / YAML / plain-text files.
3. Carefully validate every input the user supplies and return a
   structured :class:`ValidationResult` that lists *every* problem at
   once instead of failing on the first one. This lets the CLI show all
   issues to the user in a single pass.
4. Map arbitrary user labels (e.g. ``"active" / "inactive"`` or
   ``1 / 0``) onto the internal ``Pos`` / ``Neg`` contract that
   :class:`molactagent.agent.MolActAgent` expects, while remembering the
   original names so reports use the user's domain terminology.

Loading helpers never raise on validation problems. They only raise for
hard IO errors (file missing, unreadable, etc.) — semantic problems are
collected by :func:`validate_inputs`.
"""

from __future__ import annotations

import io as _io
import json
import logging
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── stdin sentinel ───────────────────────────────────────────────────
#
# A reference / test path of ``"-"`` is interpreted as "read from
# stdin". This lets MolActAgent sit in a Unix pipe, e.g.
#
#   cat ref.csv | molactagent run -r - -t test.csv ...
#
# Only ONE of -r / -t may be ``-`` at a time (otherwise both reads
# would consume the same byte stream).

STDIN_SENTINEL = "-"


def _is_stdin(path) -> bool:
    return isinstance(path, str) and path == STDIN_SENTINEL


def _read_stdin_table(stdin_format: str) -> pd.DataFrame:
    """Buffer stdin into memory and parse it as the requested format.

    We read everything up-front (rather than streaming) because the
    pipeline needs random access to the DataFrame for validation +
    feature scaling. CSV / TSV are the only realistic stdin inputs.
    """
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError(
            "Read 0 bytes from stdin. Did you forget to pipe data in? "
            "Example: `cat my.csv | molactagent run -r - -t test.csv ...`."
        )
    fmt = (stdin_format or "csv").lower()
    if fmt == "csv":
        return pd.read_csv(_io.StringIO(raw))
    if fmt == "tsv":
        return pd.read_csv(_io.StringIO(raw), sep="\t")
    raise ValueError(
        f"Unsupported --stdin-format {fmt!r}. Use 'csv' or 'tsv'."
    )


# ── public surface ────────────────────────────────────────────────────

__all__ = [
    "ValidationResult",
    "PreparedData",
    "load_table",
    "load_feature_descriptions",
    "load_background",
    "validate_inputs",
    "prepare_for_agent",
    "canonical_class_order",
    "detect_task",
    "apply_id_column",
]


SUPPORTED_TABLE_SUFFIXES = {
    ".csv", ".tsv", ".txt",
    ".xlsx", ".xls",
    ".parquet", ".pq",
    ".json", ".jsonl",
}


# ── result containers ─────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of :func:`validate_inputs`.

    Attributes
    ----------
    is_valid:
        ``True`` only when ``errors`` is empty. Warnings do not block.
    errors:
        Blocking, user-actionable problems. Each string is a complete
        sentence that names what is wrong AND what to do about it.
    warnings:
        Non-blocking messages. Validation will still pass.
    summary:
        Useful facts gathered during validation (n_samples, classes
        found, mapped Pos/Neg, etc.) that downstream code can reuse.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_valid

    def format_report(self, *, color: bool = False) -> str:
        """Build a human-readable multi-line report."""
        lines: list[str] = []
        ok_tag = "OK" if self.is_valid else "FAILED"
        lines.append(f"Validation: {ok_tag}")
        if self.summary:
            for k, v in self.summary.items():
                lines.append(f"  - {k}: {v}")
        if self.errors:
            lines.append("")
            lines.append(f"Errors ({len(self.errors)}):")
            for i, e in enumerate(self.errors, 1):
                lines.append(f"  {i}. {e}")
        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for i, w in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {w}")
        return "\n".join(lines)


@dataclass
class PreparedData:
    """Numeric arrays + metadata ready to feed into ``MolActAgent``.

    ``task`` is either ``"classification"`` or ``"regression"``,
    auto-detected from the dtype and value distribution of the label
    column. The two tasks share the same downstream pipeline; only
    the type of value sitting at each reference (and therefore the
    type the LLM emits) differs.

    For classification:
      - ``y_ref`` carries the original user class labels (e.g.
        ``"active"`` / ``"inactive"`` / ``"setosa"``); MolActAgent no
        longer remaps them to internal ``"Pos"`` / ``"Neg"`` strings.
      - ``class_labels`` is the canonical ordered list of distinct
        class names — used by reports, summaries, and colour
        assignment so output is deterministic across runs.
      - ``target_range`` is ``None``.

    For regression:
      - ``y_ref`` carries the user's continuous target values as
        ``float64``.
      - ``class_labels`` is an empty list (kept for backward
        compatibility of the dataclass shape).
      - ``target_range`` is ``(min, max)`` of the observed reference
        targets — surfaced to the LLM so it can place its predictions
        on a known scale.
    """

    X_ref: np.ndarray
    y_ref: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    feature_descriptions: dict[str, str] | None
    background: str | None
    class_labels: list[str]
    test_true_labels: np.ndarray | None
    test_index: pd.Index
    task: str = "classification"
    target_range: tuple[float, float] | None = None


# ── file loaders ──────────────────────────────────────────────────────

def _resolve_path(path: str | pathlib.Path, *, must_exist: bool = True) -> pathlib.Path:
    p = pathlib.Path(path).expanduser()
    if must_exist:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"File is empty (0 bytes): {p}")
    return p


def apply_id_column(df: pd.DataFrame, id_column: str | None) -> pd.DataFrame:
    """Move ``id_column`` from the column list into the DataFrame index.

    Returns ``df`` unchanged if ``id_column`` is ``None``. Raises
    :class:`KeyError` if the column is missing and :class:`ValueError`
    if any value is missing or the IDs are not unique.
    """
    if id_column is None:
        return df
    if id_column not in df.columns:
        raise KeyError(
            f"id_column {id_column!r} not found in table; available "
            f"columns: {list(df.columns)}."
        )
    s = df[id_column]
    if s.isna().any():
        raise ValueError(
            f"id_column {id_column!r} contains missing values; every "
            "row needs a unique non-empty identifier."
        )
    if s.duplicated().any():
        dups = s[s.duplicated()].unique().tolist()
        raise ValueError(
            f"id_column {id_column!r} contains duplicate values "
            f"(e.g. {dups[:5]!r}); each row needs a unique identifier."
        )
    out = df.copy()
    out.index = s.astype(str).values
    out.index.name = id_column
    return out.drop(columns=[id_column])


def load_table(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a tabular dataset from common formats.

    Supported extensions: ``.csv``, ``.tsv``, ``.txt`` (auto-detected
    delimiter), ``.xlsx``/``.xls``, ``.parquet``/``.pq``, ``.json``,
    ``.jsonl``. Raises :class:`ValueError` for unknown formats and
    propagates :class:`FileNotFoundError` / :class:`IsADirectoryError`
    for path problems.
    """
    p = _resolve_path(path)
    suf = p.suffix.lower()

    if suf not in SUPPORTED_TABLE_SUFFIXES:
        raise ValueError(
            f"Unsupported table format {suf!r} for file {p.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_TABLE_SUFFIXES))}."
        )

    try:
        if suf == ".csv":
            df = pd.read_csv(p)
        elif suf == ".tsv":
            df = pd.read_csv(p, sep="\t")
        elif suf == ".txt":
            df = pd.read_csv(p, sep=None, engine="python")
        elif suf in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        elif suf in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        elif suf == ".json":
            df = pd.read_json(p)
        elif suf == ".jsonl":
            df = pd.read_json(p, lines=True)
        else:  # pragma: no cover - guarded by suffix check above
            raise ValueError(f"Unsupported suffix {suf!r}")
    except pd.errors.EmptyDataError as e:
        raise ValueError(
            f"Could not read {p.name}: file appears to have no data "
            f"(only blank lines / no parseable header)."
        ) from e
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Could not parse {p.name} as a {suf} file. "
            f"Underlying parser error: {e}"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to load {p}: {type(e).__name__}: {e}"
        ) from e

    if df.shape[0] == 0 and df.shape[1] == 0:
        raise ValueError(
            f"{p.name} loaded as a completely empty table "
            "(no columns, no rows)."
        )

    logger.info(
        f"Loaded {p.name}: {df.shape[0]} rows × {df.shape[1]} columns"
    )
    return df


def load_feature_descriptions(
    path: str | pathlib.Path | None,
) -> dict[str, str] | None:
    """Load a feature-description mapping from JSON or YAML.

    The file must deserialize to a mapping of ``{column_name: description}``.
    Returns ``None`` when ``path`` is ``None`` (the caller treats this as
    "no descriptions provided"). Raises :class:`ValueError` on malformed
    files or wrong types.
    """
    if path is None:
        return None
    p = _resolve_path(path)
    suf = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suf == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{p.name} is not valid JSON: {e.msg} "
                f"(line {e.lineno}, column {e.colno})."
            ) from e
    elif suf in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - optional dep
            raise ValueError(
                "PyYAML is required to load .yaml/.yml feature "
                "descriptions; install it with `pip install pyyaml`."
            ) from e
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(
                f"{p.name} is not valid YAML: {e}"
            ) from e
    else:
        raise ValueError(
            f"Feature-description file must be .json, .yaml, or .yml; "
            f"got {suf!r} for {p.name}."
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"Feature descriptions in {p.name} must be a mapping/dict "
            f"of column_name -> description string; got "
            f"{type(data).__name__}."
        )

    cleaned: dict[str, str] = {}
    bad_keys: list[str] = []
    for k, v in data.items():
        if not isinstance(k, str):
            bad_keys.append(repr(k))
            continue
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, str):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    if bad_keys:
        raise ValueError(
            f"Feature-description keys must be strings; offending "
            f"keys in {p.name}: {', '.join(bad_keys)}."
        )

    return cleaned


def load_background(
    background: str | pathlib.Path | None,
) -> str | None:
    """Resolve a background string.

    The argument can be:
      - ``None`` -> returns ``None``
      - a string starting with ``"@"`` followed by a path -> reads the
        file and returns its contents
      - a path-like to an existing file -> returns its contents
      - any other string -> returned verbatim
    """
    if background is None:
        return None
    if isinstance(background, str) and background.startswith("@"):
        p = _resolve_path(background[1:])
        return p.read_text(encoding="utf-8").strip()
    if isinstance(background, (str, pathlib.Path)):
        p = pathlib.Path(str(background)).expanduser()
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8").strip()
    return str(background).strip() or None


# ── label normalisation ───────────────────────────────────────────────

def _normalise_label_value(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(float(v)):
            return ""
        if float(v).is_integer():
            return str(int(v))
        return str(float(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v).strip()


def _stringify_label_series(s: pd.Series) -> pd.Series:
    return s.map(_normalise_label_value).astype(str)


def canonical_class_order(classes) -> list[str]:
    """Return a stable, lexicographically-sorted list of class names.

    The order is used by every downstream component that needs a
    deterministic class ordering — for example colour assignment in
    reports and the column order in summary tables. Class names are
    treated as opaque strings; MolActAgent does not impose any
    ``"positive"`` vs ``"negative"`` semantics in the multi-class
    design.
    """
    return sorted({_normalise_label_value(c) for c in classes if c is not None})


def detect_task(label_series: pd.Series) -> str:
    """Auto-detect whether the prediction task is ``"classification"``
    or ``"regression"`` from the label column.

    Heuristic:

    - Non-numeric dtype (string / object / categorical) →
      ``"classification"``.
    - Boolean dtype → ``"classification"``.
    - Numeric dtype with ≤10 distinct integer values →
      ``"classification"`` (covers binary 0/1, multi-class 1..K, and
      ordinal scores like 1-5 or 1-10 — all of which the LLM treats
      as discrete categories).
    - Numeric dtype with non-integer values OR more than 10 distinct
      values → ``"regression"``.

    Empty / all-NaN inputs default to ``"classification"`` so the
    main validator still surfaces the underlying empty-label problem.
    Callers who want to override the heuristic can pass ``task=``
    explicitly at the pipeline level (Phase 5).
    """
    s = label_series.dropna()
    if len(s) == 0:
        return "classification"
    if pd.api.types.is_bool_dtype(s):
        return "classification"
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"
    n_distinct = int(s.nunique())
    try:
        all_integer = bool(np.all(s.values == s.values.astype(int)))
    except (TypeError, ValueError):
        all_integer = False
    if all_integer and n_distinct <= 10:
        return "classification"
    return "regression"


# ── core validation ───────────────────────────────────────────────────

def _is_numeric_dtype(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return True
    return False


def _coerce_numeric(series: pd.Series) -> tuple[pd.Series, bool]:
    """Try to make a column numeric. Returns ``(coerced_series, ok)``.

    If even after :func:`pd.to_numeric` more than 50% of non-null values
    remain unparseable, treat the coercion as having failed.
    """
    if _is_numeric_dtype(series):
        return series.astype(float), True
    coerced = pd.to_numeric(series, errors="coerce")
    n_total = series.notna().sum()
    n_kept = coerced.notna().sum()
    if n_total > 0 and n_kept / n_total < 0.5:
        return coerced, False
    return coerced, True


def validate_inputs(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    *,
    k: int = 3,
    feature_descriptions: dict[str, str] | None = None,
    background: str | None = None,
    positive_class: str | None = None,
    id_column: str | None = None,
) -> ValidationResult:
    """Validate reference + test data and ancillary inputs.

    Collects *all* problems before returning, so the user sees a complete
    list. Never raises on data semantics — only ``TypeError`` /
    ``ValueError`` for blatant API misuse (e.g. wrong argument types).

    When ``id_column`` is provided, MolActAgent enforces:

    * the column exists in BOTH ``reference_df`` and ``test_df``;
    * every value is non-empty and non-NaN;
    * IDs are unique within each dataset; AND
    * IDs do not overlap between the reference and test sets (so every
      report file in the output directory has a unique name and every
      sample can be addressed unambiguously).
    """
    errors: list[str] = []
    warnings_: list[str] = []
    summary: dict[str, Any] = {}

    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError(
            "reference_df must be a pandas.DataFrame; got "
            f"{type(reference_df).__name__}."
        )
    if not isinstance(test_df, pd.DataFrame):
        raise TypeError(
            "test_df must be a pandas.DataFrame; got "
            f"{type(test_df).__name__}."
        )
    if not isinstance(label_column, str) or not label_column:
        raise TypeError(
            "label_column must be a non-empty string."
        )

    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        errors.append(
            f"Parameter k must be an integer; got {type(k).__name__} "
            f"(value={k!r}). Pass an int such as k=3."
        )
    elif int(k) < 1:
        errors.append(
            f"Parameter k must be >= 1 (you gave k={int(k)}). k is the "
            "max number of neighbors per class drawn from the reference "
            "set; the agent needs at least one neighbor per class."
        )

    # ── reference dataset structural checks ──────────────────────
    if reference_df.shape[0] == 0:
        errors.append(
            "Reference dataset has 0 rows. Provide a CSV / Excel / "
            "Parquet file with at least 2 labeled samples (one per class)."
        )
    if reference_df.shape[1] == 0:
        errors.append(
            "Reference dataset has 0 columns. The file must contain at "
            "least the label column plus one feature column."
        )

    if label_column not in reference_df.columns:
        errors.append(
            f"Label column {label_column!r} is not in the reference "
            f"dataset. Available columns: "
            f"{list(reference_df.columns)[:20]}"
            + (" ..." if len(reference_df.columns) > 20 else "")
            + "."
        )

    # ── test dataset structural checks ───────────────────────────
    if test_df.shape[0] == 0:
        errors.append(
            "Test dataset has 0 rows. Provide at least one sample to "
            "classify."
        )
    if test_df.shape[1] == 0:
        errors.append(
            "Test dataset has 0 columns. Make sure the file contains "
            "the same feature columns as the reference dataset."
        )

    # If structural problems are already fatal there is little point
    # checking semantic content, but we still want the per-class /
    # feature-mismatch checks below when the basic shapes are present.
    if errors and (
        label_column not in reference_df.columns
        or reference_df.shape[1] == 0
        or reference_df.shape[0] == 0
    ):
        return ValidationResult(False, errors, warnings_, summary)

    # ── label semantics in the reference set ─────────────────────
    label_series = reference_df[label_column]
    n_label_nan = int(label_series.isna().sum())
    if n_label_nan == len(label_series):
        errors.append(
            f"Every value in the label column {label_column!r} is "
            "missing (NaN). The reference set must have a label/value "
            "for every row."
        )
        # Cannot continue meaningful semantic checks without any data.
        summary["task"] = "classification"
    else:
        if n_label_nan > 0:
            errors.append(
                f"Label column {label_column!r} contains {n_label_nan} "
                "missing value(s) in the reference dataset. Drop or "
                "fill them before running MolActAgent."
            )

        # Auto-detect classification vs regression from the column.
        task = detect_task(label_series)
        summary["task"] = task

        if task == "regression":
            # Regression: continuous numeric target. Validate range
            # and distinct-value count instead of per-class structure.
            finite = label_series.dropna().astype(float)
            n_distinct = int(finite.nunique())
            target_min, target_max = float(finite.min()), float(finite.max())
            summary["n_distinct_targets"] = n_distinct
            summary["target_range"] = (target_min, target_max)
            summary["target_mean"] = float(finite.mean())

            if n_distinct < 3:
                errors.append(
                    f"Regression target {label_column!r} has only "
                    f"{n_distinct} distinct value(s); regression needs "
                    "at least 3. Consider treating this as a "
                    "classification task instead."
                )
            if target_max - target_min <= 0:
                errors.append(
                    f"Regression target {label_column!r} has a "
                    f"degenerate range (min={target_min}, "
                    f"max={target_max}); cannot regress on a constant."
                )
            if reference_df.shape[0] < 2:
                errors.append(
                    f"Reference dataset has {reference_df.shape[0]} "
                    "sample(s); MolActAgent requires a minimum of 2 "
                    "reference samples for regression."
                )
            # ``class_labels`` is a no-op for regression but kept in
            # the summary for downstream code that expects the key.
            summary["class_labels"] = []
        else:
            norm_labels = _stringify_label_series(label_series.dropna())
            distinct = sorted(norm_labels.unique().tolist())
            summary["reference_classes"] = distinct
            summary["reference_class_counts"] = {
                c: int((norm_labels == c).sum()) for c in distinct
            }

            if len(distinct) == 0:
                if not any("missing (NaN)" in e for e in errors):
                    errors.append(
                        f"Label column {label_column!r} has no usable "
                        "values."
                    )
            elif len(distinct) == 1:
                errors.append(
                    f"Reference dataset only contains one class: "
                    f"{distinct[0]!r}. MolActAgent needs at least 2 "
                    "distinct classes to perform contrastive "
                    "comparisons."
                )
            else:
                per_class = summary["reference_class_counts"]
                too_small = [c for c, n in per_class.items() if n < 1]
                if too_small:
                    errors.append(
                        f"Class(es) {too_small} appear in the reference "
                        "dataset with 0 samples after stripping NaN "
                        "labels. Each class needs at least one sample."
                    )
                if reference_df.shape[0] < 2:
                    errors.append(
                        f"Reference dataset has "
                        f"{reference_df.shape[0]} sample(s); "
                        "MolActAgent requires a minimum of 2 reference "
                        "samples (at least one per class)."
                    )
                if len(distinct) > 10:
                    warnings_.append(
                        f"Reference dataset has {len(distinct)} "
                        "distinct classes. Multi-class MolActAgent "
                        "works best with up to ~5 classes; with this "
                        "many candidates the LLM must weigh more "
                        "references per call and reasoning stability "
                        "may suffer. Consider grouping classes if "
                        "appropriate."
                    )

            # Capture the canonical class order for downstream code
            # (reports, summaries, colour assignment).
            if len(distinct) >= 2 and not any(
                "Reference dataset" in e for e in errors
            ):
                summary["class_labels"] = canonical_class_order(distinct)

    # ``positive_class`` is no longer meaningful under the multi-class
    # pipeline. Surface a warning when callers still pass one (so they
    # can update their scripts) but do not fail; the value is ignored.
    if positive_class is not None:
        warnings_.append(
            "The `positive_class` argument is deprecated under the "
            "multi-class pipeline and is now ignored. Class labels are "
            "kept as-is throughout the system."
        )

    # ── k against per-class size (classification only) ───────────
    if (
        isinstance(k, (int, np.integer))
        and not isinstance(k, bool)
        and int(k) >= 1
        and summary.get("task") == "classification"
    ):
        per_class = summary.get("reference_class_counts", {}) or {}
        for cls, n in per_class.items():
            if n < int(k):
                warnings_.append(
                    f"Class {cls!r} has only {n} sample(s) but "
                    f"k={int(k)}; the agent will use up to {n} "
                    "neighbor(s) for that class."
                )

    # ── feature columns ──────────────────────────────────────────
    excluded_from_features = {label_column}
    if id_column:
        excluded_from_features.add(id_column)
    ref_feature_cols = [
        c for c in reference_df.columns if c not in excluded_from_features
    ]
    if len(ref_feature_cols) == 0:
        errors.append(
            "Reference dataset has no feature columns (only the label "
            "column). Add at least one feature column (e.g. measurements "
            "from the behavioral records)."
        )
    summary["n_features"] = len(ref_feature_cols)
    summary["feature_columns"] = ref_feature_cols

    # Check that the test dataset contains every feature column
    test_cols_set = set(test_df.columns)
    missing_in_test = [c for c in ref_feature_cols if c not in test_cols_set]
    if missing_in_test:
        preview = missing_in_test[:8]
        more = "" if len(missing_in_test) <= 8 else f" (and {len(missing_in_test) - 8} more)"
        errors.append(
            f"Test dataset is missing {len(missing_in_test)} feature "
            f"column(s) that exist in the reference dataset: "
            f"{preview}{more}. Make sure both files share the same "
            "feature columns."
        )

    extra_in_test = [
        c for c in test_df.columns
        if c not in ref_feature_cols and c not in excluded_from_features
    ]
    if extra_in_test:
        preview = extra_in_test[:8]
        more = "" if len(extra_in_test) <= 8 else f" (and {len(extra_in_test) - 8} more)"
        warnings_.append(
            f"Test dataset has {len(extra_in_test)} extra column(s) not "
            f"present in the reference dataset: {preview}{more}. They "
            "will be ignored."
        )

    # Numeric coercion check on feature columns
    bad_numeric: list[str] = []
    all_nan_cols: list[str] = []
    constant_cols: list[str] = []
    for c in ref_feature_cols:
        s = reference_df[c]
        if s.isna().all():
            all_nan_cols.append(c)
            continue
        coerced, ok = _coerce_numeric(s)
        if not ok:
            bad_numeric.append(c)
            continue
        non_null = coerced.dropna()
        if len(non_null) > 1 and float(non_null.var()) == 0.0:
            constant_cols.append(c)
    if bad_numeric:
        preview = bad_numeric[:6]
        more = "" if len(bad_numeric) <= 6 else f" (and {len(bad_numeric) - 6} more)"
        errors.append(
            f"Feature column(s) {preview}{more} contain mostly "
            "non-numeric values and could not be converted to numbers. "
            "MolActAgent requires numeric features. Convert them first "
            "(e.g. one-hot encode categorical columns) or drop them."
        )
    if all_nan_cols:
        preview = all_nan_cols[:6]
        more = "" if len(all_nan_cols) <= 6 else f" (and {len(all_nan_cols) - 6} more)"
        warnings_.append(
            f"Feature column(s) {preview}{more} are entirely missing "
            "(all NaN) in the reference dataset and will be dropped "
            "automatically by the variance filter."
        )
    if constant_cols:
        preview = constant_cols[:6]
        more = "" if len(constant_cols) <= 6 else f" (and {len(constant_cols) - 6} more)"
        warnings_.append(
            f"Feature column(s) {preview}{more} have zero variance in "
            "the reference dataset and will not contribute to the "
            "comparison."
        )

    # ── feature_descriptions completeness ────────────────────────
    if feature_descriptions is not None:
        if not isinstance(feature_descriptions, dict):
            raise TypeError(
                "feature_descriptions must be a dict mapping column "
                "names to description strings, or None."
            )
        required_keys = set(ref_feature_cols) | {label_column}
        if id_column:
            # The id_column is metadata, not a feature, so its description
            # is neither required nor flagged as extra.
            pass
        provided_keys = set(feature_descriptions.keys())
        if id_column:
            provided_keys = provided_keys - {id_column}
        missing = sorted(required_keys - provided_keys)
        if missing:
            preview = missing[:8]
            more = "" if len(missing) <= 8 else f" (and {len(missing) - 8} more)"
            errors.append(
                f"Feature descriptions are missing entries for "
                f"{len(missing)} column(s): {preview}{more}. When "
                "feature descriptions are provided they must cover "
                "EVERY column in the dataset, including the label "
                "column."
            )
        extra = sorted(provided_keys - required_keys)
        if extra:
            preview = extra[:6]
            more = "" if len(extra) <= 6 else f" (and {len(extra) - 6} more)"
            warnings_.append(
                f"Feature descriptions contain {len(extra)} key(s) not "
                f"present in the dataset: {preview}{more}. They will "
                "be ignored."
            )
        non_str = [
            k for k, v in feature_descriptions.items()
            if v is not None and not isinstance(v, str)
        ]
        if non_str:
            errors.append(
                f"Feature description values must be strings; the "
                f"following keys have non-string values: {non_str[:6]}."
            )

    # ── background sanity ─────────────────────────────────────────
    if background is not None:
        if not isinstance(background, str):
            raise TypeError(
                "background must be a string (or None). Long backgrounds "
                "may be supplied via @path/to/file.txt syntax."
            )
        if len(background) > 8000:
            warnings_.append(
                f"Background context is very long ({len(background)} "
                "characters). It will be sent to the LLM on every call; "
                "consider summarizing it to a few paragraphs to save "
                "tokens."
            )

    summary["n_reference_samples"] = int(reference_df.shape[0])
    summary["n_test_samples"] = int(test_df.shape[0])
    summary["k"] = int(k) if isinstance(k, (int, np.integer)) and not isinstance(k, bool) else k

    # ── id_column checks (mandatory in CLI; optional in the API) ──
    if id_column is not None:
        summary["id_column"] = id_column
        ref_has = id_column in reference_df.columns
        test_has = id_column in test_df.columns
        if not ref_has:
            errors.append(
                f"id_column {id_column!r} is missing from the reference "
                f"dataset (columns are {list(reference_df.columns)[:20]}). "
                "MolActAgent uses the id_column to label every output "
                "file uniquely."
            )
        if not test_has:
            errors.append(
                f"id_column {id_column!r} is missing from the test "
                f"dataset (columns are {list(test_df.columns)[:20]}). "
                "MolActAgent uses the id_column to label every output "
                "file uniquely."
            )
        if ref_has and test_has:
            ref_ids = reference_df[id_column]
            test_ids = test_df[id_column]
            ref_norm = ref_ids.map(_normalise_label_value).astype(str)
            test_norm = test_ids.map(_normalise_label_value).astype(str)

            ref_empty = (ref_norm == "") | ref_ids.isna()
            test_empty = (test_norm == "") | test_ids.isna()
            if ref_empty.any():
                errors.append(
                    f"Reference id_column {id_column!r} has "
                    f"{int(ref_empty.sum())} empty / missing value(s); "
                    "every reference row needs a unique non-empty ID."
                )
            if test_empty.any():
                errors.append(
                    f"Test id_column {id_column!r} has "
                    f"{int(test_empty.sum())} empty / missing value(s); "
                    "every test row needs a unique non-empty ID."
                )

            ref_dup = ref_norm[~ref_empty][ref_norm[~ref_empty].duplicated()].unique().tolist()
            test_dup = test_norm[~test_empty][test_norm[~test_empty].duplicated()].unique().tolist()
            if ref_dup:
                errors.append(
                    f"Reference id_column {id_column!r} contains "
                    f"duplicate value(s) (e.g. {ref_dup[:5]!r}); each "
                    "reference row must have a unique ID."
                )
            if test_dup:
                errors.append(
                    f"Test id_column {id_column!r} contains duplicate "
                    f"value(s) (e.g. {test_dup[:5]!r}); each test row "
                    "must have a unique ID."
                )

            ref_clean = set(ref_norm[~ref_empty].tolist()) - set(ref_dup)
            test_clean = set(test_norm[~test_empty].tolist()) - set(test_dup)
            overlap = sorted(ref_clean & test_clean)
            if overlap:
                errors.append(
                    f"id_column {id_column!r} value(s) {overlap[:5]!r}"
                    + (f" (and {len(overlap) - 5} more)" if len(overlap) > 5 else "")
                    + " appear in BOTH the reference and test datasets. "
                    "IDs must be globally unique so each per-sample "
                    "report has an unambiguous name. Rename the "
                    "duplicates."
                )

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings_, summary)


# ── post-validation preparation ───────────────────────────────────────

def prepare_for_agent(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    *,
    feature_descriptions: dict[str, str] | None = None,
    background: str | None = None,
    positive_class: str | None = None,
    id_column: str | None = None,
) -> PreparedData:
    """Convert validated DataFrames into numpy arrays for ``MolActAgent``.

    This routine assumes :func:`validate_inputs` has already passed.
    It does the minimum amount of cleaning (drops fully-NaN feature
    columns, coerces to ``float64``, normalises label strings) and
    keeps the user-provided class labels intact — the multi-class
    agent uses them directly without any internal remapping.

    When ``id_column`` is supplied, that column is moved to the
    DataFrame index so its values become the canonical ``test_index``
    used to name per-sample report files.

    ``positive_class`` is accepted for backward compatibility but is
    ignored; the multi-class pipeline treats every class equally.
    """
    if id_column is not None:
        if id_column in reference_df.columns:
            reference_df = apply_id_column(reference_df, id_column)
        if id_column in test_df.columns:
            test_df = apply_id_column(test_df, id_column)

    excluded = {label_column}
    if id_column:
        excluded.add(id_column)
    feature_cols = [c for c in reference_df.columns if c not in excluded]

    # Drop columns that are all-NaN in the reference set (validated as
    # warnings only; we silently honor them here).
    keep_cols = [c for c in feature_cols if not reference_df[c].isna().all()]

    # Coerce to numeric, keeping NaN for missing values (filled later by
    # DataPreprocessor with per-feature medians).
    ref_num = reference_df[keep_cols].apply(pd.to_numeric, errors="coerce")
    X_ref = ref_num.to_numpy(dtype=float)

    raw_label = reference_df[label_column]
    task = detect_task(raw_label)

    if task == "regression":
        y_ref = raw_label.astype(float).to_numpy()
        class_labels: list[str] = []
        finite = y_ref[~np.isnan(y_ref)] if y_ref.dtype.kind == "f" else y_ref
        if len(finite) > 0:
            target_range: tuple[float, float] | None = (
                float(finite.min()), float(finite.max()),
            )
        else:
            target_range = None
    else:
        label_str = _stringify_label_series(raw_label)
        class_labels = canonical_class_order(label_str.unique().tolist())
        y_ref = label_str.to_numpy()
        target_range = None

    # Reorder/select test columns and coerce
    test_aligned = test_df.reindex(columns=keep_cols)
    test_num = test_aligned.apply(pd.to_numeric, errors="coerce")
    X_test = test_num.to_numpy(dtype=float)

    test_true: np.ndarray | None = None
    if label_column in test_df.columns:
        # Capture original test labels/values (if any) for downstream
        # display; they are not used during prediction.
        raw = test_df[label_column]
        if raw.notna().any():
            if task == "regression":
                test_true = raw.astype(float).to_numpy()
            else:
                test_true = _stringify_label_series(raw).to_numpy()

    return PreparedData(
        X_ref=X_ref,
        y_ref=y_ref,
        X_test=X_test,
        feature_names=list(keep_cols),
        feature_descriptions=feature_descriptions,
        background=background,
        class_labels=class_labels,
        test_true_labels=test_true,
        test_index=test_df.index,
        task=task,
        target_range=target_range,
    )
