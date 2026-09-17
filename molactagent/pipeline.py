"""High-level MolActAgent pipeline.

Glue layer used by both :mod:`molactagent.cli` and notebook users. It
takes paths or DataFrames straight from the user, runs validation, and
either reports problems back or runs the full classify -> explain ->
render-report flow for every test sample.

Public API
----------
:func:`run_classification`
    The blocking, all-in-one entry point. Returns a list of per-sample
    :class:`SampleOutcome` records and writes JSON / Markdown / HTML
    artefacts to an output directory.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from .agent import MolActAgent
from .config import LLMConfig, LLMConfigError, build_llm_client
from .io import (
    PreparedData,
    ValidationResult,
    load_background,
    load_feature_descriptions,
    load_table,
    prepare_for_agent,
    validate_inputs,
)
from .report import (
    render_html_report,
    render_interpretability_html,
    render_interpretability_md,
    render_interpretability_text,
)

logger = logging.getLogger(__name__)


__all__ = [
    "SampleOutcome",
    "PipelineOutput",
    "run_prediction",
    "run_classification",
    "run_regression",
]


@dataclass
class SampleOutcome:
    """One row of results returned to the caller.

    For classification, ``predicted_label`` is the chosen class name
    (or ``"Inconclusive"``) and ``predicted_value`` is ``None``.
    For regression, ``predicted_label`` is the predicted value
    formatted as a string (so summary CSV / JSONL stays one stable
    column) and ``predicted_value`` is the float, while ``interval``
    holds the predicted ``(lo, hi)`` range. ``true_label`` carries
    whichever ground-truth value the test set provided (string class
    name for classification; float for regression, accepted as a
    string for column homogeneity).
    """

    test_index: int
    sample_id: str
    predicted_label: str
    confidence: float
    true_label: str | None = None
    raw_result: dict[str, Any] = field(default_factory=dict)
    explanation_md: str = ""
    explanation_html_path: str | None = None
    explanation_md_path: str | None = None
    raw_json_path: str | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    task: str = "classification"
    predicted_value: float | None = None
    interval: tuple[float, float] | None = None

    def to_summary_row(self) -> dict[str, Any]:
        return {
            "test_index": self.test_index,
            "sample_id": self.sample_id,
            "task": self.task,
            "predicted_label": self.predicted_label,
            "predicted_value": (
                "" if self.predicted_value is None
                else round(float(self.predicted_value), 4)
            ),
            "interval_lo": (
                "" if self.interval is None or self.interval[0] is None
                else round(float(self.interval[0]), 4)
            ),
            "interval_hi": (
                "" if self.interval is None or self.interval[1] is None
                else round(float(self.interval[1]), 4)
            ),
            "confidence": self.confidence,
            "true_label": self.true_label,
            "elapsed_s": round(self.elapsed_s, 2),
            "html_report": self.explanation_html_path or "",
            "error": self.error or "",
        }


@dataclass
class PipelineOutput:
    """Bundle of everything :func:`run_classification` produced."""

    validation: ValidationResult
    outcomes: list[SampleOutcome] = field(default_factory=list)
    output_dir: pathlib.Path | None = None
    summary_csv_path: str | None = None
    summary_json_path: str | None = None
    interpretability_text: str = ""
    interpretability_md_path: str | None = None
    interpretability_html_path: str | None = None
    llm_stats: dict[str, Any] | None = None

    @property
    def n_succeeded(self) -> int:
        return sum(1 for o in self.outcomes if o.error is None)

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outcomes if o.error is not None)


# ── helpers ──────────────────────────────────────────────────────────

def _coerce_to_dataframe(
    data: str | pathlib.Path | pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, pathlib.Path)):
        return load_table(data)
    raise TypeError(
        f"{name} must be a path-like string, pathlib.Path, or "
        f"pandas.DataFrame; got {type(data).__name__}."
    )


def _coerce_descriptions(
    descriptions: str | pathlib.Path | dict[str, str] | None,
) -> dict[str, str] | None:
    if descriptions is None:
        return None
    if isinstance(descriptions, dict):
        return {str(k): ("" if v is None else str(v)) for k, v in descriptions.items()}
    if isinstance(descriptions, (str, pathlib.Path)):
        return load_feature_descriptions(descriptions)
    raise TypeError(
        f"feature_descriptions must be a dict, path, or None; got "
        f"{type(descriptions).__name__}."
    )


def _build_sample_id(idx_label: Any, fallback: int) -> str:
    if idx_label is None:
        return f"sample_{fallback}"
    s = str(idx_label).strip()
    return s if s else f"sample_{fallback}"


def _safe_filename(name: str, fallback: str) -> str:
    bad = '<>:"/\\|?*\x00'
    cleaned = "".join("_" if c in bad else c for c in name)
    cleaned = cleaned.strip()
    return cleaned or fallback


def _preflight_llm(
    llm_client: Any, llm_config: LLMConfig | None,
) -> None:
    """Do a 1-token ping so credential/endpoint problems surface early.

    Translates any network / auth / model failure into
    :class:`LLMConfigError` with concrete next steps; the CLI catches
    that and prints a friendly "edit your llm_config.jsonc" banner.
    Silently no-ops if the client lacks a ``ping`` method (e.g. a
    user-supplied stub or the :class:`FakeLLM` used in tests).
    """
    ping = getattr(llm_client, "ping", None)
    if not callable(ping):
        return
    result = ping(timeout=30.0)
    if result.get("ok"):
        logger.info(
            f"[preflight] LLM reachable "
            f"({result.get('elapsed_s')}s; kind=ok)"
        )
        return

    cfg_path = getattr(llm_config, "source", None)
    kind = result.get("error_kind", "unknown")
    raw = result.get("error", "unknown error")
    model = getattr(llm_config, "model_name", "(unknown model)") or "(unknown model)"
    base_url = getattr(llm_config, "base_url", "(unknown URL)") or "(unknown URL)"

    if kind == "auth":
        hints = [
            "The provider rejected the API key. Double-check the "
            "`api_key` value in your llm_config.jsonc.",
            "Make sure you haven't accidentally left a REPLACE-WITH-… "
            "placeholder from the starter template.",
            f"Verify that the key is valid for {base_url}.",
        ]
        message = "LLM authentication failed (api_key rejected)."
    elif kind == "model":
        hints = [
            f"The endpoint does not know a model named {model!r}.",
            "Open llm_config.jsonc and set `model_name` to a model your "
            "provider actually exposes.",
            f"Example for {base_url}: see the vendor templates inside "
            "the config file.",
        ]
        message = f"LLM model {model!r} not found at {base_url}."
    elif kind == "network":
        hints = [
            f"Could not reach {base_url}. Check your internet "
            "connection and any VPN / firewall rules.",
            "Verify that `base_url` in llm_config.jsonc is spelled "
            "correctly (no trailing slash issues).",
            f"Raw error: {raw}",
        ]
        message = "LLM endpoint is unreachable."
    else:
        hints = [
            f"Raw error: {raw}",
            "If the message mentions `api_key`, update llm_config.jsonc.",
            "If it mentions `base_url` or SSL, check that the endpoint "
            "is correct.",
        ]
        message = "LLM preflight failed."

    raise LLMConfigError(
        message,
        config_path=cfg_path,
        hints=hints,
    )


# ── main entry ───────────────────────────────────────────────────────

_VALID_PER_SAMPLE_FORMATS: tuple[str, ...] = ("html", "md", "json")


def _normalise_per_sample_formats(
    fmts: Any | None,
) -> tuple[str, ...]:
    if fmts is None:
        return _VALID_PER_SAMPLE_FORMATS
    if isinstance(fmts, str):
        items = [s.strip().lower() for s in fmts.split(",") if s.strip()]
    else:
        items = [str(s).strip().lower() for s in fmts if str(s).strip()]
    bad = [s for s in items if s not in _VALID_PER_SAMPLE_FORMATS]
    if bad:
        raise ValueError(
            f"Unknown per_sample_formats value(s): {bad!r}. "
            f"Choose from {list(_VALID_PER_SAMPLE_FORMATS)!r}."
        )
    seen: list[str] = []
    for s in items:
        if s not in seen:
            seen.append(s)
    return tuple(seen) or _VALID_PER_SAMPLE_FORMATS


def run_prediction(
    reference: str | pathlib.Path | pd.DataFrame,
    test: str | pathlib.Path | pd.DataFrame,
    label_column: str,
    *,
    task: str | None = None,
    k: int = 3,
    background: str | None = None,
    feature_descriptions: str | pathlib.Path | dict[str, str] | None = None,
    positive_class: str | None = None,
    id_column: str | None = None,
    output_dir: str | pathlib.Path | None = None,
    llm_config: LLMConfig | None = None,
    llm_client: Any = None,
    metric: str = "euclidean",
    scaler: str = "standard",
    max_workers: int = 8,
    per_sample_reports: bool = False,
    per_sample_formats: tuple[str, ...] | list[str] | str | None = None,
    write_summary_csv: bool = True,
    write_summary_json: bool = False,
    write_interpretability: bool = True,
    generate_explanations: bool | None = None,
    verify_llm: bool = True,
    progress_callback: Callable[[int, int, SampleOutcome], None] | None = None,
    raise_on_validation_error: bool = False,
) -> PipelineOutput:
    """Run end-to-end classification on a reference + test pair.

    Parameters
    ----------
    reference, test:
        Either DataFrames or paths to a CSV/Excel/Parquet/JSON file.
    label_column:
        Name of the column in the reference dataset that holds class
        labels. MolActAgent supports two or more distinct classes; the
        test dataset may also include this column (used as a
        ground-truth annotation in the summary if present) but is not
        required to.
    k:
        Maximum number of nearest neighbors per class drawn from the
        reference dataset for each test sample.
    background:
        Optional free-text research/domain context handed to the LLM.
        Strings starting with ``"@"`` are treated as a path to a file.
    feature_descriptions:
        Optional dict (or path to JSON / YAML) mapping every column —
        including the label column — to a short description. When
        provided, validation requires complete coverage of all columns.
    positive_class:
        Deprecated. Accepted for backward compatibility but ignored
        under the multi-class pipeline; class labels are kept as-is
        throughout the system. A deprecation warning is emitted via
        ``validate_inputs`` when this is non-None.
    output_dir:
        Where reports go. Created if missing. Pass ``None`` to skip
        writing any files; in-memory outcomes are still returned.
    llm_config:
        A pre-resolved :class:`LLMConfig`. Required unless ``llm_client``
        is supplied (e.g. for testing).
    llm_client:
        A pre-built LLM client object. Useful for tests / mocks. If both
        ``llm_client`` and ``llm_config`` are given, the client wins.
    metric, scaler, max_workers:
        Forwarded to :class:`MolActAgent`.

    per_sample_reports:
        Default ``False``. When ``True``, generate per-sample
        artefacts in the formats listed by ``per_sample_formats``.
        Generating a per-sample HTML or Markdown report costs **one
        extra LLM call per test sample** (the explanation pass);
        leave this off to halve token consumption.
    per_sample_formats:
        Subset of ``("html", "md", "json")``. Only relevant when
        ``per_sample_reports=True``. Accepts a tuple/list/string
        (comma-separated). Default: all three formats.
    write_summary_csv:
        Whether to write ``summary.csv``. Default ``True``.
    write_summary_json:
        Whether to write ``summary.json``. Default ``False`` — turn on
        if you need a machine-readable summary that includes the full
        validation block and LLM stats.
    write_interpretability:
        Whether to render the aggregate interpretability files
        (``interpretability.md`` and ``interpretability.html``) and
        attach the rendered text to ``PipelineOutput``. Default
        ``True``. These are derived from the per-pair evidence already
        captured during prediction, so they cost zero extra LLM calls.

    generate_explanations:
        Deprecated alias for ``per_sample_reports``. Kept for back-
        compat; if the explicit flag is provided it overrides the
        default of ``per_sample_reports``.

    progress_callback:
        Optional ``fn(i, n, outcome)`` invoked after each test sample.
    raise_on_validation_error:
        When ``True``, validation failure raises :class:`ValueError`
        instead of returning early.

    Returns
    -------
    :class:`PipelineOutput` carrying the validation result, per-sample
    outcomes, file paths of generated artefacts, and the LLM stats dict.
    """
    if generate_explanations is not None:
        # Back-compat: --no-explanations / generate_explanations=False
        # used to disable the LLM explanation call. The new switch is
        # ``per_sample_reports`` which controls the same behaviour
        # plus the file writes.
        per_sample_reports = bool(generate_explanations)
    fmts = _normalise_per_sample_formats(per_sample_formats)
    needs_explanation_call = per_sample_reports and ("html" in fmts or "md" in fmts)
    ref_df = _coerce_to_dataframe(reference, "reference")
    test_df = _coerce_to_dataframe(test, "test")

    bg_text = (
        background if isinstance(background, str) and not background.startswith("@")
        else load_background(background)
    )
    fdescs = _coerce_descriptions(feature_descriptions)

    validation = validate_inputs(
        ref_df, test_df, label_column,
        k=k,
        feature_descriptions=fdescs,
        background=bg_text,
        positive_class=positive_class,
        id_column=id_column,
    )

    if not validation.is_valid:
        msg = (
            "Input validation failed:\n"
            + validation.format_report()
        )
        logger.error(msg)
        if raise_on_validation_error:
            raise ValueError(msg)
        return PipelineOutput(validation=validation, outcomes=[])

    detected_task = validation.summary.get("task", "classification")
    if task is not None:
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification', 'regression', or None; "
                f"got {task!r}"
            )
        if task != detected_task:
            raise ValueError(
                f"Explicit task={task!r} does not match the dataset's "
                f"detected task ({detected_task!r}). Either change the "
                "explicit task to match, or remove the override and "
                "let the auto-detection take over."
            )
    effective_task = detected_task

    for w in validation.warnings:
        logger.warning(f"[validation] {w}")

    prepared = prepare_for_agent(
        ref_df, test_df, label_column,
        feature_descriptions=fdescs,
        background=bg_text,
        positive_class=positive_class,
        id_column=id_column,
    )

    if llm_client is None:
        if llm_config is None:
            raise ValueError(
                "Either llm_config or llm_client must be supplied so the "
                "agent can talk to an LLM."
            )
        llm_client = build_llm_client(llm_config)
    if verify_llm:
        _preflight_llm(llm_client, llm_config)

    out_path: pathlib.Path | None = None
    if output_dir is not None:
        out_path = pathlib.Path(output_dir).expanduser()
        if out_path.exists() and not out_path.is_dir():
            raise NotADirectoryError(
                f"output_dir {out_path} already exists and is a file, not "
                "a directory."
            )
        out_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing pipeline artefacts to {out_path}")

    agent = MolActAgent(
        llm_client=llm_client,
        k=int(k),
        metric=metric,
        scaler=scaler,
        feature_names=list(prepared.feature_names),
        feature_descriptions=prepared.feature_descriptions,
        background=prepared.background,
        max_workers=max_workers,
    )
    agent.fit(prepared.X_ref, prepared.y_ref, task=effective_task)

    n_test = prepared.X_test.shape[0]
    outcomes: list[SampleOutcome] = []

    for i in range(n_test):
        idx_label = prepared.test_index[i] if i < len(prepared.test_index) else i
        sample_id = _build_sample_id(idx_label, i)
        truth_raw = (
            prepared.test_true_labels[i]
            if prepared.test_true_labels is not None
            else None
        )
        truth = (
            None if truth_raw is None
            else (
                f"{float(truth_raw):.4g}" if effective_task == "regression"
                else str(truth_raw)
            )
        )
        t0 = time.time()
        try:
            result = agent.predict_one(prepared.X_test[i], test_id=i)
            elapsed = time.time() - t0

            if effective_task == "regression":
                pv = result.get("predicted_value")
                iv = result.get("interval") or (None, None)
                if pv is None:
                    display_label = "REFUSED"
                    pv_for_outcome: float | None = None
                    interval_for_outcome: tuple[float, float] | None = None
                else:
                    display_label = f"{float(pv):.4g}"
                    pv_for_outcome = float(pv)
                    if iv[0] is not None and iv[1] is not None:
                        interval_for_outcome = (float(iv[0]), float(iv[1]))
                    else:
                        interval_for_outcome = None
            else:
                display_label = result.get("final_label", "Inconclusive")
                pv_for_outcome = None
                interval_for_outcome = None

            md_text = ""
            if needs_explanation_call:
                md_text = agent.generate_explanation(
                    result,
                    class_labels=prepared.class_labels,
                    target_name=label_column,
                )

            html_path = md_path = json_path = None
            if out_path is not None and per_sample_reports:
                base = _safe_filename(sample_id, f"sample_{i}")
                if "json" in fmts:
                    json_path = str(out_path / f"{base}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            result, f, indent=2, ensure_ascii=False, default=str,
                        )
                if "md" in fmts:
                    md_path = str(out_path / f"{base}.md")
                    pathlib.Path(md_path).write_text(md_text, encoding="utf-8")
                if "html" in fmts:
                    html = render_html_report(
                        result,
                        explanation=md_text,
                        class_labels=prepared.class_labels,
                        title=f"MolActAgent Report — {sample_id}",
                    )
                    html_path = str(out_path / f"{base}.html")
                    pathlib.Path(html_path).write_text(html, encoding="utf-8")

            outcome = SampleOutcome(
                test_index=i,
                sample_id=sample_id,
                predicted_label=display_label,
                confidence=float(result.get("confidence", 0.0)),
                true_label=truth,
                raw_result=result,
                explanation_md=md_text,
                explanation_html_path=html_path,
                explanation_md_path=md_path,
                raw_json_path=json_path,
                elapsed_s=elapsed,
                task=effective_task,
                predicted_value=pv_for_outcome,
                interval=interval_for_outcome,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            logger.exception(
                f"[pipeline] sample {i} failed after {elapsed:.1f}s: {exc}"
            )
            outcome = SampleOutcome(
                test_index=i,
                sample_id=sample_id,
                predicted_label="Error",
                confidence=0.0,
                true_label=truth,
                raw_result={},
                elapsed_s=elapsed,
                error=str(exc),
                task=effective_task,
            )

        outcomes.append(outcome)
        if progress_callback is not None:
            try:
                progress_callback(i + 1, n_test, outcome)
            except Exception:
                logger.debug("progress_callback raised", exc_info=True)

    summary_csv_path = summary_json_path = None
    interp_md_path = interp_html_path = None
    interp_text = ""
    if out_path is not None:
        rows = [o.to_summary_row() for o in outcomes]
        if write_summary_csv:
            df = pd.DataFrame(rows)
            summary_csv_path = str(out_path / "summary.csv")
            df.to_csv(summary_csv_path, index=False)
        if write_summary_json:
            summary_json_path = str(out_path / "summary.json")
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "validation_summary": validation.summary,
                        "validation_warnings": validation.warnings,
                        "task": effective_task,
                        "class_labels": list(prepared.class_labels),
                        "target_range": (
                            list(prepared.target_range)
                            if prepared.target_range is not None
                            else None
                        ),
                        "rows": rows,
                        "llm_stats": getattr(llm_client, "stats", {}) or {},
                    },
                    f, indent=2, ensure_ascii=False, default=str,
                )

    if write_interpretability:
        interp_text = render_interpretability_text(
            outcomes,
            class_labels=prepared.class_labels,
        )
        if out_path is not None and interp_text:
            md_text = render_interpretability_md(
                outcomes,
                class_labels=prepared.class_labels,
            )
            interp_md_path = str(out_path / "interpretability.md")
            pathlib.Path(interp_md_path).write_text(md_text, encoding="utf-8")
            html_text = render_interpretability_html(
                outcomes,
                class_labels=prepared.class_labels,
            )
            interp_html_path = str(out_path / "interpretability.html")
            pathlib.Path(interp_html_path).write_text(html_text, encoding="utf-8")

    llm_stats = None
    try:
        llm_stats = getattr(llm_client, "stats", None)
        if callable(llm_stats):
            llm_stats = llm_stats()  # type: ignore[misc]
    except Exception:
        llm_stats = None

    return PipelineOutput(
        validation=validation,
        outcomes=outcomes,
        output_dir=out_path,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        interpretability_text=interp_text,
        interpretability_md_path=interp_md_path,
        interpretability_html_path=interp_html_path,
        llm_stats=llm_stats,
    )


def run_classification(
    reference: str | pathlib.Path | pd.DataFrame,
    test: str | pathlib.Path | pd.DataFrame,
    label_column: str,
    **kwargs,
) -> PipelineOutput:
    """Classification-only convenience wrapper around
    :func:`run_prediction`.

    Asserts the auto-detected task is classification; raises
    ``ValueError`` otherwise so the caller doesn't accidentally feed
    a continuous-target dataset into the classification path.
    """
    return run_prediction(
        reference, test, label_column,
        task="classification",
        **kwargs,
    )


def run_regression(
    reference: str | pathlib.Path | pd.DataFrame,
    test: str | pathlib.Path | pd.DataFrame,
    target_column: str,
    **kwargs,
) -> PipelineOutput:
    """Regression-only convenience wrapper around
    :func:`run_prediction`.

    Argument ``target_column`` is a semantic alias for
    ``label_column`` — same column-pointer mechanism, different
    word for the regression case. Asserts the auto-detected task is
    regression; raises ``ValueError`` otherwise.
    """
    return run_prediction(
        reference, test, target_column,
        task="regression",
        **kwargs,
    )
