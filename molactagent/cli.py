"""Command-line interface for MolActAgent.

Subcommands
-----------
``molactagent validate``
    Sanity-check the user's reference + test inputs without making any
    LLM calls. Prints a structured report listing every issue found and
    exits with status 0 (clean), 1 (validation failed), or 2
    (file/format error before validation could even run).

``molactagent run``
    Validate, then run the full pipeline: classify each test sample,
    generate Markdown explanations, and write per-sample HTML / JSON
    reports plus a top-level summary CSV.

``molactagent init-config``
    Drop a starter ``llm_config.jsonc`` (JSON-with-comments) into the
    chosen path so the user can fill in their API key.

``molactagent --version``
    Print the installed package version.

Exit codes
----------
0   success
1   validation failed (semantic errors)
2   IO / format error before validation could run
3   misuse (missing required CLI flag, bad value, etc.)
4   pipeline ran but at least one test sample failed during prediction
130 user interrupted with ^C
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Sequence

from . import setup_logging
from .config import (
    DEFAULT_CONFIG_FILENAMES,
    LLMConfig,
    LLMConfigError,
    load_llm_config,
    write_example_config,
)
from .io import (
    ValidationResult,
    load_background,
    load_feature_descriptions,
    load_table,
    validate_inputs,
)
from .pipeline import PipelineOutput, run_prediction

logger = logging.getLogger(__name__)


def _parse_int_or_die(parser: argparse.ArgumentParser, name: str, value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        parser.error(f"--{name} must be an integer (got {value!r}).")


# ── argument parser construction ─────────────────────────────────────

def _add_common_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--reference", "-r", required=True,
        help="Path to the reference dataset "
             "(CSV / TSV / Excel / Parquet / JSON), or '-' to read CSV "
             "from stdin. Must contain at least 2 distinct label "
             "classes and at least one labeled sample per class.",
    )
    p.add_argument(
        "--test", "-t", required=True,
        help="Path to the test dataset (one or more samples to classify), "
             "or '-' to read CSV from stdin. Only one of --reference / "
             "--test may use '-' (otherwise both reads would consume the "
             "same stream).",
    )
    p.add_argument(
        "--stdin-format", default="csv", choices=["csv", "tsv"],
        help="Format of data read from stdin (only used when --reference "
             "or --test is '-').",
    )
    p.add_argument(
        "--label-col", "--label-column", "--target-col", "--target-column",
        "-l", required=True,
        dest="label_column",
        help="Name of the column in the reference dataset that holds "
             "the class labels (classification) or numeric target "
             "values (regression). The flag is the same for both "
             "tasks; ``--target-col`` is a semantic alias for "
             "regression callers.",
    )
    p.add_argument(
        "--task", default="auto",
        choices=["auto", "classification", "regression"],
        help="Task type. Auto-detects from the label column's dtype "
             "and value distribution by default. Pass "
             "``classification`` or ``regression`` to override the "
             "detection (e.g. to force regression on a small-"
             "cardinality integer target).",
    )
    p.add_argument(
        "--k", type=int, default=3,
        help="Maximum number of nearest neighbours per class drawn "
             "from the reference dataset for each test sample. Default: 3.",
    )
    p.add_argument(
        "--background", default=None,
        help="Optional research / domain background string. Prefix "
             "with '@' to read from a file (e.g. '@notes/intro.txt').",
    )
    p.add_argument(
        "--feature-descriptions", "--descriptions", default=None,
        dest="feature_descriptions",
        help="Optional path to a JSON / YAML file mapping every column "
             "(including the label column) to a short description.",
    )
    p.add_argument(
        "--positive-class", default=None,
        help="Deprecated. Accepted for backward compatibility but "
             "ignored under the multi-class pipeline; class labels are "
             "kept as-is throughout the system.",
    )
    p.add_argument(
        "--id-column", required=True, dest="id_column",
        help="Required: column name that holds a unique identifier for "
             "every row. The column must exist in BOTH the reference and "
             "test datasets, and IDs must be unique across the combined "
             "set (so each per-sample report has an unambiguous name).",
    )


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM API")
    g.add_argument(
        "--llm-config", default=None,
        help="Path to an LLM config file. Defaults to the first of "
             f"{list(DEFAULT_CONFIG_FILENAMES)} found by walking up from "
             "the current directory.",
    )
    g.add_argument(
        "--profile", default=None,
        help="Profile name within the config file (e.g. 'deepseek', "
             "'qwen', 'first', 'last'). Defaults to 'last'.",
    )
    g.add_argument("--api-key", default=None, help="Override LLM API key.")
    g.add_argument(
        "--base-url", default=None,
        help="Override OpenAI-compatible base URL.",
    )
    g.add_argument(
        "--model", "--model-name", default=None, dest="model_name",
        help="Override LLM model name.",
    )
    g.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature passed to every LLM call. Default: 0.0.",
    )
    g.add_argument(
        "--max-tokens", type=int, default=None,
        help="max_tokens cap for each LLM call. Default: 2000.",
    )
    g.add_argument(
        "--timeout", type=float, default=None,
        help="Per-call timeout in seconds. Default: 120.",
    )
    g.add_argument(
        "--max-retries", type=int, default=None,
        help="Number of retry attempts for transient LLM errors. Default: 10.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molactagent",
        description=(
            "MolActAgent — LLM-driven contrastive predictor for both "
            "classification (any number of classes) and regression "
            "(continuous targets). See the tutorial in docs/tutorial.md "
            "for a step-by-step guide."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=_get_version(),
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress INFO log lines on stderr.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print DEBUG log lines on stderr (overrides --quiet).",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Directory for the rotating brief/full log files. "
             "Disabled by default (logs only go to stderr).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── validate ───────────────────────────────────────────────────
    p_val = sub.add_parser(
        "validate",
        help="Check inputs for problems without calling the LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_data_args(p_val)
    p_val.add_argument(
        "--format", "-F", default=None,
        choices=["human", "json"],
        help="Stdout format. 'human' is the pretty default for "
             "interactive use; 'json' emits a single JSON object that "
             "downstream tools (jq, scripts) can parse. Auto-detected "
             "from TTY when omitted: 'human' if attached to a terminal, "
             "'json' if piped/redirected.",
    )
    p_val.add_argument(
        "--json", dest="legacy_json_flag", action="store_true",
        help="Deprecated alias for --format json.",
    )

    # ── run ────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Validate + classify each test sample + write reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_data_args(p_run)
    _add_llm_args(p_run)
    p_run.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory where on-disk reports go. OPTIONAL: if omitted, "
             "no files are written and MolActAgent acts as pure stdout "
             "middleware (use with --format json/jsonl/tsv/csv to feed "
             "another tool).",
    )
    p_run.add_argument(
        "--format", "-F", default=None,
        choices=["human", "json", "jsonl", "tsv", "csv"],
        help="Stdout format. 'human' is the pretty interactive view; "
             "'jsonl' emits one JSON line per test sample (great for "
             "`jq` / streaming consumers); 'json' emits a single JSON "
             "object with everything; 'tsv' / 'csv' emit just the "
             "verdicts table. Auto-detected from TTY when omitted: "
             "'human' if attached to a terminal, 'jsonl' if piped/"
             "redirected.",
    )
    p_run.add_argument(
        "--metric", default="euclidean",
        help="Distance metric used to find nearest reference neighbours.",
    )
    p_run.add_argument(
        "--scaler", default="standard",
        choices=["standard", "minmax", "none"],
        help="Feature scaling strategy applied before neighbour search.",
    )
    p_run.add_argument(
        "--max-workers", type=int, default=8,
        help="Maximum parallel LLM calls per test sample.",
    )
    # ── per-sample output controls ───────────────────────────────
    p_run.add_argument(
        "--per-sample", dest="per_sample_reports", action="store_true",
        help="Generate per-sample HTML / Markdown / JSON reports. OFF "
             "by default — turn this on for the full per-flower "
             "narratives. Each report adds ONE extra LLM call per "
             "test sample (the explanation pass), so token cost goes "
             "up roughly 1.5×.",
    )
    p_run.add_argument(
        "--per-sample-format", dest="per_sample_format",
        default="html,md,json",
        help="Comma-separated subset of {html,md,json} to emit when "
             "--per-sample is on. Defaults to all three.",
    )
    # Back-compat: older scripts still pass --no-explanations.
    p_run.add_argument(
        "--no-explanations", dest="generate_explanations",
        action="store_false", default=None,
        help=argparse.SUPPRESS,
    )

    # ── aggregate output controls ────────────────────────────────
    p_run.add_argument(
        "--summary-json", dest="write_summary_json", action="store_true",
        help="Also write summary.json alongside summary.csv (machine-"
             "readable summary including the validation block and "
             "LLM stats). OFF by default.",
    )
    p_run.add_argument(
        "--no-summary-csv", dest="write_summary_csv", action="store_false",
        help="Skip summary.csv. Rarely useful — left in for completeness.",
    )
    p_run.add_argument(
        "--no-interpretability", dest="write_interpretability",
        action="store_false",
        help="Skip writing interpretability.md and interpretability.html "
             "(the aggregate 'why did MolActAgent decide that?' files). "
             "These cost zero extra LLM calls so are normally kept on.",
    )

    # ── init-config ────────────────────────────────────────────────
    p_init = sub.add_parser(
        "init-config",
        help="Write a starter LLM API config file you can fill in.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_init.add_argument(
        "--path", default="llm_config.jsonc",
        help="Where to write the starter config.",
    )
    p_init.add_argument(
        "--format", default="jsonc", choices=["jsonc", "json", "txt"],
        help="Format of the starter config file. JSONC (JSON with "
             "comments) is the recommended format; 'json' is an alias "
             "and 'txt' is the legacy LLM_API.txt format.",
    )
    p_init.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting an existing file at --path.",
    )

    return parser


# ── helpers ──────────────────────────────────────────────────────────

def _get_version() -> str:
    try:
        from . import __version__
        return f"molactagent {__version__}"
    except Exception:  # pragma: no cover
        return "molactagent (dev)"


def _setup_cli_logging(args) -> None:
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    if args.log_dir:
        setup_logging(log_dir=args.log_dir, level=level, console=True)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stderr,
        )
        # ``basicConfig`` is a no-op if the root logger already has
        # handlers (which is the case under pytest, where an autouse
        # fixture forces WARNING). Force the level here so our
        # ``logger.info(...)`` "wrote summary CSV" messages reach
        # stderr in non-human modes regardless of any pre-existing
        # configuration.
        logging.getLogger().setLevel(level)


def _validation_to_jsonable(v: ValidationResult) -> dict:
    return {
        "is_valid": v.is_valid,
        "errors": list(v.errors),
        "warnings": list(v.warnings),
        "summary": v.summary,
    }


# ── stdout-format plumbing ──────────────────────────────────────────
#
# These helpers exist so the CLI can act as Unix-style middleware: in
# non-human formats stdout carries ONLY the structured payload, while
# logging, progress messages, banners, and "saved to X" notes go to
# stderr. The format is auto-detected from TTY when --format is
# omitted, so:
#
#   molactagent run ...                 # human view in your terminal
#   molactagent run ... | jq            # auto switches to jsonl
#   molactagent run ... -F json > a.json # explicit format wins


def _resolve_stdout_format(
    explicit: str | None, *, default_pipe: str, default_tty: str = "human",
) -> str:
    """Pick the stdout format. Explicit user choice always wins.

    When the user did not pass --format, fall back to ``default_tty`` if
    stdout is a real terminal, or ``default_pipe`` if stdout is being
    piped or redirected (the typical "I'm middleware" case).
    """
    if explicit:
        return explicit
    try:
        if sys.stdout.isatty():
            return default_tty
    except Exception:  # pragma: no cover - defensive
        pass
    return default_pipe


def _load_table_or_stdin(path, stdin_format: str):
    """Either ``load_table(path)`` or read CSV/TSV from stdin if path is "-"."""
    from .io import _is_stdin, _read_stdin_table  # local import: cli imports io
    if _is_stdin(path):
        return _read_stdin_table(stdin_format)
    return load_table(path)


def _check_no_double_stdin(args) -> str | None:
    """Return an error string if both --reference and --test are '-'."""
    from .io import _is_stdin
    if _is_stdin(args.reference) and _is_stdin(args.test):
        return (
            "Both --reference and --test are '-' (stdin). Only one of "
            "them may read from stdin in a single invocation, otherwise "
            "they would consume the same byte stream. Save one to a "
            "file (or to /dev/fd/N) and try again."
        )
    return None


def _outcome_machine_dict(outcome) -> dict:
    """Compact, JSON-friendly view of a SampleOutcome.

    Includes the verdict + the most useful interpretability fields
    (vote_summary, confidence_basis, top key drivers) so a downstream
    tool can act on either the verdict alone or the full attribution.
    Branches on ``outcome.task`` so regression rows surface
    ``predicted_value`` / ``interval`` instead of vote_summary.
    """
    raw = outcome.raw_result or {}
    drivers = raw.get("driver_attribution") or []
    task = outcome.task
    base = {
        "test_index": outcome.test_index,
        "sample_id": outcome.sample_id,
        "task": task,
        "predicted_label": outcome.predicted_label,
        "confidence": round(outcome.confidence, 4),
        "true_label": outcome.true_label,
        "elapsed_s": round(outcome.elapsed_s, 2),
        "error": outcome.error,
        "confidence_basis": raw.get("confidence_basis") or {},
        "reference_info": raw.get("reference_info") or {},
        "explanation_html_path": outcome.explanation_html_path,
        "explanation_md_path": outcome.explanation_md_path,
        "raw_json_path": outcome.raw_json_path,
    }
    if task == "regression":
        base.update({
            "predicted_value": outcome.predicted_value,
            "interval": (
                list(outcome.interval) if outcome.interval is not None else None
            ),
            "extrapolation_flagged": raw.get("extrapolation_flagged", False),
            "predictions": raw.get("predictions", []),
            "key_drivers": [
                {
                    "feature": fa.get("feature"),
                    "dominant_direction": fa.get("dominant_direction"),
                    "weighted_score": fa.get("weighted_score"),
                    "n_cited": fa.get("n_cited"),
                }
                for fa in drivers[:5]
            ],
        })
    else:
        base.update({
            "vote_summary": raw.get("vote_summary") or {},
            "candidate_classes": raw.get("candidate_classes") or [],
            "key_drivers": [
                {
                    "feature": fa.get("feature"),
                    "dominant_class": fa.get("dominant_class"),
                    "class_strength": fa.get("class_strength") or {},
                    "weighted_score": fa.get("weighted_score"),
                    "n_cited": fa.get("n_cited"),
                }
                for fa in drivers[:5]
            ],
            "key_features_per_class": raw.get("key_features_per_class") or {},
            "supporting_evidence": raw.get("supporting_evidence", ""),
            "opposing_evidence": raw.get("opposing_evidence", ""),
        })
    return base


def _emit_run_jsonl(result) -> None:
    """One compact JSON object per outcome on stdout."""
    for o in result.outcomes:
        json.dump(
            _outcome_machine_dict(o), sys.stdout,
            ensure_ascii=False, default=str,
        )
        sys.stdout.write("\n")
    sys.stdout.flush()


def _emit_run_json(result, *, class_labels: list | None = None) -> None:
    """A single JSON object containing everything the pipeline produced."""
    payload = {
        "validation": _validation_to_jsonable(result.validation),
        "class_labels": list(class_labels or []),
        "outcomes": [_outcome_machine_dict(o) for o in result.outcomes],
        "n_succeeded": result.n_succeeded,
        "n_failed": result.n_failed,
        "summary_csv_path": result.summary_csv_path,
        "summary_json_path": result.summary_json_path,
        "interpretability_md_path": result.interpretability_md_path,
        "interpretability_html_path": result.interpretability_html_path,
        "interpretability_text": result.interpretability_text,
        "llm_stats": result.llm_stats or {},
    }
    json.dump(
        payload, sys.stdout, indent=2,
        ensure_ascii=False, default=str,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def _emit_run_table(result, *, sep: str) -> None:
    """Verdicts-only TSV / CSV. No banners, no quoting tricks."""
    cols = (
        "sample_id", "predicted_label", "confidence",
        "true_label", "elapsed_s", "error",
    )
    writer = _CsvLikeWriter(sys.stdout, sep=sep)
    writer.row(cols)
    for o in result.outcomes:
        writer.row([
            o.sample_id,
            o.predicted_label,
            f"{o.confidence:.4f}",
            o.true_label or "",
            f"{o.elapsed_s:.2f}",
            o.error or "",
        ])
    sys.stdout.flush()


class _CsvLikeWriter:
    """Tiny CSV / TSV writer with minimal quoting for shell pipelines.

    ``csv.writer`` works too but uses ``\\r\\n`` by default; we force
    ``\\n`` and a deterministic quoting rule (only quote when a value
    contains the separator, a newline, or a double quote) so output
    plays nicely with `cut`, `awk`, and friends.
    """

    def __init__(self, fh, *, sep: str):
        self.fh = fh
        self.sep = sep

    def _quote(self, value: str) -> str:
        if any(ch in value for ch in (self.sep, "\n", '"')):
            return '"' + value.replace('"', '""') + '"'
        return value

    def row(self, values) -> None:
        self.fh.write(self.sep.join(self._quote(str(v)) for v in values))
        self.fh.write("\n")


# ── command implementations ─────────────────────────────────────────

def _emit_validation(result, fmt: str) -> None:
    if fmt == "json":
        json.dump(
            _validation_to_jsonable(result), sys.stdout,
            indent=2, ensure_ascii=False, default=str,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(result.format_report())


def _emit_validation_failure(msg: str, fmt: str, return_code: int) -> int:
    if fmt == "json":
        json.dump(
            {"is_valid": False, "errors": [msg], "warnings": [], "summary": {}},
            sys.stdout, indent=2, ensure_ascii=False,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(f"Validation: FAILED\n  - {msg}", file=sys.stdout)
    return return_code


def _cmd_validate(args, parser) -> int:
    fmt = _resolve_stdout_format(args.format, default_pipe="json")
    if getattr(args, "legacy_json_flag", False):
        fmt = "json"

    if (err := _check_no_double_stdin(args)) is not None:
        print(err, file=sys.stderr)
        return 3

    try:
        ref_df = _load_table_or_stdin(args.reference, args.stdin_format)
        test_df = _load_table_or_stdin(args.test, args.stdin_format)
    except (FileNotFoundError, IsADirectoryError, ValueError, KeyError) as e:
        return _emit_validation_failure(
            f"Could not load input data: {e}", fmt, return_code=2,
        )

    fdescs = None
    if args.feature_descriptions:
        try:
            fdescs = load_feature_descriptions(args.feature_descriptions)
        except (FileNotFoundError, IsADirectoryError, ValueError) as e:
            return _emit_validation_failure(
                f"Could not load feature descriptions: {e}", fmt, return_code=2,
            )

    bg_text = None
    if args.background:
        try:
            bg_text = load_background(args.background)
        except (FileNotFoundError, IsADirectoryError, ValueError) as e:
            return _emit_validation_failure(
                f"Could not load background text: {e}", fmt, return_code=2,
            )

    result = validate_inputs(
        ref_df, test_df, args.label_column,
        k=args.k,
        feature_descriptions=fdescs,
        background=bg_text,
        positive_class=args.positive_class,
        id_column=args.id_column,
    )
    _emit_validation(result, fmt)
    return 0 if result.is_valid else 1


def _print_run_human_summary(result, args) -> None:
    """Pretty stdout summary used in --format human mode."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    if result.outcomes:
        max_id = max(len(o.sample_id) for o in result.outcomes) + 2
    else:
        max_id = 12
    header = (
        f"{'Sample':<{max_id}}{'Predicted':<14}{'Confidence':<12}"
        f"{'True':<14}{'Time(s)':>8}"
    )
    print(header)
    print("-" * len(header))
    for o in result.outcomes:
        status = "ERR" if o.error else o.predicted_label
        print(
            f"{o.sample_id:<{max_id}}{status:<14}"
            f"{o.confidence*100:>5.1f}%      "
            f"{(o.true_label or '—'):<14}{o.elapsed_s:>8.1f}"
        )
    print("-" * len(header))
    print(f"{result.n_succeeded} succeeded, {result.n_failed} failed.")
    if result.summary_csv_path:
        print(f"Summary CSV: {result.summary_csv_path}")
    if result.summary_json_path:
        print(f"Summary JSON: {result.summary_json_path}")
    if result.llm_stats:
        s = result.llm_stats
        print(
            f"LLM stats: {s.get('calls', '?')} calls, "
            f"{s.get('total_tokens', '?')} tokens, "
            f"{s.get('total_time_s', '?')}s wall."
        )

    # Hard-failure escalation banner (also goes to stdout in human mode).
    if result.outcomes and result.n_succeeded == 0:
        print("")
        print("=" * 60)
        print(
            "All test samples failed. This usually means the LLM API "
            "rejected our calls."
        )
        print("ACTION: please check your llm_config.jsonc:")
        print("  * Is the api_key correct and not expired?")
        print("  * Is the base_url reachable from this machine?")
        print("  * Is the model_name spelled exactly as the provider expects?")
        if args.llm_config:
            print(f"  * Config file in use: {args.llm_config}")
        print("=" * 60)

    # Aggregate interpretability banner (rendered by the pipeline).
    interp = result.interpretability_text
    if interp:
        print("")
        print(interp)
        if result.interpretability_md_path:
            print(
                f"\nInterpretability (Markdown) saved to: "
                f"{result.interpretability_md_path}"
            )
        if result.interpretability_html_path:
            print(
                f"Interpretability (HTML) saved to: "
                f"{result.interpretability_html_path}"
            )


def _log_run_artefacts_to_stderr(result) -> None:
    """In non-human modes, surface 'saved to X' notes via the logger
    (which writes to stderr) so stdout stays a pure data stream."""
    paths = [
        ("summary CSV", result.summary_csv_path),
        ("summary JSON", result.summary_json_path),
        ("interpretability MD", result.interpretability_md_path),
        ("interpretability HTML", result.interpretability_html_path),
    ]
    for label, path in paths:
        if path:
            logger.info(f"wrote {label}: {path}")
    if result.llm_stats:
        s = result.llm_stats
        logger.info(
            f"LLM stats: {s.get('calls', '?')} calls, "
            f"{s.get('total_tokens', '?')} tokens, "
            f"{s.get('total_time_s', '?')}s wall."
        )
    if result.outcomes and result.n_succeeded == 0:
        logger.error(
            "All test samples failed. Check your llm_config.jsonc: "
            "api_key correct, base_url reachable, model_name accurate."
        )


def _cmd_run(args, parser) -> int:
    fmt = _resolve_stdout_format(args.format, default_pipe="jsonl")
    is_human = (fmt == "human")

    if (err := _check_no_double_stdin(args)) is not None:
        print(err, file=sys.stderr)
        return 3

    try:
        cfg = load_llm_config(
            config_path=args.llm_config,
            profile=args.profile,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    except LLMConfigError as e:
        _print_llm_config_error(e)
        return 3
    except (FileNotFoundError, ValueError) as e:
        print(f"LLM configuration error: {e}", file=sys.stderr)
        return 3

    logger.info(
        f"LLM configured: model={cfg.model_name} "
        f"base_url={cfg.base_url} (source={cfg.source or 'env/cli'})"
    )

    per_sample_reports = bool(args.per_sample_reports)
    if args.generate_explanations is False:
        per_sample_reports = False
        logger.warning(
            "--no-explanations is deprecated; pass --per-sample/"
            "--no-per-sample to control per-sample HTML/MD/JSON output. "
            "Per-sample reports are OFF by default in v0.3+."
        )

    try:
        per_sample_formats = _parse_per_sample_formats(args.per_sample_format)
    except ValueError as e:
        print(f"--per-sample-format: {e}", file=sys.stderr)
        return 3

    # Resolve stdin / file paths for reference & test up-front so the
    # pipeline never has to think about stdin.
    try:
        reference_arg = (
            _load_table_or_stdin(args.reference, args.stdin_format)
            if args.reference == "-" else args.reference
        )
        test_arg = (
            _load_table_or_stdin(args.test, args.stdin_format)
            if args.test == "-" else args.test
        )
    except (FileNotFoundError, IsADirectoryError, ValueError) as e:
        print(f"Could not load input data: {e}", file=sys.stderr)
        return 2

    # Progress messages: stderr always, but only emit in human mode (or
    # when explicitly asked) so jsonl/json consumers don't get noise.
    progress_cb = _print_progress if is_human else _stderr_progress_brief

    try:
        task_arg = (
            None if getattr(args, "task", "auto") == "auto"
            else args.task
        )
        result = run_prediction(
            reference=reference_arg,
            test=test_arg,
            label_column=args.label_column,
            task=task_arg,
            k=args.k,
            background=args.background,
            feature_descriptions=args.feature_descriptions,
            positive_class=args.positive_class,
            id_column=args.id_column,
            output_dir=args.output_dir,  # may be None now
            llm_config=cfg,
            metric=args.metric,
            scaler=args.scaler,
            max_workers=args.max_workers,
            per_sample_reports=per_sample_reports,
            per_sample_formats=per_sample_formats,
            write_summary_csv=args.write_summary_csv,
            write_summary_json=args.write_summary_json,
            write_interpretability=args.write_interpretability,
            progress_callback=progress_cb,
        )
    except LLMConfigError as e:
        _print_llm_config_error(e)
        return 3
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError) as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        if isinstance(e, (FileNotFoundError, IsADirectoryError, NotADirectoryError)):
            return 2
        return 3

    if not result.validation.is_valid:
        if is_human:
            print("\n" + result.validation.format_report(), file=sys.stderr)
            print(
                "\nNo classification was performed because validation failed. "
                "Fix the issues above and try again.",
                file=sys.stderr,
            )
        else:
            # In non-human modes still emit a structured payload so the
            # downstream tool sees the validation errors.
            class_labels: list = []
            if fmt == "json":
                _emit_run_json(result, class_labels=class_labels)
            elif fmt == "jsonl":
                # jsonl-of-zero-outcomes would be empty; emit a single
                # "validation_failed" line instead so the pipe is never
                # empty (better DX for `wc -l`).
                json.dump(
                    {
                        "validation_failed": True,
                        "errors": list(result.validation.errors),
                        "warnings": list(result.validation.warnings),
                    },
                    sys.stdout, ensure_ascii=False,
                )
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:  # tsv / csv: empty header-only file
                _emit_run_table(result, sep="\t" if fmt == "tsv" else ",")
            for line in result.validation.format_report().splitlines():
                logger.error(line)
        return 1

    class_labels = list(result.validation.summary.get("class_labels") or [])

    if is_human:
        _print_run_human_summary(result, args)
    else:
        if fmt == "jsonl":
            _emit_run_jsonl(result)
        elif fmt == "json":
            _emit_run_json(result, class_labels=class_labels)
        elif fmt == "tsv":
            _emit_run_table(result, sep="\t")
        elif fmt == "csv":
            _emit_run_table(result, sep=",")
        _log_run_artefacts_to_stderr(result)

    return 0 if result.n_failed == 0 else 4


def _stderr_progress_brief(i: int, n: int, outcome) -> None:
    """Quiet, single-line progress message (stderr). Used by non-human
    output formats so downstream pipes see no noise on stdout."""
    pred = outcome.error or outcome.predicted_label
    print(
        f"  [{i}/{n}] {outcome.sample_id} -> {pred}",
        file=sys.stderr,
    )


def _print_llm_config_error(err: LLMConfigError) -> None:
    """Render an LLMConfigError as an unmissable banner."""
    lines = [
        "",
        "=" * 68,
        "LLM CONFIGURATION ERROR",
        "=" * 68,
        err.user_facing_message(),
        "=" * 68,
    ]
    print("\n".join(lines), file=sys.stderr)


def _parse_per_sample_formats(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated --per-sample-format value.

    Returns a deduped tuple of {html, md, json} entries in the order
    the user listed them. Raises :class:`ValueError` on unknown items.
    """
    valid = {"html", "md", "json"}
    items = [s.strip().lower() for s in (spec or "").split(",") if s.strip()]
    if not items:
        return ("html", "md", "json")
    bad = [s for s in items if s not in valid]
    if bad:
        raise ValueError(
            f"unknown format(s) {bad!r}. Choose any combination of "
            f"{sorted(valid)!r} (comma-separated)."
        )
    seen: list[str] = []
    for s in items:
        if s not in seen:
            seen.append(s)
    return tuple(seen)


def _cmd_init_config(args, parser) -> int:
    try:
        path = write_example_config(
            args.path, fmt=args.format, overwrite=args.overwrite,
        )
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return 3
    print(f"Wrote starter config to {path}")
    print("Now edit it and replace the placeholder values with your API key.")
    return 0


def _print_progress(i: int, n: int, outcome) -> None:
    pred = outcome.error or outcome.predicted_label
    print(
        f"  [{i}/{n}] {outcome.sample_id} -> {pred} "
        f"(confidence {outcome.confidence*100:.0f}%, "
        f"{outcome.elapsed_s:.1f}s)",
        file=sys.stderr,
    )


# ── entry points ────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_cli_logging(args)

    try:
        if args.command == "validate":
            return _cmd_validate(args, parser)
        if args.command == "run":
            return _cmd_run(args, parser)
        if args.command == "init-config":
            return _cmd_init_config(args, parser)
        parser.error(f"Unknown command {args.command!r}")
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
