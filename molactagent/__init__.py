"""MolActAgent — LLM-driven contrastive predictor with a layered Python API.

Public surface, organised by intent:

Engine layer (the original surface — unchanged, every existing test
exercises this directly):

    molactagent.MolActAgent
    molactagent.CrossClassificationRegressor
    molactagent.LLMClient

High-level pipeline (the original "all-in-one" entry point — unchanged):

    molactagent.run_prediction        # auto-detect task
    molactagent.run_classification
    molactagent.run_regression
    molactagent.PipelineOutput, molactagent.SampleOutcome

IO / validation / config (unchanged):

    molactagent.load_table, molactagent.load_feature_descriptions
    molactagent.load_background
    molactagent.validate_inputs, molactagent.prepare_for_agent
    molactagent.ValidationResult, molactagent.PreparedData
    molactagent.detect_task
    molactagent.load_llm_config, molactagent.build_llm_client
    molactagent.LLMConfig

scikit-learn-style estimators (NEW — for users who want fit/predict/
predict_proba/score and integration with sklearn pipelines):

    molactagent.MolActAgentClassifier
    molactagent.MolActAgentRegressor

Result wrappers (NEW — typed access to engine output dicts):

    molactagent.ClassificationResult
    molactagent.RegressionResult
    molactagent.wrap_result

Scanpy-style analytical namespaces (NEW — discoverable via
tab-completion):

    molactagent.pp        # preprocessing namespace
    molactagent.tl        # tools / downstream analysis (+ alias: molactagent.tools)
    molactagent.metrics   # classification + regression metrics
    molactagent.datasets  # bundled examples (iris, penguins, BSLT)

The two-letter Scanpy-style names are the canonical import path.
``molactagent.tools`` is exposed as an alias of ``molactagent.tl`` for
users who prefer the full-word convention. (We do not ship a
``molactagent.preprocessing`` alias because the package already has
an internal ``preprocessing`` submodule — the public preprocessing
surface is ``molactagent.pp``.) Every namespace is a regular Python
package — ``ca.pp.<TAB>`` shows what's available.

The CLI (``molactagent run`` / ``molactagent validate`` /
``molactagent init-config``) and the high-level ``run_*`` functions
remain the canonical orchestrated entry points and are not
deprecated. The estimator + namespace layer is purely additive —
backward-compatible by construction. See ``docs/design.md`` Part III
for how the layers fit together.
"""

import logging
import pathlib
from datetime import datetime

from .agent import MolActAgent
from .cross_classification import CrossClassificationRegressor
from .estimators import MolActAgentClassifier, MolActAgentRegressor
from .llm import LLMClient
from .report import render_html_report
from .result import ClassificationResult, DriverEntry, RegressionResult, wrap_result

__version__ = "0.3.0"
__all__ = [
    # ── engine ───────────────────────────────────────
    "MolActAgent",
    "CrossClassificationRegressor",
    "LLMClient",
    # ── reports + logging ────────────────────────────
    "render_html_report",
    "setup_logging",
    # ── high-level pipeline ──────────────────────────
    "run_prediction",
    "run_classification",
    "run_regression",
    "PipelineOutput",
    "SampleOutcome",
    # ── IO + validation ──────────────────────────────
    "validate_inputs",
    "load_table",
    "load_feature_descriptions",
    "load_background",
    "ValidationResult",
    "PreparedData",
    "prepare_for_agent",
    "detect_task",
    # ── config ───────────────────────────────────────
    "load_llm_config",
    "build_llm_client",
    "LLMConfig",
    # ── estimator wrappers (NEW) ─────────────────────
    "MolActAgentClassifier",
    "MolActAgentRegressor",
    # ── result wrappers (NEW) ────────────────────────
    "ClassificationResult",
    "RegressionResult",
    "DriverEntry",
    "wrap_result",
]


def __getattr__(name):
    """Lazy re-exports for the public surface.

    Importing :mod:`molactagent` should be cheap; the IO / pipeline
    helpers pull in :mod:`pandas` so we only import them when first
    accessed. The Scanpy-style namespaces (``pp``, ``tl``, ``metrics``,
    ``datasets``) and their full-word aliases are also lazy so that a
    plain ``import molactagent`` doesn't pull in scikit-learn,
    scipy.stats, RDKit, etc. unless the user actually reaches for
    them.
    """
    if name in {
        "validate_inputs",
        "load_table",
        "load_feature_descriptions",
        "load_background",
        "ValidationResult",
        "PreparedData",
        "prepare_for_agent",
        "detect_task",
    }:
        from . import io as _io
        return getattr(_io, name)
    if name in {
        "run_prediction",
        "run_classification",
        "run_regression",
        "PipelineOutput",
        "SampleOutcome",
    }:
        from . import pipeline as _pl
        return getattr(_pl, name)
    if name in {"load_llm_config", "build_llm_client", "LLMConfig"}:
        from . import config as _cfg
        return getattr(_cfg, name)

    # Scanpy-style analytical namespaces. ``molactagent.tools`` is
    # an alias for ``molactagent.tl``; ``molactagent.preprocessing``
    # is NOT aliased because the package already has an internal
    # ``preprocessing`` submodule that we do not want to shadow.
    #
    # CRITICAL: we use ``importlib.import_module`` rather than
    # ``from . import <name>`` here. The ``from . import <name>``
    # form triggers attribute lookup on this module before the
    # submodule is added to ``sys.modules``, which loops back into
    # this very __getattr__ and recurses infinitely. Going through
    # importlib directly bypasses the parent's attribute machinery
    # and only consults ``sys.modules`` / the import machinery.
    import importlib as _il
    if name == "pp":
        return _il.import_module("molactagent.pp")
    if name in {"tl", "tools"}:
        return _il.import_module("molactagent.tl")
    if name == "metrics":
        return _il.import_module("molactagent.metrics")
    if name == "datasets":
        return _il.import_module("molactagent.datasets")

    raise AttributeError(f"module 'molactagent' has no attribute {name!r}")


def setup_logging(
    log_dir: str | pathlib.Path = "logs",
    level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    console: bool = True,
) -> dict[str, pathlib.Path]:
    """Configure dual-file + console logging.

    Returns dict with 'brief' and 'full' log file paths.
    - brief log: INFO level — concise runtime monitoring
    - full log:  DEBUG level — complete LLM responses for post-analysis
    - console:   INFO level
    """
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    brief_file = log_dir / f"molactagent_{timestamp}_brief.log"
    full_file = log_dir / f"molactagent_{timestamp}_full.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh_full = logging.FileHandler(full_file, encoding="utf-8")
    fh_full.setLevel(logging.DEBUG)
    fh_full.setFormatter(fmt)
    root.addHandler(fh_full)

    fh_brief = logging.FileHandler(brief_file, encoding="utf-8")
    fh_brief.setLevel(logging.INFO)
    fh_brief.setFormatter(fmt)
    root.addHandler(fh_brief)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    log = logging.getLogger(__name__)
    log.info(f"Brief log: {brief_file}")
    log.info(f"Full log:  {full_file}")
    return {"brief": brief_file, "full": full_file}
