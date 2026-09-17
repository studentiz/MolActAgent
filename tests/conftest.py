"""Shared pytest fixtures for the MolActAgent test suite.

Includes:
  - ``FakeLLM`` — a deterministic stand-in for ``LLMClient`` that
    inspects each multi-way tuple prompt and emits a syntactically
    valid response. This lets us exercise the full pipeline
    (validation -> agent -> report) in tests without ever talking to
    a real LLM.
  - ``project_root``, ``tmp_workspace`` — convenient path helpers.
  - ``simple_reference`` / ``simple_test`` — tiny binary synthetic
    datasets (the M=2 special case).
  - ``multiclass_reference`` / ``multiclass_test`` — tiny three-class
    synthetic datasets exercising the M=2 (clearly one cluster) and
    M=3 (centroid sample) paths.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import threading

import numpy as np
import pandas as pd
import pytest


# ── Project paths ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def tmp_workspace(tmp_path) -> pathlib.Path:
    """A clean temp directory for every test that writes files."""
    return tmp_path


# ── Synthetic data ───────────────────────────────────────────────────

@pytest.fixture()
def simple_reference() -> pd.DataFrame:
    """A 6-row, 4-feature reference set with two classes ('active'/'inactive')."""
    rng = np.random.default_rng(42)
    n_per = 3
    pos = rng.normal(loc=5.0, scale=0.4, size=(n_per, 4))
    neg = rng.normal(loc=-5.0, scale=0.4, size=(n_per, 4))
    X = np.vstack([pos, neg])
    y = ["active"] * n_per + ["inactive"] * n_per
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
    df["activity"] = y
    return df


@pytest.fixture()
def simple_test() -> pd.DataFrame:
    """A 2-row test set; the first row should clearly look like 'active'."""
    rng = np.random.default_rng(7)
    pos_like = rng.normal(loc=5.0, scale=0.3, size=(1, 4))
    neg_like = rng.normal(loc=-5.0, scale=0.3, size=(1, 4))
    X = np.vstack([pos_like, neg_like])
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
    df["activity"] = ["active", "inactive"]
    return df


@pytest.fixture()
def multiclass_reference() -> pd.DataFrame:
    """A 12-row, 4-feature reference set with three classes
    ('low' / 'mid' / 'high'), well-separated in feature space."""
    rng = np.random.default_rng(123)
    n_per = 4
    low = rng.normal(loc=-5.0, scale=0.3, size=(n_per, 4))
    mid = rng.normal(loc=0.0, scale=0.3, size=(n_per, 4))
    high = rng.normal(loc=5.0, scale=0.3, size=(n_per, 4))
    X = np.vstack([low, mid, high])
    y = ["low"] * n_per + ["mid"] * n_per + ["high"] * n_per
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
    df["bucket"] = y
    return df


@pytest.fixture()
def multiclass_test() -> pd.DataFrame:
    """Three test rows targeting different M cases:

    * row 0 — clearly inside the ``low`` cluster (expect M=2: low + mid).
    * row 1 — clearly inside the ``high`` cluster (expect M=2: high + mid).
    * row 2 — sitting near the centroid of all three clusters
      (expect M=3 in many configurations).
    """
    X = np.array([
        [-5.0, -5.0, -5.0, -5.0],   # low
        [5.0, 5.0, 5.0, 5.0],       # high
        [0.0, 0.0, 0.0, 0.0],       # centroid (mid, but all 3 are nearby)
    ])
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
    df["bucket"] = ["low", "high", "mid"]
    return df


def _with_sample_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return a copy of ``df`` with a unique ``sample_id`` column prepended."""
    out = df.copy()
    out.insert(0, "sample_id", [f"{prefix}_{i}" for i in range(len(out))])
    return out


@pytest.fixture()
def simple_reference_with_ids(simple_reference) -> pd.DataFrame:
    """Reference DataFrame with a unique ``sample_id`` column (``R_0``..``R_5``)."""
    return _with_sample_ids(simple_reference, "R")


@pytest.fixture()
def simple_test_with_ids(simple_test) -> pd.DataFrame:
    """Test DataFrame with globally-unique IDs (``T_0``, ``T_1``)."""
    return _with_sample_ids(simple_test, "T")


@pytest.fixture()
def multiclass_reference_with_ids(multiclass_reference) -> pd.DataFrame:
    return _with_sample_ids(multiclass_reference, "R")


@pytest.fixture()
def multiclass_test_with_ids(multiclass_test) -> pd.DataFrame:
    return _with_sample_ids(multiclass_test, "T")


@pytest.fixture()
def simple_descriptions() -> dict[str, str]:
    return {
        "alpha": "First measurement",
        "beta": "Second measurement",
        "gamma": "Third measurement",
        "delta": "Fourth measurement",
        "activity": "Class label (active vs inactive)",
    }


@pytest.fixture()
def multiclass_descriptions() -> dict[str, str]:
    return {
        "alpha": "First measurement",
        "beta": "Second measurement",
        "gamma": "Third measurement",
        "delta": "Fourth measurement",
        "bucket": "Class label (low / mid / high)",
    }


# ── Reusable on-disk dataset (small CSVs) ────────────────────────────

def _write_csv(path: pathlib.Path, df: pd.DataFrame) -> pathlib.Path:
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def csv_pair(tmp_workspace, simple_reference, simple_test):
    """A (reference_csv, test_csv) path pair WITHOUT an id_column.

    Used by tests that exercise the plain Python API where id_column is
    optional.
    """
    ref = _write_csv(tmp_workspace / "ref.csv", simple_reference)
    tst = _write_csv(tmp_workspace / "test.csv", simple_test)
    return ref, tst


@pytest.fixture()
def csv_pair_with_ids(
    tmp_workspace, simple_reference_with_ids, simple_test_with_ids,
):
    """A (reference_csv, test_csv) path pair with a ``sample_id`` column.

    Used by the CLI tests, which now require ``--id-column``.
    """
    ref = _write_csv(tmp_workspace / "ref.csv", simple_reference_with_ids)
    tst = _write_csv(tmp_workspace / "test.csv", simple_test_with_ids)
    return ref, tst


@pytest.fixture()
def multiclass_csv_pair(
    tmp_workspace, multiclass_reference, multiclass_test,
):
    ref = _write_csv(tmp_workspace / "mc_ref.csv", multiclass_reference)
    tst = _write_csv(tmp_workspace / "mc_test.csv", multiclass_test)
    return ref, tst


@pytest.fixture()
def multiclass_csv_pair_with_ids(
    tmp_workspace, multiclass_reference_with_ids, multiclass_test_with_ids,
):
    ref = _write_csv(tmp_workspace / "mc_ref.csv", multiclass_reference_with_ids)
    tst = _write_csv(tmp_workspace / "mc_test.csv", multiclass_test_with_ids)
    return ref, tst


# ── Fake LLM that satisfies the agent's contract ─────────────────────

class FakeLLM:
    """A drop-in replacement for ``LLMClient`` that never calls a real API.

    Strategy:
      - For prompts tagged ``*_ANA*`` (tuple classification), parse the
        prompt to discover the candidate classes, the discriminative
        feature names, and the deterministic baseline class, then emit
        a JSON object that agrees with the baseline so
        :meth:`_check_tuple` passes on the first attempt.
      - For prompts tagged ``*_EXPLAIN*``, return a short Markdown body.
    """

    def __init__(self, *, fail_first_n: int = 0):
        self._lock = threading.Lock()
        self._call_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_time = 0.0
        self._call_log: list[dict] = []
        self._throttle_count = 0
        self._fail_remaining = fail_first_n
        self.calls: list[dict] = []

    @property
    def throttle_count(self) -> int:
        return self._throttle_count

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_time_s": 0.0,
            "avg_time_s": 0.0,
        }

    @property
    def call_log(self) -> list[dict]:
        return list(self._call_log)

    def chat(self, messages, tag="", max_tokens=None, temperature=None):
        with self._lock:
            self._call_count += 1
            cid = self._call_count
            if self._fail_remaining > 0:
                self._fail_remaining -= 1
                raise RuntimeError("simulated transient LLM failure")
        self.calls.append({"id": cid, "tag": tag, "messages": messages})
        self._call_log.append({"call_id": cid, "tag": tag})

        joined = "\n".join(m.get("content", "") for m in messages)
        # Regression EXPLAIN prompts carry the unique
        # "regression-prediction report" phrase from the system
        # message; the legacy bootstrap-tuple regression prompt's
        # "WITNESS BASELINE" marker is also matched for backwards
        # compatibility with any caller that still constructs the
        # old prompt directly.
        is_regression_prompt = (
            "regression-prediction" in joined
            or "WITNESS BASELINE" in joined
        )
        if "EXPLAIN" in tag.upper():
            if is_regression_prompt or "_REG" in tag.upper():
                return _build_fake_regression_explanation()
            return _build_fake_explanation()
        if "WITNESS BASELINE" in joined:
            # Legacy regression-tuple prompt path — not used by the
            # deployed CCR backend, kept for historical callers.
            return _build_fake_regression_tuple_json(joined)
        # Default: classification tuple prompt. CCR's two phases also
        # land here because they delegate to the same multi-class
        # contrastive engine.
        return _build_fake_tuple_json(joined)

    def ping(self, *, timeout: float = 30.0) -> dict:
        return {"ok": True, "elapsed_s": 0.0, "response": "ping"}


# ── Fake-LLM helpers ────────────────────────────────────────────────

_CANDIDATES_RE = re.compile(
    r'Candidate classes for this comparison:\s*([^.]+)\.'
)
_BASELINE_RE = re.compile(
    r'Baseline verdict from this tally:\s*\*\*([^*]+)\*\*'
)
_ALIGN_LINE_RE = re.compile(
    r'Test is numerically closest to (\S+?)_Ref on (\d+) feature'
)
_NEUTRAL_RE = re.compile(
    r'Test is equidistant.*?on (\d+) feature'
)


def _parse_candidate_classes(prompt: str) -> list[str]:
    m = _CANDIDATES_RE.search(prompt)
    if not m:
        return []
    raw = m.group(1)
    return re.findall(r'"([^"]+)"', raw)


def _parse_alignment(prompt: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in _ALIGN_LINE_RE.finditer(prompt):
        out[m.group(1)] = int(m.group(2))
    return out


def _parse_feature_rows(prompt: str, n_classes: int) -> list[dict]:
    """Parse rows of the form 'name | v1 | v2 | ... | vN | test_v'."""
    rows: list[dict] = []
    name_re = re.compile(r"^[A-Za-z_][A-Za-z_0-9.]*$")
    for line in prompt.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != n_classes + 2:
            continue
        name = parts[0]
        if not name_re.fullmatch(name):
            continue
        try:
            ref_vals = [float(p) for p in parts[1:1 + n_classes]]
            test_v = float(parts[-1])
        except ValueError:
            continue
        rows.append({"name": name, "refs": ref_vals, "test": test_v})
    return rows


def _build_fake_tuple_json(prompt: str) -> str:
    candidates = _parse_candidate_classes(prompt)
    baseline_match = _BASELINE_RE.search(prompt)
    baseline_raw = baseline_match.group(1).strip() if baseline_match else "Unknown"
    n_align = _parse_alignment(prompt)

    if not candidates:
        # Defensive fallback: synthesise a minimally valid record.
        return json.dumps({
            "label": "Unknown",
            "confidence": 0.5,
            "key_drivers": [
                {
                    "feature": "f0", "favored_class": "Unknown",
                    "strength": 0.5, "test_value": 0,
                    "favored_value": 0, "why": "synthetic",
                },
                {
                    "feature": "f1", "favored_class": "Unknown",
                    "strength": 0.5, "test_value": 0,
                    "favored_value": 0, "why": "synthetic",
                },
            ],
            "supporting_features": [],
            "opposing_features": [],
            "summary": "Synthetic verdict (no candidates parsed).",
            "counter_evidence": "none significant",
            "uncertainty": "n/a",
        })

    rows = _parse_feature_rows(prompt, len(candidates))
    baseline = baseline_raw if baseline_raw in candidates else "Unknown"
    fav_default = baseline if baseline in candidates else candidates[0]

    pick = rows[: max(2, min(4, len(rows)))]
    drivers = []
    for r in pick:
        try:
            fav_idx = candidates.index(fav_default)
            fav_value = r["refs"][fav_idx]
        except (ValueError, IndexError):
            fav_value = r["refs"][0]
        drivers.append({
            "feature": r["name"],
            "favored_class": fav_default,
            "strength": 0.8,
            "test_value": r["test"],
            "favored_value": fav_value,
            "why": f"feature {r['name']} aligns with the baseline class",
        })
    if len(drivers) < 2:
        # Pad with synthetic entries so the rule-checker's "at least 2"
        # requirement is satisfied even on tiny prompts.
        for i in range(2 - len(drivers)):
            drivers.append({
                "feature": f"_pad_{i}",
                "favored_class": fav_default,
                "strength": 0.5,
                "test_value": 0,
                "favored_value": 0,
                "why": "synthetic padding",
            })

    total = sum(n_align.values())
    if total > 0 and len(n_align) >= 1:
        sorted_counts = sorted(n_align.values(), reverse=True)
        top = sorted_counts[0]
        runner = sorted_counts[1] if len(sorted_counts) > 1 else 0
        margin = (top - runner) / total
    else:
        margin = 0.0
    confidence = round(0.5 + 0.4 * max(0.0, min(1.0, margin)), 2)

    label = baseline if baseline in candidates else "Unknown"
    return json.dumps({
        "label": label,
        "confidence": confidence,
        "key_drivers": drivers,
        "supporting_features": [r["name"] for r in pick],
        "opposing_features": [],
        "summary": "Synthetic deterministic-aligned verdict.",
        "counter_evidence": "none significant",
        "uncertainty": "n/a",
    })


def _build_fake_explanation() -> str:
    return (
        "## Executive Summary\n\n"
        "The submitted sample is consistent with the indicated "
        "reference profile based on the contrastive comparisons.\n\n"
        "## Key Drivers\n\n"
        "| Feature | Sample value | Verdict |\n"
        "|---------|--------------|---------|\n"
        "| alpha   | 5.0          | first   |\n\n"
        "## Why These Drivers Matter\n\n"
        "### alpha\n\n"
        "Alpha is decisive because the sample value sits squarely "
        "inside the indicated cluster's range and far from the "
        "alternatives.\n\n"
        "## Robustness Across Comparisons\n\n"
        "All evaluated reference tuples agreed.\n\n"
        "## Counter-evidence Considered\n\n"
        "No appreciable counter-evidence emerged.\n\n"
        "## Confidence and Caveats\n\n"
        "Confidence reflects unanimous tuple agreement on a small "
        "reference set.\n\n"
        "## Recommended Next Steps\n\n"
        "1. Confirm with an independent assay.\n"
        "2. Expand the reference set if more decisive evidence is needed.\n"
    )


@pytest.fixture()
def fake_llm():
    return FakeLLM()


@pytest.fixture()
def patched_pipeline_llm(monkeypatch):
    """Fixture: monkey-patch the pipeline's LLM client builder so any
    ``run_classification`` / ``run_regression`` / ``molactagent.cli``
    invocation transparently uses a ``FakeLLM`` (or any client the
    test supplies).

    Returns a callable: ``patch(client) -> None``. Tests do something
    like::

        def test_cli_path(patched_pipeline_llm, fake_llm, tmp_path):
            patched_pipeline_llm(fake_llm)   # any subsequent
                                              # build_llm_client(cfg) call
                                              # returns fake_llm
            # ...invoke the CLI or run_prediction(...)...

    Encapsulates a pattern that several validation tests previously
    rolled by hand. Centralising it here lets future tests opt into
    the same in-process CLI path without re-implementing the
    monkey-patch.
    """
    def _patch(client):
        monkeypatch.setattr(
            "molactagent.pipeline.build_llm_client",
            lambda cfg: client,
        )
    return _patch


# ── Regression FakeLLM helpers ───────────────────────────────────────

_REG_BASELINE_RE = re.compile(
    r"Median target of the \d+ chosen references:\s*([0-9.\-eE+]+)"
)
_REG_RANGE_RE = re.compile(
    r"Observed target range[^:]*:\s*\[([0-9.\-eE+]+),\s*([0-9.\-eE+]+)\]"
)


def _parse_regression_main_table(prompt: str) -> list[str]:
    """Find the feature names listed in the regression prompt's main
    table header. Returns the column names in the order they appear,
    excluding the ``Identifier`` and ``Target`` columns.
    """
    for line in prompt.splitlines():
        if "|" not in line or "Identifier" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0] == "Identifier" and parts[1] == "Target":
            return parts[2:]
    return []


def _build_fake_regression_tuple_json(prompt: str) -> str:
    """Echo the witness baseline prediction with two simple drivers
    referencing features from the prompt's main table."""
    bm = _REG_BASELINE_RE.search(prompt)
    baseline = float(bm.group(1)) if bm else 0.0
    rm = _REG_RANGE_RE.search(prompt)
    target_min = float(rm.group(1)) if rm else baseline - 1.0
    target_max = float(rm.group(2)) if rm else baseline + 1.0

    feature_names = _parse_regression_main_table(prompt)
    if len(feature_names) < 2:
        feature_names = (feature_names + ["f0", "f1"])[:2]

    drivers = []
    for i, fname in enumerate(feature_names[:3]):
        direction = "increases" if i % 2 == 0 else "decreases"
        drivers.append({
            "feature": fname,
            "direction": direction,
            "strength": 0.7 - 0.1 * i,
            "test_value": 0.0,
            "why": f"feature {fname} positions Test in the local landscape",
        })
    if len(drivers) < 2:
        drivers.append({
            "feature": "_pad_0",
            "direction": "mixed",
            "strength": 0.5,
            "test_value": 0.0,
            "why": "synthetic padding so the validator has 2+ drivers",
        })

    # Snap the predicted value into the observed target range so the
    # validator never flags extrapolation on synthetic prompts.
    predicted = max(min(baseline, target_max), target_min)
    return json.dumps({
        "predicted_value": predicted,
        "interval": [
            max(target_min, predicted - 1.0),
            min(target_max, predicted + 1.0),
        ],
        "confidence": 0.8,
        "extrapolation_flag": False,
        "drivers": drivers,
        "rationale": "Synthetic regression: echoed witness baseline.",
        "uncertainty": "Synthetic FakeLLM emits the witness baseline.",
    })


def _build_fake_regression_explanation() -> str:
    return (
        "## Executive Summary\n\n"
        "Synthetic regression demo: the predicted value matches the "
        "witness baseline.\n\n"
        "## Where the Sample Sits\n\n"
        "The sample sits comfortably inside the observed target "
        "range; the references collectively bracket it from both "
        "sides.\n\n"
        "## Key Drivers\n\n"
        "| Feature | Sample value | Direction | Net weighted strength |\n"
        "|---------|--------------|-----------|-----------------------|\n"
        "| f1      | 0.0          | increases | +0.70                 |\n"
        "| f2      | 0.0          | decreases | -0.50                 |\n\n"
        "## Why These Drivers Matter\n\n"
        "### f1\n\n"
        "Feature f1 pushes the prediction up.\n\n"
        "## Confidence and Caveats\n\n"
        "Synthetic confidence reflects FakeLLM agreement.\n\n"
        "## Recommended Next Steps\n\n"
        "1. Use a real LLM to validate.\n"
    )


# ── Regression fixtures ──────────────────────────────────────────────


@pytest.fixture()
def regression_reference() -> pd.DataFrame:
    """A 24-row, 4-feature reference set with a continuous target.

    The target is a smooth function of the first two features so a
    competent regressor should track it; the noise term keeps it
    realistic.
    """
    rng = np.random.default_rng(2026)
    n = 24
    X = rng.normal(loc=0.0, scale=1.5, size=(n, 4))
    target = (
        5.0 * X[:, 0]
        + 2.0 * X[:, 1]
        + 0.3 * X[:, 2] * X[:, 3]
        + rng.normal(0.0, 0.5, size=n)
        + 50.0
    )
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
    df["target"] = target.astype(float)
    return df


@pytest.fixture()
def regression_test() -> pd.DataFrame:
    """Three test rows with known true target values."""
    rng = np.random.default_rng(99)
    X = rng.normal(loc=0.0, scale=1.5, size=(3, 4))
    target = 5.0 * X[:, 0] + 2.0 * X[:, 1] + 50.0
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
    df["target"] = target.astype(float)
    return df


@pytest.fixture()
def regression_reference_with_ids(regression_reference) -> pd.DataFrame:
    return _with_sample_ids(regression_reference, "R")


@pytest.fixture()
def regression_test_with_ids(regression_test) -> pd.DataFrame:
    return _with_sample_ids(regression_test, "T")


@pytest.fixture()
def regression_csv_pair(
    tmp_workspace, regression_reference, regression_test,
):
    ref = _write_csv(tmp_workspace / "reg_ref.csv", regression_reference)
    tst = _write_csv(tmp_workspace / "reg_test.csv", regression_test)
    return ref, tst


@pytest.fixture()
def regression_csv_pair_with_ids(
    tmp_workspace,
    regression_reference_with_ids,
    regression_test_with_ids,
):
    """Regression CSVs with a ``sample_id`` column — used by CLI
    tests, which require ``--id-column``."""
    ref = _write_csv(tmp_workspace / "reg_ref.csv", regression_reference_with_ids)
    tst = _write_csv(tmp_workspace / "reg_test.csv", regression_test_with_ids)
    return ref, tst


@pytest.fixture()
def regression_descriptions() -> dict[str, str]:
    return {
        "alpha": "First measurement (drives the target up)",
        "beta": "Second measurement (smaller positive influence)",
        "gamma": "Third measurement (interaction effect)",
        "delta": "Fourth measurement (interaction partner)",
        "target": "Continuous target value to predict.",
    }


# ── Logging quietness for the test session ───────────────────────────

@pytest.fixture(autouse=True)
def _quiet_logging():
    logging.getLogger().setLevel(logging.WARNING)
    yield
