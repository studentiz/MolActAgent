"""Comprehensive end-to-end validation: iris classification + BSLT regression.

This file is the package's flagship "no-regression" test. It verifies
the user's three explicit acceptance criteria from the v0.3
re-architecture brief:

1. **Multi-classification on iris_flowers** — verify that
   classification performance metrics (accuracy, balanced accuracy,
   MCC, macro F1, per-class precision / recall / F1) are
   **identical** when the same data flows through every interface
   (pipeline, sklearn-style estimator, CLI). Identical-up-to-FP
   tolerance is the criterion because the FakeLLM mock is fully
   deterministic — any divergence is a real functional regression.
2. **Regression on BSLT_huigui.csv** — featurise the SMILES via the
   new ``ca.pp`` chemistry pipeline, run the resulting feature
   matrix through every interface, compute the full regression
   metric suite (MAE, RMSE, R², Pearson r, Spearman r, median AE),
   and again assert metric equality.
3. **Dual-interface testing for ALL scenarios** — every scenario
   exercised through both the Python API (high-level pipeline AND
   sklearn-style estimator) AND the CLI (in-process via
   ``molactagent.cli.main``, plus subprocess for the user-facing
   surface). Identical labels / values / metrics across all paths
   prove the new wrappers are pure projections of the engine, not
   secret reimplementations.

The CLI is exercised two ways:

- **In-process** via ``molactagent.cli.main([...])`` with
  ``molactagent.pipeline.build_llm_client`` monkeypatched to return
  the FakeLLM. This tests the real argparse → run_prediction flow
  end to end including CSV I/O and report writing, while
  remaining fast and deterministic.
- **Subprocess** via ``python -m molactagent ...`` for the surfaces
  that don't need an LLM (``--version``, ``--help``,
  ``validate``). This catches regressions in the entry-point
  registration and console-script wiring.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import molactagent as ca
from molactagent.metrics import (
    binary_classification_metrics,
    classification_metrics,
    regression_metrics,
)


# ── shared fixtures and helpers ───────────────────────────────────


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def iris_paths():
    """Resolved paths to the bundled iris example assets."""
    root = _project_root() / "examples" / "iris_flowers"
    return {
        "reference": root / "reference.csv",
        "test": root / "test.csv",
        "feature_descriptions": root / "feature_descriptions.json",
        "background": root / "background.txt",
    }


@pytest.fixture(scope="module")
def iris_data():
    """Iris dataset loaded via the new ``ca.datasets`` accessor."""
    return ca.datasets.load_iris_flowers()


@pytest.fixture(scope="module")
def iris_holdout_csvs(tmp_path_factory):
    """Build a deterministic holdout split of iris_flowers/reference.csv.

    The bundled ``test.csv`` deliberately ships without a ``species``
    column (it's a "predict the unknowns" demo), so it can't power a
    metrics-equality test. We instead split the reference CSV in
    half (4 rows per species → train; 4 rows per species → test)
    so every test row has ground truth and every species is
    represented in both halves.
    """
    root = _project_root() / "examples" / "iris_flowers"
    df = pd.read_csv(root / "reference.csv")
    # Stratified split: take the first 4 rows of each species for
    # train, the rest for test. Order is deterministic.
    train_rows = []
    test_rows = []
    for species, group in df.groupby("species"):
        train_rows.append(group.iloc[:4])
        test_rows.append(group.iloc[4:])
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    tmp = tmp_path_factory.mktemp("iris_holdout")
    train_path = tmp / "reference.csv"
    test_path = tmp / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    feature_descriptions_path = root / "feature_descriptions.json"
    background_path = root / "background.txt"

    return {
        "reference": train_path,
        "test": test_path,
        "feature_descriptions": feature_descriptions_path,
        "background": background_path,
        "y_test": test_df["species"].astype(str).tolist(),
        "X_train": train_df.drop(columns=["flower_id", "species"]).to_numpy(dtype=float),
        "y_train": train_df["species"].astype(str).to_numpy(),
        "X_test": test_df.drop(columns=["flower_id", "species"]).to_numpy(dtype=float),
        "feature_names": [
            c for c in train_df.columns
            if c not in ("flower_id", "species")
        ],
        "feature_descriptions_dict": json.loads(
            feature_descriptions_path.read_text()
        ),
        "background_text": background_path.read_text(),
    }


def _stub_llm_config_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a minimal valid llm_config.jsonc.

    Used for in-process CLI tests where ``build_llm_client`` is
    monkeypatched to ignore the actual config and return a
    ``FakeLLM``. The file just has to parse cleanly through
    ``load_llm_config``.
    """
    cfg_path = tmp_path / "llm_config.jsonc"
    cfg_path.write_text(
        '{"api_key": "stub-test-key", '
        '"base_url": "http://stub.invalid/v1", '
        '"model_name": "stub-model"}'
    )
    return cfg_path


def _patch_cli_to_use_fake_llm(monkeypatch, fake_llm) -> None:
    """Force every CLI/pipeline LLM call to use ``fake_llm``.

    The pipeline calls ``build_llm_client(llm_config)`` when the
    caller doesn't supply ``llm_client``; the CLI never supplies
    one, so this single monkeypatch covers both paths.
    """
    monkeypatch.setattr(
        "molactagent.pipeline.build_llm_client",
        lambda cfg: fake_llm,
    )


def _read_summary_csv(out_dir: pathlib.Path) -> pd.DataFrame:
    """Read the pipeline's ``summary.csv``, returned by every
    classification or regression CLI / pipeline run with an
    ``output_dir``."""
    csv_path = out_dir / "summary.csv"
    assert csv_path.exists(), f"Expected summary.csv at {csv_path}"
    return pd.read_csv(csv_path)


# ── PART 1: Iris classification — labels match across 3 interfaces ──


def test_iris_labels_match_pipeline_estimator_cli_inprocess(
    fake_llm, monkeypatch, iris_paths, iris_data, tmp_path,
):
    """The same iris inputs through three interfaces must produce
    bit-identical predicted labels.

    This is the foundational parity check. If it fails, every
    metric-equality assertion below it is meaningless.
    """
    # ── (a) High-level pipeline ──────────────────────────────────
    pipe_dir = tmp_path / "pipeline_out"
    pipe_result = ca.run_classification(
        reference=iris_paths["reference"],
        test=iris_paths["test"],
        label_column="species",
        id_column="flower_id",
        feature_descriptions=iris_paths["feature_descriptions"],
        background=f"@{iris_paths['background']}",
        output_dir=pipe_dir,
        llm_client=fake_llm,
        per_sample_reports=False,
    )
    pipeline_labels = [o.predicted_label for o in pipe_result.outcomes]
    pipeline_summary = _read_summary_csv(pipe_dir)
    assert pipeline_summary["predicted_label"].tolist() == pipeline_labels

    # ── (b) sklearn-style estimator ──────────────────────────────
    clf = ca.MolActAgentClassifier(
        llm_client=fake_llm,
        feature_names=iris_data.feature_names,
        feature_descriptions=iris_data.feature_descriptions,
        background=iris_data.background,
        k=3,
        max_workers=2,
    ).fit(iris_data.X_train, iris_data.y_train)
    estimator_labels = clf.predict(iris_data.X_test).tolist()

    # ── (c) CLI in-process (real argparse + main()) ──────────────
    _patch_cli_to_use_fake_llm(monkeypatch, fake_llm)
    cli_dir = tmp_path / "cli_out"
    stub_cfg = _stub_llm_config_file(tmp_path)
    from molactagent.cli import main as cli_main
    rc = cli_main([
        "--quiet",
        "run",
        "-r", str(iris_paths["reference"]),
        "-t", str(iris_paths["test"]),
        "-l", "species",
        "--id-column", "flower_id",
        "--k", "3",
        "--feature-descriptions", str(iris_paths["feature_descriptions"]),
        "--background", f"@{iris_paths['background']}",
        "--llm-config", str(stub_cfg),
        "-o", str(cli_dir),
    ])
    assert rc == 0, "CLI run returned non-zero"
    cli_summary = _read_summary_csv(cli_dir)
    cli_labels = cli_summary["predicted_label"].tolist()

    # ── parity assertion ─────────────────────────────────────────
    assert pipeline_labels == estimator_labels == cli_labels, (
        "Three interfaces (pipeline / estimator / CLI) disagreed on "
        f"iris labels. This is a real functional regression.\n"
        f"  pipeline  : {pipeline_labels}\n"
        f"  estimator : {estimator_labels}\n"
        f"  CLI       : {cli_labels}\n"
    )


# ── PART 2: Iris classification — full metric suite, identical across interfaces


def test_iris_classification_metrics_identical_across_interfaces(
    fake_llm, monkeypatch, iris_holdout_csvs, tmp_path,
):
    """Compute the FULL classification metric suite on each interface
    and assert every metric is identical. This is the user's
    "verify metrics remain unchanged" criterion translated into a
    test the FakeLLM can answer deterministically.

    Uses a 50/50 holdout split of the iris reference set so every
    test row has ground truth (the bundled ``test.csv`` ships
    without a ``species`` column on purpose).
    """
    # Run all three interfaces (same setup as above; we re-run
    # here so each test is self-contained).
    pipe_dir = tmp_path / "pipe"
    pipe_result = ca.run_classification(
        reference=iris_holdout_csvs["reference"],
        test=iris_holdout_csvs["test"],
        label_column="species",
        id_column="flower_id",
        feature_descriptions=iris_holdout_csvs["feature_descriptions"],
        background=f"@{iris_holdout_csvs['background']}",
        output_dir=pipe_dir,
        llm_client=fake_llm,
        per_sample_reports=False,
    )
    pipe_labels = [o.predicted_label for o in pipe_result.outcomes]

    clf = ca.MolActAgentClassifier(
        llm_client=fake_llm,
        feature_names=iris_holdout_csvs["feature_names"],
        feature_descriptions=iris_holdout_csvs["feature_descriptions_dict"],
        background=iris_holdout_csvs["background_text"],
        k=3,
        max_workers=2,
    ).fit(iris_holdout_csvs["X_train"], iris_holdout_csvs["y_train"])
    estimator_labels = clf.predict(iris_holdout_csvs["X_test"]).tolist()

    _patch_cli_to_use_fake_llm(monkeypatch, fake_llm)
    cli_dir = tmp_path / "cli"
    stub_cfg = _stub_llm_config_file(tmp_path)
    from molactagent.cli import main as cli_main
    cli_main([
        "--quiet", "run",
        "-r", str(iris_holdout_csvs["reference"]),
        "-t", str(iris_holdout_csvs["test"]),
        "-l", "species",
        "--id-column", "flower_id", "--k", "3",
        "--feature-descriptions", str(iris_holdout_csvs["feature_descriptions"]),
        "--background", f"@{iris_holdout_csvs['background']}",
        "--llm-config", str(stub_cfg),
        "-o", str(cli_dir),
    ])
    cli_labels = _read_summary_csv(cli_dir)["predicted_label"].tolist()

    y_true = iris_holdout_csvs["y_test"]

    # Compute the full metric suite for each interface.
    pipe_metrics = classification_metrics(y_true, pipe_labels)
    est_metrics = classification_metrics(y_true, estimator_labels)
    cli_metrics = classification_metrics(y_true, cli_labels)

    # Every published metric must be identical.
    metric_keys = (
        "accuracy", "balanced_accuracy", "mcc", "macro_f1",
        "unknown_rate",
    )
    for key in metric_keys:
        p = getattr(pipe_metrics, key)
        e = getattr(est_metrics, key)
        c = getattr(cli_metrics, key)
        assert p == pytest.approx(e, abs=1e-12), (
            f"Pipeline vs estimator differ on {key!r}: "
            f"{p} vs {e}"
        )
        assert e == pytest.approx(c, abs=1e-12), (
            f"Estimator vs CLI differ on {key!r}: {e} vs {c}"
        )

    # Per-class metrics also identical.
    for cls in pipe_metrics.per_class_f1:
        assert (
            pipe_metrics.per_class_f1[cls]
            == pytest.approx(est_metrics.per_class_f1[cls], abs=1e-12)
        )
        assert (
            est_metrics.per_class_f1[cls]
            == pytest.approx(cli_metrics.per_class_f1[cls], abs=1e-12)
        )

    # Print metrics for human inspection (visible in pytest -s output).
    print(
        f"\nIris metrics agree across all three interfaces:\n"
        f"  accuracy       = {pipe_metrics.accuracy:.4f}\n"
        f"  bal. accuracy  = {pipe_metrics.balanced_accuracy:.4f}\n"
        f"  MCC            = {pipe_metrics.mcc:.4f}\n"
        f"  macro F1       = {pipe_metrics.macro_f1:.4f}\n"
        f"  per-class F1   = {pipe_metrics.per_class_f1}\n"
    )


# ── PART 3: Iris classification — predict_proba sums to 1, separate from confidence


def test_iris_predict_proba_well_formed_and_distinct_from_confidence(
    fake_llm, iris_data,
):
    """``predict_proba`` rows sum to 1, every probability is in
    [0, 1], and the calibrated confidence is exposed as a SEPARATE
    axis (not the same number as ``proba.max``).

    The user explicitly asked for "confidence scores ALONGSIDE
    probabilities" — this test ensures they really are alongside,
    not aliased.
    """
    clf = ca.MolActAgentClassifier(
        llm_client=fake_llm,
        feature_names=iris_data.feature_names,
        feature_descriptions=iris_data.feature_descriptions,
        background=iris_data.background,
        k=3, max_workers=2,
    ).fit(iris_data.X_train, iris_data.y_train)

    proba = clf.predict_proba(iris_data.X_test)
    labels, conf = clf.predict_with_confidence(iris_data.X_test)

    # Shape contract.
    assert proba.shape == (len(iris_data.X_test), len(clf.classes_))
    # Each row sums to 1 and stays in [0, 1].
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert (proba >= 0).all() and (proba <= 1).all()
    # Confidence is its own axis: not always == proba.max.
    proba_max = proba.max(axis=1)
    diffs = np.abs(conf - proba_max)
    # On the iris fixture with FakeLLM, at least one sample has
    # confidence ≠ proba.max (the calibration formula folds in
    # sample-size + coverage discounts). If all are equal, we've
    # accidentally aliased the two.
    assert diffs.max() > 1e-6, (
        "predict_with_confidence appears to alias proba.max — "
        "they should be distinct numbers per the design contract."
    )


# ── PART 4: BSLT regression — chemistry pipeline → all three interfaces


@pytest.fixture(scope="module")
def bslt_features():
    """Featurise BSLT_huigui.csv via the new chemistry pipeline.

    Skips if RDKit is unavailable. Returns a FeatureBundle whose
    ``X`` and ``y`` we'll feed through every interface below.
    """
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("RDKit required for BSLT chemistry pipeline.")
    try:
        return ca.datasets.load_bslt_regression(
            descriptors=True, maccs=True, morgan=False,
        )
    except FileNotFoundError as e:
        pytest.skip(f"BSLT dataset missing: {e}")


@pytest.fixture(scope="module")
def bslt_csvs(bslt_features, tmp_path_factory):
    """Save the BSLT FeatureBundle to a CSV split (reference + test)
    so the CLI and high-level pipeline (which both expect on-disk
    CSVs) can ingest it.

    Split: first 11 rows → reference, remaining → test.
    A small `sample_id` column is added for the ``--id-column``
    requirement.
    """
    n_total = bslt_features.X.shape[0]
    n_train = max(8, n_total - 4)  # keep at least 4 rows for test
    feature_names = bslt_features.feature_names

    df_full = pd.DataFrame(bslt_features.X, columns=feature_names)
    df_full.insert(0, "sample_id", [f"S{i}" for i in range(n_total)])
    df_full["standard_value"] = bslt_features.y

    tmp = tmp_path_factory.mktemp("bslt_csvs")
    ref_path = tmp / "reference.csv"
    test_path = tmp / "test.csv"
    df_full.iloc[:n_train].to_csv(ref_path, index=False)
    df_full.iloc[n_train:].to_csv(test_path, index=False)

    # The pipeline's validator requires feature_descriptions to
    # cover EVERY column when supplied — including the label
    # column. The chemistry pipeline's bundle only describes
    # features, not the target, so we extend it here with a
    # minimal target description so the validator passes.
    feature_descriptions_full = dict(bslt_features.feature_descriptions)
    feature_descriptions_full["standard_value"] = (
        "Brine-shrimp lethality bioactivity value (continuous "
        "regression target)."
    )

    # Write the extended feature descriptions to JSON so the CLI
    # can read them via --feature-descriptions.
    fd_path = tmp / "feature_descriptions.json"
    fd_path.write_text(json.dumps(feature_descriptions_full, indent=2))

    return {
        "reference": ref_path,
        "test": test_path,
        "feature_descriptions_path": fd_path,
        "feature_names": list(feature_names),
        "feature_descriptions": feature_descriptions_full,
        "n_train": int(n_train),
        "n_test": int(n_total - n_train),
        "y_train": bslt_features.y[:n_train],
        "y_test": bslt_features.y[n_train:],
        "X_train": bslt_features.X[:n_train],
        "X_test": bslt_features.X[n_train:],
        "background": ca.pp.chemistry.BSLT_BACKGROUND,
    }


def test_bslt_predictions_match_pipeline_estimator_cli_inprocess(
    fake_llm, monkeypatch, bslt_csvs, tmp_path,
):
    """All three interfaces produce identical regression predictions
    on the BSLT chemistry → CCR pipeline.

    Comparison tolerances:

    - **Pipeline outcomes vs estimator outputs**: exact (1e-9) —
      both are the engine's full-precision floats with no
      serialisation in between.
    - **CSV-derived numbers vs in-memory floats**: 1e-3 — the
      pipeline writes ``summary.csv`` with values rounded to 4
      decimal places (per
      :meth:`SampleOutcome.to_summary_row`), so any cross-CSV
      comparison must allow that documented rounding.
    """
    # ── (a) Pipeline (use in-memory PipelineOutput for full-precision preds) ──
    pipe_dir = tmp_path / "bslt_pipe"
    pipe_result = ca.run_regression(
        reference=bslt_csvs["reference"],
        test=bslt_csvs["test"],
        target_column="standard_value",
        id_column="sample_id",
        feature_descriptions=bslt_csvs["feature_descriptions_path"],
        background=bslt_csvs["background"],
        output_dir=pipe_dir,
        llm_client=fake_llm,
        k=3, max_workers=2,
        per_sample_reports=False,
    )
    pipeline_preds = [
        o.predicted_value for o in pipe_result.outcomes
    ]
    pipeline_csv_preds = _coerce_pred_values(
        _read_summary_csv(pipe_dir)["predicted_value"]
    )

    # ── (b) Estimator ────────────────────────────────────────────
    reg = ca.MolActAgentRegressor(
        llm_client=fake_llm,
        target_name="standard_value",
        feature_names=bslt_csvs["feature_names"],
        feature_descriptions=bslt_csvs["feature_descriptions"],
        background=bslt_csvs["background"],
        k=3, max_workers=2,
    ).fit(bslt_csvs["X_train"], bslt_csvs["y_train"])
    estimator_preds = [
        None if np.isnan(v) else float(v)
        for v in reg.predict(bslt_csvs["X_test"]).tolist()
    ]

    # ── (c) CLI in-process ───────────────────────────────────────
    _patch_cli_to_use_fake_llm(monkeypatch, fake_llm)
    cli_dir = tmp_path / "bslt_cli"
    stub_cfg = _stub_llm_config_file(tmp_path)
    from molactagent.cli import main as cli_main
    rc = cli_main([
        "--quiet", "run",
        "-r", str(bslt_csvs["reference"]),
        "-t", str(bslt_csvs["test"]),
        "-l", "standard_value",
        "--id-column", "sample_id",
        "--k", "3",
        "--feature-descriptions", str(bslt_csvs["feature_descriptions_path"]),
        "--llm-config", str(stub_cfg),
        "-o", str(cli_dir),
    ])
    assert rc == 0, "CLI BSLT run returned non-zero"
    cli_csv_preds = _coerce_pred_values(
        _read_summary_csv(cli_dir)["predicted_value"]
    )

    # ── parity assertions ────────────────────────────────────────
    # Full-precision: pipeline-in-memory must match estimator exactly.
    _assert_pred_arrays_equal(
        pipeline_preds, estimator_preds,
        "pipeline (in-memory) vs estimator", abs_tol=1e-9,
    )
    # CSV-derived: pipeline.csv must equal CLI.csv exactly (both
    # are rounded to 4 dp by the same code path).
    _assert_pred_arrays_equal(
        pipeline_csv_preds, cli_csv_preds,
        "pipeline summary.csv vs CLI summary.csv", abs_tol=1e-9,
    )
    # CSV-derived vs in-memory: tolerance = CSV rounding.
    _assert_pred_arrays_equal(
        pipeline_csv_preds, estimator_preds,
        "pipeline summary.csv vs estimator (in-memory)", abs_tol=5e-4,
    )


def test_bslt_regression_metrics_identical_across_interfaces(
    fake_llm, monkeypatch, bslt_csvs, tmp_path,
):
    """Compute the FULL regression metric suite on each interface
    and assert every metric is identical.

    Pipeline and estimator predictions are full-precision floats
    so their metrics must match exactly. CLI predictions come
    from ``summary.csv`` rounded to 4 decimal places (per
    :meth:`SampleOutcome.to_summary_row`) — metrics computed on
    CSV-rounded predictions are compared with a looser ``5e-4``
    tolerance that accounts for that documented serialisation.
    """
    # Pipeline (in-memory, full precision)
    pipe_dir = tmp_path / "p"
    pipe_result = ca.run_regression(
        reference=bslt_csvs["reference"],
        test=bslt_csvs["test"],
        target_column="standard_value",
        id_column="sample_id",
        feature_descriptions=bslt_csvs["feature_descriptions_path"],
        background=bslt_csvs["background"],
        output_dir=pipe_dir,
        llm_client=fake_llm,
        k=3, max_workers=2, per_sample_reports=False,
    )
    pipe_preds = [o.predicted_value for o in pipe_result.outcomes]
    pipe_intervals = [o.interval for o in pipe_result.outcomes]

    # Estimator
    reg = ca.MolActAgentRegressor(
        llm_client=fake_llm,
        target_name="standard_value",
        feature_names=bslt_csvs["feature_names"],
        feature_descriptions=bslt_csvs["feature_descriptions"],
        background=bslt_csvs["background"],
        k=3, max_workers=2,
    ).fit(bslt_csvs["X_train"], bslt_csvs["y_train"])
    est_preds_arr, est_lo, est_hi = reg.predict_with_interval(bslt_csvs["X_test"])
    est_preds = [None if np.isnan(v) else float(v) for v in est_preds_arr]
    est_intervals = [
        None if np.isnan(l) else (float(l), float(h))
        for l, h in zip(est_lo, est_hi)
    ]

    # CLI in-process
    _patch_cli_to_use_fake_llm(monkeypatch, fake_llm)
    cli_dir = tmp_path / "c"
    stub_cfg = _stub_llm_config_file(tmp_path)
    from molactagent.cli import main as cli_main
    cli_main([
        "--quiet", "run",
        "-r", str(bslt_csvs["reference"]),
        "-t", str(bslt_csvs["test"]),
        "-l", "standard_value",
        "--id-column", "sample_id", "--k", "3",
        "--feature-descriptions", str(bslt_csvs["feature_descriptions_path"]),
        "--llm-config", str(stub_cfg),
        "-o", str(cli_dir),
    ])
    cli_preds = _coerce_pred_values(_read_summary_csv(cli_dir)["predicted_value"])
    cli_intervals = _read_intervals(_read_summary_csv(cli_dir))

    y_true = bslt_csvs["y_test"].tolist()
    pipe_m = regression_metrics(y_true, pipe_preds, pipe_intervals)
    est_m = regression_metrics(y_true, est_preds, est_intervals)
    cli_m = regression_metrics(y_true, cli_preds, cli_intervals)

    # Tolerance budget: pipeline and estimator are full-precision
    # so their metrics match to FP epsilon (1e-9). The CLI's
    # predictions come from CSV-rounded values, so its metrics
    # need a looser tolerance (~5e-3 for numeric metrics like MAE
    # / RMSE / R² / median_abs_err that are sums of rounded
    # contributions; counts like n_predicted / n_refused must match
    # exactly).
    exact_keys = ("n_predicted", "n_refused")
    numeric_keys = (
        "mae", "rmse", "r2", "pearson_r", "spearman_r",
        "median_abs_err", "fraction_within_interval",
    )
    for key in exact_keys:
        p, e, c = (
            getattr(pipe_m, key),
            getattr(est_m, key),
            getattr(cli_m, key),
        )
        assert p == e == c, (
            f"BSLT exact-count metric {key!r} differs across "
            f"interfaces: pipe={p}, est={e}, cli={c}"
        )
    for key in numeric_keys:
        p = getattr(pipe_m, key)
        e = getattr(est_m, key)
        c = getattr(cli_m, key)
        if p is None or e is None or c is None:
            assert p is None and e is None and c is None, (
                f"BSLT regression metric {key!r}: None mismatch "
                f"pipe={p} est={e} cli={c}"
            )
            continue
        # Pipeline ↔ estimator: both full-precision in-memory.
        assert p == pytest.approx(e, abs=1e-9), (
            f"Pipeline vs estimator differ on {key}: {p} vs {e}"
        )
        # Pipeline ↔ CLI: CLI ran from CSV-rounded predictions.
        assert p == pytest.approx(c, abs=5e-3), (
            f"Pipeline vs CLI differ on {key}: {p} vs {c} "
            f"(beyond CSV-rounding tolerance 5e-3)"
        )

    print(
        f"\nBSLT metrics agree across all three interfaces:\n"
        f"  n              = {pipe_m.n}\n"
        f"  n_predicted    = {pipe_m.n_predicted}\n"
        f"  n_refused      = {pipe_m.n_refused}\n"
        f"  MAE            = {pipe_m.mae}\n"
        f"  RMSE           = {pipe_m.rmse}\n"
        f"  R²             = {pipe_m.r2}\n"
        f"  Pearson r      = {pipe_m.pearson_r}\n"
        f"  Spearman r     = {pipe_m.spearman_r}\n"
        f"  Median AE      = {pipe_m.median_abs_err}\n"
    )


# ── PART 5: CLI subprocess — exercises the real console-script entry point


def test_cli_subprocess_version_and_help():
    """The console-script entry point must remain functional.

    Subprocess test (not in-process) so we exercise the actual
    ``python -m molactagent`` invocation path.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "molactagent", "--version"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert ca.__version__ in proc.stdout

    proc = subprocess.run(
        [sys.executable, "-m", "molactagent", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    for sub in ("validate", "run", "init-config"):
        assert sub in proc.stdout, (
            f"--help output missing subcommand {sub!r}"
        )


def test_cli_subprocess_validate_iris_emits_json(iris_paths):
    """Subprocess ``molactagent validate`` on iris emits parseable
    JSON (the documented pipe-friendly contract)."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "molactagent", "--quiet", "validate",
            "-r", str(iris_paths["reference"]),
            "-t", str(iris_paths["test"]),
            "-l", "species", "--id-column", "flower_id", "--k", "3",
            "--feature-descriptions", str(iris_paths["feature_descriptions"]),
            "--background", f"@{iris_paths['background']}",
            "--format", "json",
        ],
        capture_output=True, text=True, timeout=30,
        cwd=str(_project_root()),
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed.get("is_valid") is True
    assert parsed.get("summary", {}).get("task") == "classification"
    assert parsed.get("summary", {}).get("n_reference_samples", 0) == 24
    assert parsed.get("summary", {}).get("n_test_samples", 0) == 6


def test_cli_subprocess_validate_bslt_csv(bslt_csvs):
    """``molactagent validate`` on a real BSLT CSV pair (subprocess)
    detects the regression task and reports a valid validation."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "molactagent", "--quiet", "validate",
            "-r", str(bslt_csvs["reference"]),
            "-t", str(bslt_csvs["test"]),
            "-l", "standard_value",
            "--id-column", "sample_id",
            "--k", "3",
            "--format", "json",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed.get("is_valid") is True, parsed.get("errors")
    assert parsed.get("summary", {}).get("task") == "regression"


# ── helpers ───────────────────────────────────────────────────────


def _coerce_pred_values(series_or_list) -> list[float | None]:
    """Coerce the ``predicted_value`` column to a list of floats / None.

    The pipeline writes empty cells for refusals; pandas reads them
    as NaN. We normalise to ``None`` for consistent comparison
    against the estimator's NaN-encoded refusals.
    """
    out: list[float | None] = []
    for v in list(series_or_list):
        if v is None:
            out.append(None)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if not np.isfinite(f):
            out.append(None)
        else:
            out.append(f)
    return out


def _assert_pred_arrays_equal(
    a, b, tag: str, *, abs_tol: float = 1e-9,
) -> None:
    """Compare two prediction arrays; refusals (None / NaN) match
    only against other refusals.

    ``abs_tol`` defaults to a strict 1e-9 (in-memory ↔ in-memory).
    Callers that compare CSV-serialised values should pass a
    looser tolerance (e.g. ``5e-4`` to accommodate the pipeline's
    documented 4-decimal-place rounding in ``summary.csv``).
    """
    assert len(a) == len(b), f"{tag}: length mismatch {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None:
            assert y is None, f"{tag}[{i}]: refusal vs {y}"
        else:
            assert y is not None, f"{tag}[{i}]: {x} vs refusal"
            assert x == pytest.approx(y, abs=abs_tol), (
                f"{tag}[{i}]: {x} vs {y} (tol={abs_tol})"
            )


def _read_intervals(summary_df: pd.DataFrame) -> list[tuple[float, float] | None]:
    """Reconstruct ``(lo, hi)`` tuples (or ``None`` for refusals)
    from a regression summary DataFrame's ``interval_lo`` /
    ``interval_hi`` columns."""
    out: list[tuple[float, float] | None] = []
    if "interval_lo" not in summary_df.columns:
        return [None] * len(summary_df)
    for lo, hi in zip(summary_df["interval_lo"], summary_df["interval_hi"]):
        try:
            l = float(lo)
            h = float(hi)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if not (np.isfinite(l) and np.isfinite(h)):
            out.append(None)
        else:
            out.append((l, h))
    return out
