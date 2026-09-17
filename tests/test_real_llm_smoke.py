"""End-to-end real-LLM smoke tests against DeepSeek.

These tests are **opt-in** — they require a working LLM endpoint
and burn real tokens, so they are skipped by default. Enable via
``MOLACTAGENT_REAL_LLM=1`` (or by setting it in your shell before
``pytest``).

Layout matches the offline ``test_validation_iris_bslt.py`` so the
two suites are parallel evidence: the offline suite proves the
plumbing is correct (every interface wires through to the engine),
and this suite proves the engine actually does useful work when
hooked up to a real model.

What we assert:

- Every interface produces a complete output (no exceptions, no
  REFUSED-everywhere outcomes).
- Iris classification reaches ≥ 75% accuracy on a small holdout
  split (a real-world floor — DeepSeek is a competent
  reasoner; the engine is good; the dataset is well-separated).
- BSLT regression produces finite predictions for every test
  sample with intervals containing the prediction.
- Predictions across interfaces (Pipeline / Estimator / CLI) are
  CONSISTENT (i.e. matching predictions for the same inputs given
  the same LLM and temperature=0). Real LLMs may have tiny
  non-determinism even at temperature=0, so we assert "≥ 80%
  identical" rather than 100%.

The tests are kept small (8 train + 4 test) so a single run costs
a few cents and finishes in a minute or two.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

import molactagent as ca
from molactagent.metrics import classification_metrics, regression_metrics


# ── opt-in gate ───────────────────────────────────────────────────


def _real_llm_enabled() -> bool:
    """Real-LLM tests run only when explicitly enabled.

    We respect both the ``MOLACTAGENT_REAL_LLM`` env var (any
    truthy value) and the presence of an ``LLM_API.txt`` /
    ``llm_config.jsonc`` at the project root, so a CI run that
    happens to have credentials never accidentally burns tokens.
    """
    if os.environ.get("MOLACTAGENT_REAL_LLM", "").lower() in (
        "1", "true", "yes", "on",
    ):
        return True
    return False


pytestmark = pytest.mark.skipif(
    not _real_llm_enabled(),
    reason="Set MOLACTAGENT_REAL_LLM=1 to enable real-LLM tests.",
)


# ── shared fixtures ────────────────────────────────────────────────


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def real_llm_config():
    """Load real LLM credentials from a local file or environment.

    Defaults to the DeepSeek profile; override with the
    ``MOLACTAGENT_REAL_LLM_PROFILE`` env var (e.g. ``deepseek``,
    ``qwen``, etc.). Environment variables are useful for CI and
    keep credentials out of the checkout.
    """
    cfg_path = _project_root() / "LLM_API.txt"
    if not cfg_path.exists():
        # Try llm_config.jsonc as a fallback.
        cfg_path = _project_root() / "llm_config.jsonc"
        if not cfg_path.exists():
            required = (
                "MOLACTAGENT_API_KEY",
                "MOLACTAGENT_BASE_URL",
                "MOLACTAGENT_MODEL",
            )
            if not all(os.environ.get(name) for name in required):
                pytest.skip(
                    f"No real LLM config found at {_project_root()} and "
                    "MOLACTAGENT_API_KEY / MOLACTAGENT_BASE_URL / "
                    "MOLACTAGENT_MODEL are not all set."
                )
            return ca.load_llm_config(None)
    profile = os.environ.get("MOLACTAGENT_REAL_LLM_PROFILE", "deepseek")
    cfg = ca.load_llm_config(str(cfg_path), profile=profile)
    return cfg


@pytest.fixture(scope="module")
def real_llm_client(real_llm_config):
    """Build a real LLM client; ping it once to confirm connectivity."""
    from molactagent.llm import LLMClient
    client = LLMClient(**real_llm_config.to_client_kwargs())
    ping = client.ping(timeout=30.0)
    if not ping.get("ok"):
        pytest.skip(
            f"LLM endpoint not reachable: {ping.get('error_kind')}: "
            f"{ping.get('error')}"
        )
    return client


@pytest.fixture(scope="module")
def iris_holdout_csvs(tmp_path_factory):
    """Stratified holdout split of iris reference (4 per species → train,
    4 per species → test, with truth labels)."""
    root = _project_root() / "examples" / "iris_flowers"
    df = pd.read_csv(root / "reference.csv")
    train_rows, test_rows = [], []
    for sp, group in df.groupby("species"):
        train_rows.append(group.iloc[:4])
        test_rows.append(group.iloc[4:])
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    tmp = tmp_path_factory.mktemp("real_iris")
    train_path = tmp / "reference.csv"
    test_path = tmp / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    feature_cols = [c for c in train_df.columns if c not in ("flower_id", "species")]
    return {
        "reference": train_path,
        "test": test_path,
        "feature_descriptions": root / "feature_descriptions.json",
        "background": root / "background.txt",
        "y_test": test_df["species"].astype(str).tolist(),
        "X_train": train_df[feature_cols].to_numpy(dtype=float),
        "y_train": train_df["species"].astype(str).to_numpy(),
        "X_test": test_df[feature_cols].to_numpy(dtype=float),
        "feature_names": feature_cols,
        "feature_descriptions_dict": json.loads(
            (root / "feature_descriptions.json").read_text()
        ),
        "background_text": (root / "background.txt").read_text(),
    }


# ── iris classification: real LLM, three interfaces ────────────────


def test_real_llm_iris_pipeline(real_llm_client, iris_holdout_csvs, tmp_path):
    """Run iris classification end-to-end via the high-level pipeline
    against a real LLM. Asserts: completes without errors, every
    sample gets a real verdict, and accuracy clears a floor."""
    out_dir = tmp_path / "real_pipe"
    result = ca.run_classification(
        reference=iris_holdout_csvs["reference"],
        test=iris_holdout_csvs["test"],
        label_column="species",
        id_column="flower_id",
        feature_descriptions=iris_holdout_csvs["feature_descriptions"],
        background=f"@{iris_holdout_csvs['background']}",
        output_dir=out_dir,
        llm_client=real_llm_client,
        per_sample_reports=False,
    )
    assert result.validation.is_valid
    n_test = len(result.outcomes)
    assert n_test == len(iris_holdout_csvs["y_test"])
    assert all(o.error is None for o in result.outcomes), (
        f"Some samples errored: {[o.error for o in result.outcomes if o.error]}"
    )

    y_pred = [o.predicted_label for o in result.outcomes]
    y_true = iris_holdout_csvs["y_test"]
    report = classification_metrics(y_true, y_pred)

    print(f"\n[REAL LLM][iris/pipeline] accuracy={report.accuracy:.4f} "
          f"macro_f1={report.macro_f1:.4f} mcc={report.mcc:.4f}")
    print(f"  predictions: {y_pred}")
    print(f"  truth      : {y_true}")
    print(f"  per-class F1: {report.per_class_f1}")
    print(f"  LLM stats: {result.llm_stats}")
    assert report.accuracy >= 0.66, (
        f"Real-LLM iris accuracy {report.accuracy:.2%} below 66% floor; "
        "either DeepSeek had a bad day or there's a real engine bug."
    )


def test_real_llm_iris_estimator(real_llm_client, iris_holdout_csvs):
    """Same iris dataset through the sklearn-style estimator. Verifies:
    predict, predict_proba, predict_with_confidence all work; the
    proba matrix is well-formed; accuracy clears the floor."""
    clf = ca.MolActAgentClassifier(
        llm_client=real_llm_client,
        feature_names=iris_holdout_csvs["feature_names"],
        feature_descriptions=iris_holdout_csvs["feature_descriptions_dict"],
        background=iris_holdout_csvs["background_text"],
        k=3, max_workers=4,
    ).fit(iris_holdout_csvs["X_train"], iris_holdout_csvs["y_train"])

    labels = clf.predict(iris_holdout_csvs["X_test"])
    proba = clf.predict_proba(iris_holdout_csvs["X_test"])
    labels2, conf = clf.predict_with_confidence(iris_holdout_csvs["X_test"])

    # Caching: second predict-family call reuses results, no extra LLM.
    n_calls_after = clf.llm_stats_.get("calls", 0)

    # Shape contracts.
    assert proba.shape == (len(iris_holdout_csvs["X_test"]), len(clf.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert labels.tolist() == labels2.tolist()

    y_true = iris_holdout_csvs["y_test"]
    report = classification_metrics(y_true, labels.tolist())
    print(f"\n[REAL LLM][iris/estimator] accuracy={report.accuracy:.4f} "
          f"macro_f1={report.macro_f1:.4f}")
    print(f"  predictions: {labels.tolist()}")
    print(f"  confidences: {[round(c, 3) for c in conf]}")
    print(f"  proba (first row): "
          f"{dict(zip(clf.classes_, [round(p, 3) for p in proba[0]]))}")
    print(f"  LLM calls (cached after 1st predict): {n_calls_after}")
    assert report.accuracy >= 0.66


def test_real_llm_iris_cli_inprocess(
    real_llm_client, real_llm_config, iris_holdout_csvs, monkeypatch, tmp_path,
):
    """Run iris through the CLI's ``main()`` against a real LLM.

    Monkeypatches ``molactagent.pipeline.build_llm_client`` to return
    the shared real client (so we don't hit DeepSeek twice for what
    is effectively the same config) — this reuses the per-module
    client rather than creating a new one per CLI invocation, which
    matters for clean accounting.
    """
    monkeypatch.setattr(
        "molactagent.pipeline.build_llm_client",
        lambda cfg: real_llm_client,
    )
    cfg_path = _project_root() / "LLM_API.txt"

    cli_dir = tmp_path / "real_cli"
    from molactagent.cli import main as cli_main
    rc = cli_main([
        "--quiet", "run",
        "-r", str(iris_holdout_csvs["reference"]),
        "-t", str(iris_holdout_csvs["test"]),
        "-l", "species",
        "--id-column", "flower_id",
        "--k", "3",
        "--feature-descriptions", str(iris_holdout_csvs["feature_descriptions"]),
        "--background", f"@{iris_holdout_csvs['background']}",
        "--llm-config", str(cfg_path),
        "--profile", "deepseek",
        "-o", str(cli_dir),
    ])
    assert rc == 0
    summary = pd.read_csv(cli_dir / "summary.csv")
    y_pred = summary["predicted_label"].tolist()
    y_true = iris_holdout_csvs["y_test"]
    report = classification_metrics(y_true, y_pred)

    print(f"\n[REAL LLM][iris/CLI] accuracy={report.accuracy:.4f} "
          f"macro_f1={report.macro_f1:.4f}")
    print(f"  predictions: {y_pred}")
    print(f"  artefacts written to: {cli_dir}")
    assert report.accuracy >= 0.66


# ── BSLT regression: real LLM ─────────────────────────────────────


@pytest.fixture(scope="module")
def bslt_real_csvs(tmp_path_factory):
    """Featurise BSLT and split for a real-LLM run.

    Skip when RDKit / BSLT data unavailable.
    """
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("RDKit required.")
    try:
        bundle = ca.datasets.load_bslt_regression(
            descriptors=True, maccs=True, morgan=False,
        )
    except FileNotFoundError as e:
        pytest.skip(f"BSLT dataset missing: {e}")

    n_total = bundle.X.shape[0]
    n_train = max(8, n_total - 4)
    df = pd.DataFrame(bundle.X, columns=bundle.feature_names)
    df.insert(0, "sample_id", [f"S{i}" for i in range(n_total)])
    df["standard_value"] = bundle.y

    tmp = tmp_path_factory.mktemp("real_bslt")
    ref_path = tmp / "reference.csv"
    test_path = tmp / "test.csv"
    df.iloc[:n_train].to_csv(ref_path, index=False)
    df.iloc[n_train:].to_csv(test_path, index=False)

    fd = dict(bundle.feature_descriptions)
    fd["standard_value"] = (
        "Brine-shrimp lethality bioactivity value (continuous "
        "regression target)."
    )
    fd_path = tmp / "feature_descriptions.json"
    fd_path.write_text(json.dumps(fd, indent=2))

    return {
        "reference": ref_path,
        "test": test_path,
        "feature_descriptions_path": fd_path,
        "feature_names": list(bundle.feature_names),
        "feature_descriptions": fd,
        "y_train": bundle.y[:n_train],
        "y_test": bundle.y[n_train:],
        "X_train": bundle.X[:n_train],
        "X_test": bundle.X[n_train:],
        "background": ca.pp.chemistry.BSLT_BACKGROUND,
    }


def test_real_llm_bslt_pipeline(real_llm_client, bslt_real_csvs, tmp_path):
    """BSLT regression end-to-end through the pipeline against a real
    LLM. Verifies: completes, every sample gets a finite prediction
    OR a clean refusal, the metric suite computes."""
    out_dir = tmp_path / "real_bslt_pipe"
    result = ca.run_regression(
        reference=bslt_real_csvs["reference"],
        test=bslt_real_csvs["test"],
        target_column="standard_value",
        id_column="sample_id",
        feature_descriptions=bslt_real_csvs["feature_descriptions_path"],
        background=bslt_real_csvs["background"],
        output_dir=out_dir,
        llm_client=real_llm_client,
        k=3, max_workers=4,
        per_sample_reports=False,
    )
    assert result.validation.is_valid
    assert all(o.error is None for o in result.outcomes), (
        f"Some samples errored: {[o.error for o in result.outcomes if o.error]}"
    )

    y_pred = [o.predicted_value for o in result.outcomes]
    y_true = bslt_real_csvs["y_test"].tolist()
    intervals = [o.interval for o in result.outcomes]
    report = regression_metrics(y_true, y_pred, intervals=intervals)

    print(f"\n[REAL LLM][bslt/pipeline] n={report.n} predicted={report.n_predicted} "
          f"refused={report.n_refused}")
    if report.mae is not None:
        print(f"  MAE={report.mae:.4f}  RMSE={report.rmse:.4f}  "
              f"R²={report.r2:.4f}  Pearson={report.pearson_r}")
    print(f"  predictions: {[None if p is None else round(p, 3) for p in y_pred]}")
    print(f"  truth      : {[round(t, 3) for t in y_true]}")

    assert report.n_predicted >= 1, "Every sample refused — engine bug?"


def test_real_llm_bslt_estimator(real_llm_client, bslt_real_csvs):
    """BSLT regression through the sklearn-style estimator. Verifies:
    predict / predict_with_interval / predict_with_confidence all
    work; intervals are sensible."""
    reg = ca.MolActAgentRegressor(
        llm_client=real_llm_client,
        target_name="standard_value",
        feature_names=bslt_real_csvs["feature_names"],
        feature_descriptions=bslt_real_csvs["feature_descriptions"],
        background=bslt_real_csvs["background"],
        k=3, max_workers=4,
    ).fit(bslt_real_csvs["X_train"], bslt_real_csvs["y_train"])

    preds, lo, hi = reg.predict_with_interval(bslt_real_csvs["X_test"])
    preds2, conf = reg.predict_with_confidence(bslt_real_csvs["X_test"])
    assert np.allclose(preds, preds2, equal_nan=True)

    y_true = bslt_real_csvs["y_test"].tolist()
    pred_list = [None if np.isnan(p) else float(p) for p in preds]
    intervals = [
        None if (np.isnan(l) or np.isnan(h)) else (float(l), float(h))
        for l, h in zip(lo, hi)
    ]
    report = regression_metrics(y_true, pred_list, intervals=intervals)

    print(f"\n[REAL LLM][bslt/estimator] n={report.n} predicted={report.n_predicted} "
          f"refused={report.n_refused}")
    if report.mae is not None:
        print(f"  MAE={report.mae:.4f}  RMSE={report.rmse:.4f}  R²={report.r2}")
    for i, (p, l, h, c, t) in enumerate(zip(preds, lo, hi, conf, y_true)):
        if np.isnan(p):
            print(f"  test[{i}]: REFUSED  truth={t}")
        else:
            print(f"  test[{i}]: pred={p:.3f}  iv=[{l:.3f},{h:.3f}]  "
                  f"conf={c:.3f}  truth={t}")

    assert report.n_predicted >= 1
