"""TabPFN regressor LOOCV on the bundled penguins dataset, with an
inline comparison against every existing method in the historical
``loocv_regression_reference_k5_round*`` runs.

Usage:
    conda activate molactagent_env
    python scripts/run_tabpfn_penguins_loocv.py

Outputs:
    data/loocv_regression_reference_k5_tabpfn.json   — TabPFN metrics + per-fold preds
    data/loocv_regression_reference_k5_tabpfn.txt    — pretty fixed-width comparison

No LLM calls. Reuses ``compute_regression_metrics`` and
``format_regression_metrics_table`` from ``scripts.run_loocv_metrics``
so the numbers line up with the existing baselines.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from run_loocv_metrics import (  # noqa: E402
    compute_regression_metrics,
    format_regression_metrics_table,
    load_regression_csv,
)

logger = logging.getLogger("tabpfn_loocv")


def run_tabpfn_loocv(
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[dict, list[dict]]:
    """LOOCV with TabPFNRegressor.

    Re-fits the regressor on every fold's n-1 training rows. TabPFN
    is in-context (a forward pass over the training rows at predict
    time), so "fitting" is just storing the data. Predict time
    dominates.

    Returns ``(metrics, fold_records)`` where each fold record carries
    ``{fold, true, pred, elapsed_s}``.
    """
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as e:
        raise RuntimeError(
            "TabPFN is required: pip install tabpfn"
        ) from e

    n = len(y)
    fold_records: list[dict] = []
    completed = 0
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_te = X[i:i + 1]
        try:
            reg = TabPFNRegressor(
                device=device,
                random_state=seed,
                ignore_pretraining_limits=True,
            )
            reg.fit(X_tr, y_tr)
            t0 = time.time()
            pred = float(reg.predict(X_te)[0])
            elapsed = time.time() - t0
        except Exception as exc:
            logger.error(f"TabPFN fold {i} failed: {exc}")
            pred = float("nan")
            elapsed = float("nan")
        fold_records.append({
            "fold": int(i),
            "true": float(y[i]),
            "pred": pred if np.isfinite(pred) else None,
            "elapsed_s": round(elapsed, 2) if np.isfinite(elapsed) else None,
        })
        completed += 1
        logger.info(
            f"  fold {i+1}/{n} done in {elapsed:.2f}s  "
            f"true={y[i]:.4g}  pred={pred:.4g}  "
            f"({completed}/{n} folds)"
        )

    y_true = [r["true"] for r in fold_records]
    y_pred = [r["pred"] for r in fold_records]
    metrics = compute_regression_metrics(y_true, y_pred, intervals=None)
    return metrics, fold_records


def load_baseline_table_for_penguins() -> dict[str, dict]:
    """Load the historical penguins LOOCV metrics from the most
    recent MolActAgent run AND from the previous round so we have a
    full comparison surface for the TabPFN result.
    """
    baseline_metrics: dict[str, dict] = {}
    runs = [
        # Source: each file's "metrics" dict. Picks ML baselines from
        # the freshest run (round6) since they're deterministic and
        # identical across runs; picks MolActAgent from each LLM-driven
        # run separately for the round-by-round comparison.
        (
            "round4 (legacy MolActAgent regression)",
            "data/loocv_regression_reference_k5_round4.json",
            ["MolActAgent"],
        ),
        (
            "round5 (CCR, N=floor(sqrt(n)))",
            "data/loocv_regression_reference_k5_round5_ccr.json",
            ["MolActAgent"],
        ),
        (
            "round6 (CCR, N=floor(sqrt(n))+1 — current default)",
            "data/loocv_regression_reference_k5_round6_sqrtN_plus1.json",
            None,  # take everything (ML baselines + MolActAgent)
        ),
    ]
    for tag, path, methods in runs:
        full_path = (_PROJECT_ROOT / path).resolve()
        if not full_path.exists():
            logger.warning(f"baseline file missing: {full_path}")
            continue
        blob = json.loads(full_path.read_text(encoding="utf-8"))
        m = blob.get("metrics", {})
        if methods is None:
            for k, v in m.items():
                # Only take the rows we don't already have.
                if k not in baseline_metrics and k != "Linear":
                    # Drop Linear — its R^2 = -288 is from an
                    # over-fit on a 23-train, 3-feature problem and
                    # would dominate the table's RMSE column.
                    baseline_metrics[k] = v
        else:
            for k in methods:
                if k in m:
                    label = f"{k} ({tag.split(' ')[0]})"
                    baseline_metrics[label] = m[k]
    return baseline_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=str,
        default=str(
            _PROJECT_ROOT / "examples" / "penguins" / "reference.csv"
        ),
    )
    parser.add_argument("--target-col", type=str, default="body_mass_g")
    parser.add_argument(
        "--feature-cols", nargs="+",
        default=["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
    )
    parser.add_argument("--id-col", type=str, default="sample_id")
    parser.add_argument("--device", type=str, default="cpu",
                        help="TabPFN device: 'cpu' or 'cuda' (default cpu).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "loocv_regression_reference_k5_tabpfn.json"
        ),
    )
    parser.add_argument(
        "--output-table", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "loocv_regression_reference_k5_tabpfn.txt"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    csv_path = pathlib.Path(args.csv).resolve()
    X, y, feature_names, sample_ids = load_regression_csv(
        csv_path, target_col=args.target_col,
        feature_cols=args.feature_cols, id_col=args.id_col,
    )
    n = len(y)
    logger.info(
        f"Loaded {n} regression samples ({X.shape[1]} features) "
        f"from {csv_path.name}; target range "
        f"[{y.min():.4g}, {y.max():.4g}]"
    )

    logger.info(f"Running TabPFN LOOCV on device={args.device!r}...")
    t0 = time.time()
    tabpfn_metrics, tabpfn_records = run_tabpfn_loocv(
        X, y, device=args.device, seed=args.seed,
    )
    elapsed = time.time() - t0
    logger.info(f"TabPFN LOOCV finished in {elapsed:.1f}s")

    baseline_metrics = load_baseline_table_for_penguins()

    all_metrics: dict[str, dict] = {}
    all_metrics.update(baseline_metrics)
    all_metrics["TabPFN"] = tabpfn_metrics

    # Sort by MAE for the comparison table.
    sorted_methods = sorted(
        all_metrics.keys(), key=lambda k: all_metrics[k].get("mae") or float("inf"),
    )
    sorted_metrics = {k: all_metrics[k] for k in sorted_methods}

    table = format_regression_metrics_table(sorted_metrics)
    print()
    print(table)
    print()

    out_table = pathlib.Path(args.output_table).resolve()
    out_table.write_text(table + "\n", encoding="utf-8")

    out_json = pathlib.Path(args.output).resolve()
    out_json.write_text(
        json.dumps(
            {
                "args": vars(args),
                "task": "regression",
                "n_samples": n,
                "target_range": [float(y.min()), float(y.max())],
                "metrics": all_metrics,
                "tabpfn_predictions": tabpfn_records,
                "tabpfn_total_elapsed_s": round(elapsed, 2),
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved metrics  -> {out_json}")
    logger.info(f"Saved table    -> {out_table}")


if __name__ == "__main__":
    main()
