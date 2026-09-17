"""TabPFN regressor LOOCV on BSLT_huigui.csv, with an inline
comparison against every existing method in the historical
``loocv_regression_BSLT_huigui_*`` runs.

BSLT is the brine-shrimp lethality regression benchmark (n=16
cyclic-peptide / polyketide compounds, target = continuous
``standard_value``). Features are built from SMILES via the same
RDKit-descriptor + MACCS-fingerprint pipeline used by every other
BSLT run, so the comparison is apples-to-apples.

Usage:
    conda activate molactagent_env
    python scripts/run_tabpfn_BSLT_loocv.py

Outputs:
    data/loocv_regression_BSLT_huigui_tabpfn.json  — TabPFN metrics + per-fold preds
    data/loocv_regression_BSLT_huigui_tabpfn.txt   — pretty fixed-width comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from run_loocv_BSLT_regression import (  # noqa: E402
    build_regression_dataset_from_csv,
)
from run_loocv_metrics import (  # noqa: E402
    compute_regression_metrics,
    format_regression_metrics_table,
)
from run_tabpfn_penguins_loocv import run_tabpfn_loocv  # noqa: E402

logger = logging.getLogger("tabpfn_loocv_bslt")


def load_baseline_table_for_bslt() -> dict[str, dict]:
    """Pull every existing BSLT method (ML baselines + PureLLM +
    MolActAgent regression + CCR variants at N=4 hierarchical and at
    N=4 Leiden) into a single comparison surface.
    """
    baseline_metrics: dict[str, dict] = {}
    runs = [
        # File path, list of method names to take (None = take all
        # except dropped methods listed below).
        (
            "data/loocv_regression_BSLT_huigui_k5_full_v1.json",
            None,
        ),
        (
            "data/loocv_regression_BSLT_huigui_k5_step2_sqrt_hier.json",
            ["CCR-mean", "CCR-conf"],
        ),
        (
            "data/loocv_regression_BSLT_huigui_k5_step3_sqrt_leiden.json",
            ["CCR-conf-leiden"],
        ),
    ]
    drop = {"Linear"}  # huge negative R^2 dominates the table
    for path, methods in runs:
        full_path = (_PROJECT_ROOT / path).resolve()
        if not full_path.exists():
            logger.warning(f"baseline file missing: {full_path}")
            continue
        blob = json.loads(full_path.read_text(encoding="utf-8"))
        m = blob.get("metrics", {})
        if methods is None:
            for k, v in m.items():
                if k in baseline_metrics or k in drop:
                    continue
                baseline_metrics[k] = v
        else:
            for k in methods:
                if k in m and k not in baseline_metrics:
                    baseline_metrics[k] = m[k]
    return baseline_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=str,
        default=str(_PROJECT_ROOT / "data" / "BSLT_huigui.csv"),
    )
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="standard_value")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "loocv_regression_BSLT_huigui_tabpfn.json"
        ),
    )
    parser.add_argument(
        "--output-table", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "loocv_regression_BSLT_huigui_tabpfn.txt"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    csv_path = pathlib.Path(args.csv).resolve()
    X, y, meta = build_regression_dataset_from_csv(
        csv_path,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        verbose=True,
    )
    n = len(y)
    logger.info(
        f"BSLT: {n} compounds, {X.shape[1]} features "
        f"(target range [{y.min():.3g}, {y.max():.3g}])"
    )

    logger.info(
        f"Running TabPFN LOOCV on device={args.device!r} "
        f"({n} folds)..."
    )
    t0 = time.time()
    tabpfn_metrics, tabpfn_records = run_tabpfn_loocv(
        X, y, device=args.device, seed=args.seed,
    )
    elapsed = time.time() - t0
    logger.info(f"TabPFN LOOCV finished in {elapsed:.1f}s")

    baseline_metrics = load_baseline_table_for_bslt()

    all_metrics: dict[str, dict] = {}
    all_metrics.update(baseline_metrics)
    all_metrics["TabPFN"] = tabpfn_metrics

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
                "n_features": int(X.shape[1]),
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
