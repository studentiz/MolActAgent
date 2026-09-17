"""
Ablation study on antiTB_evidence_table.csv:
  1) no_background      — no dataset description, real feature names + descriptions
  2) no_semantic         — no background, no feature descriptions, AND anonymous
                           feature names (f0, f1, ...) so the LLM cannot infer
                           domain knowledge from feature names alone

ML baselines are run once (unaffected by prompt-level information).

Usage:
  python scripts/run_ablation_antiTB.py --llm-profile deepseek
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

from molactagent import MolActAgent, LLMClient, setup_logging  # noqa: E402
from prepare_features import build_dataset_from_dataframe  # noqa: E402
from run_demo import compute_metrics, load_llm_config, print_metrics_table  # noqa: E402
from run_loocv_evidence import (  # noqa: E402
    BACKGROUND_ANTITB,
    run_molactagent_loocv,
    run_ml_loocv,
)

logger = logging.getLogger("ablation_antiTB")

ABLATION_CONFIGS = {
    "no_background": {
        "label": "CA (no background)",
        "background": None,
        "drop_feature_desc": False,
        "anonymize_features": False,
    },
    "no_semantic": {
        "label": "CA (no semantic)",
        "background": None,
        "drop_feature_desc": True,
        "anonymize_features": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Ablation LOOCV on antiTB")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--use-scaled-values", action="store_true")
    parser.add_argument(
        "--llm-profile", type=str, default="deepseek",
        help="LLM provider profile (deepseek or qwen)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Output JSON path (default: data/ablation_antiTB_results.json)",
    )
    args = parser.parse_args()

    log_files = setup_logging(log_dir=_PROJECT_ROOT / "logs")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log:  {log_files['full']}")

    csv_path = _PROJECT_ROOT / "data" / "antiTB_evidence_table.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    X, y, meta = build_dataset_from_dataframe(df, BACKGROUND_ANTITB, verbose=True)
    feature_names = meta["feature_names"]
    feature_descriptions = meta["feature_descriptions"]

    logger.info(
        f"Dataset: {len(y)} samples  Pos={int((y=='Pos').sum())}  "
        f"Neg={int((y=='Neg').sum())}  Features={X.shape[1]}"
    )

    all_metrics: dict[str, dict] = {}

    logger.info("\n" + "=" * 60 + "\n  ML baselines (LOOCV)\n" + "=" * 60)
    t0 = time.time()
    ml_metrics = run_ml_loocv(X, y)
    all_metrics.update(ml_metrics)
    logger.info(f"ML LOOCV finished in {time.time() - t0:.1f}s")

    llm_cfg = load_llm_config(_PROJECT_ROOT, profile=args.llm_profile)
    logger.info(
        f"LLM profile={args.llm_profile!r} -> model={llm_cfg['model_name']!r} "
        f"base_url={llm_cfg['base_url']!r}"
    )
    llm = LLMClient(**llm_cfg, timeout=120.0)

    anon_feature_names = [f"f{i}" for i in range(len(feature_names))]

    ablation_details: dict[str, list[dict]] = {}

    for abl_name, cfg in ABLATION_CONFIGS.items():
        bg = cfg["background"]
        fd = None if cfg["drop_feature_desc"] else dict(feature_descriptions)
        fn = anon_feature_names if cfg["anonymize_features"] else list(feature_names)

        logger.info(
            f"\n{'#' * 60}\n"
            f"  ABLATION: {cfg['label']}\n"
            f"  background={'<None>' if bg is None else bg[:60] + '...'}\n"
            f"  feature_desc={'<None>' if fd is None else f'{len(fd)} entries'}\n"
            f"  feature_names={'anonymous (f0..f{len(fn)-1})' if cfg['anonymize_features'] else 'real'}\n"
            f"{'#' * 60}"
        )

        t0 = time.time()
        ca_metrics, ca_details, ca_preds = run_molactagent_loocv(
            llm, X, y,
            feature_names=fn,
            feature_descriptions=fd or {},
            background=bg or "",
            k=args.k,
            metric=args.metric,
            max_workers=args.max_workers,
            use_scaled_for_prompt=args.use_scaled_values,
        )
        elapsed = time.time() - t0

        all_metrics[cfg["label"]] = ca_metrics
        ablation_details[abl_name] = ca_details

        logger.info(
            f"\n  {cfg['label']} done in {elapsed:.1f}s  "
            f"bal_acc={ca_metrics['balanced_accuracy']:.3f}  "
            f"mcc={ca_metrics['mcc']:.3f}\n"
        )

        logger.info("Per-fold predictions:")
        for i, (pred, true_label) in enumerate(zip(ca_preds, y)):
            mark = "OK" if pred == true_label else "MISS"
            logger.info(f"  [{mark}] fold {i}: true={true_label} pred={pred}")

        llm.print_stats()

    print_metrics_table(all_metrics)

    out_path = (
        pathlib.Path(args.output_json).resolve()
        if args.output_json
        else _PROJECT_ROOT / "data" / "ablation_antiTB_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "dataset": str(csv_path),
                "n_samples": int(len(y)),
                "metrics": all_metrics,
                "ablation_details": ablation_details,
                "llm_stats": llm.stats,
            },
            f, indent=2, ensure_ascii=False, default=str,
        )
    logger.info(f"Results saved to {out_path}")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log:  {log_files['full']}")


if __name__ == "__main__":
    main()
