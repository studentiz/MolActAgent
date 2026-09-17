"""Comprehensive prior-adaptation test on penguin-shape data:
TabPFN, MolActAgent (CCR-backed regression), and the ML baselines
all evaluated on the same four regimes (real penguins + 3 synthetic
shape-clones) with the full metric suite (MAE, RMSE, R^2, Pearson r,
Spearman r, Median AE, plus the MAE / Linear ratio for cross-
regime comparison).

Adds two gaps the first version of this test missed:
  1. MolActAgent in the evaluation. Without it we cannot tell
     whether the TabPFN-vs-Linear ratio swing across regimes
     reflects something specific to TabPFN's pre-training prior
     or just a general property of "flexible learners on small
     tabular data".
  2. Full metric suite. MAE alone does not reveal whether a
     method's degradation on synthetic regimes shows up the same
     way in rank correlation or tail error.

MolActAgent is run with its deployed CCR configuration
(``ccr_aggregator='conf'``, ``ccr_n_mode='sqrt'``,
``ccr_k_per_class=2``, ``ccr_agent_k=2``) and a deliberately
generic background string so that the LLM cannot fall back on
domain knowledge of real penguin biology when the synthetic
relationship is novel — this isolates the contrastive engine from
any "the LLM happens to know penguins" effect.

Usage:
    conda activate molactagent_env
    python scripts/test_prior_adaptation_full.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from molactagent import MolActAgent, LLMClient  # noqa: E402

from run_demo import load_llm_config  # noqa: E402
from run_loocv_metrics import (  # noqa: E402
    compute_regression_metrics,
    load_regression_csv,
)
from run_tabpfn_penguins_loocv import run_tabpfn_loocv  # noqa: E402

logger = logging.getLogger("prior_test")


GENERIC_BACKGROUND = (
    "Continuous-target regression from three numeric morphometric "
    "features. The (features -> target) relationship in this "
    "reference set is what you should infer from the data; do NOT "
    "rely on any external biological knowledge — the relationship "
    "may not match real-world expectations."
)
GENERIC_FEATURE_DESCRIPTIONS = {
    "bill_length_mm":   "Numeric feature 1 — bill length (mm)",
    "bill_depth_mm":    "Numeric feature 2 — bill depth (mm)",
    "flipper_length_mm": "Numeric feature 3 — flipper length (mm)",
}


def ml_loocv(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """LOOCV for Linear / Ridge / KNN3 / RF on standard-scaled X."""
    models = {
        "Linear": LinearRegression(),
        "Ridge":  Ridge(alpha=1.0),
        "KNN3":   KNeighborsRegressor(n_neighbors=3),
        "RF":     RandomForestRegressor(n_estimators=200, random_state=42),
    }
    n = len(y)
    out: dict[str, list[float]] = {name: [] for name in models}
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        scaler = StandardScaler().fit(X[train_idx])
        X_tr_s = scaler.transform(X[train_idx])
        X_te_s = scaler.transform(X[i:i + 1])
        for name, model in models.items():
            try:
                m = type(model)(**model.get_params())
                m.fit(X_tr_s, y[train_idx])
                pred = float(m.predict(X_te_s)[0])
            except Exception as exc:
                logger.error(f"{name} fold {i} failed: {exc}")
                pred = float("nan")
            out[name].append(pred)
    return {
        name: compute_regression_metrics(y.tolist(), preds, intervals=None)
        for name, preds in out.items()
    }


def molactagent_loocv(
    X: np.ndarray,
    y: np.ndarray,
    llm: LLMClient,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    fold_workers: int = 4,
    max_workers_per_fold: int = 20,
) -> dict:
    """LOOCV with MolActAgent (CCR-backed regression) using its
    deployed configuration. Folds are evaluated in parallel via a
    thread pool of size ``fold_workers``; within each fold the inner
    MolActAgent fans out its tuple LLM calls via ``max_workers``.
    """
    n = len(y)
    preds: list[float] = [float("nan")] * n

    def _one_fold(i: int) -> tuple[int, float]:
        train_idx = [j for j in range(n) if j != i]
        try:
            agent = MolActAgent(
                llm_client=llm,
                feature_names=list(feature_names),
                feature_descriptions=dict(feature_descriptions),
                background=background,
                max_workers=max_workers_per_fold,
                llm_temperature=0.0,
            ).fit(X[train_idx], y[train_idx], task="regression")
            res = agent.predict_one(X[i], test_id=i)
            pv = res.get("predicted_value")
            value = (
                float(pv) if pv is not None and np.isfinite(pv)
                else float("nan")
            )
        except Exception as exc:
            logger.error(f"MolActAgent fold {i} failed: {exc}")
            value = float("nan")
        return i, value

    if fold_workers > 1:
        with ThreadPoolExecutor(max_workers=fold_workers) as pool:
            futures = {pool.submit(_one_fold, i): i for i in range(n)}
            for fut in as_completed(futures):
                idx, val = fut.result()
                preds[idx] = val
    else:
        for i in range(n):
            _, preds[i] = _one_fold(i)
    return compute_regression_metrics(y.tolist(), preds, intervals=None)


def _avg_metrics(metric_dicts: list[dict]) -> dict:
    """Per-key average over a list of metric dicts (skipping Nones)."""
    if not metric_dicts:
        return {}
    keys = ("mae", "rmse", "r2", "pearson_r", "spearman_r",
            "median_abs_err")
    out = {}
    for k in keys:
        vals = [
            float(d[k]) for d in metric_dicts
            if d.get(k) is not None and np.isfinite(d.get(k, 0.0))
        ]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def _format_regime_table(
    regime_label: str,
    metrics_by_method: dict[str, dict],
    method_order: list[str],
) -> str:
    cols = [
        ("MAE",      "mae"),
        ("RMSE",     "rmse"),
        ("R²",       "r2"),
        ("Pearson",  "pearson_r"),
        ("Spearman", "spearman_r"),
        ("MedianAE", "median_abs_err"),
    ]
    name_w = max(len(n) for n in method_order)
    name_w = max(name_w, 12)
    header = (
        f"  {'Method':<{name_w}}  "
        + "  ".join(f"{label:>9}" for label, _ in cols)
        + f"  {'MAE/Lin':>9}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [f"\n  ── {regime_label} ──", header, sep]
    linear_mae = metrics_by_method.get("Linear", {}).get("mae")
    for name in method_order:
        m = metrics_by_method.get(name, {})
        cells = []
        for _, key in cols:
            v = m.get(key)
            if v is None or not np.isfinite(v):
                cells.append(f"{'—':>9}")
            elif key in ("mae", "rmse", "median_abs_err"):
                cells.append(f"{v:>9.2f}")
            else:
                cells.append(f"{v:>9.3f}")
        if linear_mae and m.get("mae") is not None:
            ratio = m["mae"] / linear_mae
            ratio_cell = f"{ratio:>9.3f}"
        else:
            ratio_cell = f"{'—':>9}"
        lines.append(
            f"  {name:<{name_w}}  " + "  ".join(cells) + f"  {ratio_cell}"
        )
    return "\n".join(lines)


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
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2],
        help="Random seeds per synthetic regime (each runs all methods)."
             " Default = 2 to keep MolActAgent LLM cost manageable.",
    )
    parser.add_argument("--llm-profile", type=str, default="deepseek")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "prior_adaptation_test_full.json"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    csv_path = pathlib.Path(args.csv).resolve()
    X, y, feature_names, _ = load_regression_csv(
        csv_path, target_col=args.target_col,
        feature_cols=args.feature_cols, id_col=args.id_col,
    )
    n, p = X.shape
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    logger.info(
        f"Real penguins: n={n}, p={p}, target mean={y_mean:.4g}, "
        f"std={y_std:.4g}"
    )

    llm_cfg = load_llm_config(_PROJECT_ROOT, profile=args.llm_profile)
    logger.info(
        f"LLM profile={args.llm_profile!r} -> "
        f"model={llm_cfg['model_name']!r}"
    )
    llm = LLMClient(**llm_cfg, timeout=120.0)

    # ── 0. Anchor: real penguins ─────────────────────────────────
    logger.info("=== Anchor — real penguins ===")
    t0 = time.time()
    real_ml = ml_loocv(X, y)
    real_tabpfn_metrics, _ = run_tabpfn_loocv(
        X, y, device=args.device, seed=42,
    )
    real_molactagent = molactagent_loocv(
        X, y, llm,
        feature_names=feature_names,
        feature_descriptions=GENERIC_FEATURE_DESCRIPTIONS,
        background=GENERIC_BACKGROUND,
    )
    real_metrics = {
        **real_ml,
        "TabPFN": real_tabpfn_metrics,
        "MolActAgent": real_molactagent,
    }
    logger.info(f"Anchor took {time.time() - t0:.1f}s")

    # Calibrate synthetic noise to real-penguins Linear-LOOCV residual std.
    real_linear_residual_std = float(np.std(
        np.array([
            real_ml["Linear"]["mae"],  # placeholder
        ])
    ))
    # The above is wrong — recompute residual std from per-fold preds.
    # We don't store per-fold preds in the metrics dict, so recompute
    # from a clean Linear LOOCV here.
    n_total = len(y)
    linear_residuals = []
    for i in range(n_total):
        train_idx = [j for j in range(n_total) if j != i]
        scaler = StandardScaler().fit(X[train_idx])
        X_tr_s = scaler.transform(X[train_idx])
        X_te_s = scaler.transform(X[i:i + 1])
        m = LinearRegression().fit(X_tr_s, y[train_idx])
        linear_residuals.append(float(m.predict(X_te_s)[0]) - float(y[i]))
    real_linear_residual_std = float(np.std(linear_residuals))
    logger.info(
        f"Calibrated synthetic noise std = "
        f"{real_linear_residual_std:.3f} (matches Linear LOOCV "
        "residual std on real penguins)"
    )

    feat_means = X.mean(axis=0)
    feat_stds = X.std(axis=0)

    def gen_synthetic_X(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed * 17 + 3)
        return rng.normal(loc=feat_means, scale=feat_stds, size=(n, p))

    def gen_linear_y(X_in: np.ndarray, seed: int) -> np.ndarray:
        Xc = X_in - X_in.mean(axis=0)
        sds = X_in.std(axis=0)
        sds = np.where(sds < 1e-9, 1.0, sds)
        sig = (
            -150.0 * (Xc[:, 0] / sds[0])
            - 100.0 * (Xc[:, 1] / sds[1])
            - 200.0 * (Xc[:, 2] / sds[2])
        )
        sig *= (y_std / max(float(np.std(sig)), 1e-9))
        rng = np.random.default_rng(seed * 23 + 11)
        noise = rng.normal(0.0, real_linear_residual_std, size=n)
        return sig + y_mean + noise

    def gen_nonlinear_y(X_in: np.ndarray, seed: int) -> np.ndarray:
        Xc = X_in - X_in.mean(axis=0)
        sds = X_in.std(axis=0)
        sds = np.where(sds < 1e-9, 1.0, sds)
        z1 = Xc[:, 0] / sds[0]
        z2 = Xc[:, 1] / sds[1]
        z3 = Xc[:, 2] / sds[2]
        sig = (
            -120.0 * z1
            + 80.0 * z2 * (z2 > 0)
            - 200.0 * z3
            - 60.0 * z1 * z3
        )
        sig *= (y_std / max(float(np.std(sig)), 1e-9))
        rng = np.random.default_rng(seed * 29 + 7)
        noise = rng.normal(0.0, real_linear_residual_std, size=n)
        return sig + y_mean + noise

    regimes = [
        ("A: real X, synthetic-linear y", "real", "linear"),
        ("B: synthetic X, synthetic-linear y", "syn", "linear"),
        ("C: synthetic X, mild-nonlinear y", "syn", "nonlinear"),
    ]

    method_order = ["Linear", "Ridge", "KNN3", "RF", "TabPFN", "MolActAgent"]
    regime_summaries = []
    for label, x_kind, y_kind in regimes:
        logger.info(
            f"\n=== {label} (avg over {len(args.seeds)} seeds) ==="
        )
        per_seed_metrics: dict[str, list[dict]] = {n: [] for n in method_order}
        for seed in args.seeds:
            X_use = X if x_kind == "real" else gen_synthetic_X(seed)
            y_use = (
                gen_linear_y(X_use, seed) if y_kind == "linear"
                else gen_nonlinear_y(X_use, seed)
            )
            t_seed = time.time()
            ml_metrics = ml_loocv(X_use, y_use)
            tabpfn_metrics, _ = run_tabpfn_loocv(
                X_use, y_use, device=args.device, seed=seed,
            )
            ca_metrics = molactagent_loocv(
                X_use, y_use, llm,
                feature_names=feature_names,
                feature_descriptions=GENERIC_FEATURE_DESCRIPTIONS,
                background=GENERIC_BACKGROUND,
            )
            seed_metrics = {
                **ml_metrics,
                "TabPFN": tabpfn_metrics,
                "MolActAgent": ca_metrics,
            }
            for name in method_order:
                per_seed_metrics[name].append(seed_metrics[name])
            logger.info(
                f"  seed={seed} ({time.time() - t_seed:.1f}s):  "
                + "  ".join(
                    f"{name}={seed_metrics[name]['mae']:.2f}"
                    for name in method_order
                )
            )

        avg = {name: _avg_metrics(per_seed_metrics[name]) for name in method_order}
        regime_summaries.append({"label": label, "avg": avg})

    # ── Output ────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("Prior-adaptation test — TabPFN, MolActAgent, and ML baselines")
    print(
        f"n={n}, p={p}, target std={y_std:.2f}, "
        f"calibrated synthetic noise std={real_linear_residual_std:.2f}"
    )
    print("=" * 100)

    print(_format_regime_table("Real penguins (anchor)", real_metrics, method_order))
    for rs in regime_summaries:
        print(_format_regime_table(rs["label"], rs["avg"], method_order))

    print()
    print(
        "Reading: 'MAE/Lin' is each method's MAE divided by Linear's "
        "MAE on the same dataset, normalising away the noise floor "
        "so we can compare 'lift over the linear baseline' across "
        "regimes with different absolute MAE scales."
    )
    print(
        "  - Real penguins is the anchor.\n"
        "  - If a method's MAE/Lin ratio shifts sharply between real "
        "and synthetic regimes, that method's lift on real data "
        "depends on dataset-shape properties the synthetic regimes "
        "don't reproduce — i.e. its prior is partly tuned to "
        "penguin-style data.\n"
        "  - If the ratio stays steady, the method's lift transfers, "
        "i.e. the prior is general-purpose."
    )

    out_path = pathlib.Path(args.output).resolve()
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "real_metrics": real_metrics,
                "regimes": regime_summaries,
                "calibrated_noise_std": real_linear_residual_std,
                "y_std": y_std,
                "feature_names": feature_names,
                "method_order": method_order,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved -> {out_path}")
    llm.print_stats()


if __name__ == "__main__":
    main()
