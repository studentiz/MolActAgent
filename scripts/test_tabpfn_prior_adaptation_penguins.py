"""Test whether TabPFN's penguins win reflects general in-context
learning OR a prior implicitly tuned to small, clean, morphometric
-> mass-style tabular benchmarks.

Design:
  Hand TabPFN a dataset of the SAME SHAPE as penguins (n=24 samples
  x 3 features, comparable feature scales, comparable noise level)
  but with a KNOWN, DIFFERENT (X -> y) relationship. Compare its
  MAE to the same models' MAE on real penguins.

Three "shape clones" of penguins:

  A. Real X, synthetic y (linear with non-penguin coefficients):
     Uses the actual penguin X rows so the feature joint
     distribution is identical to real penguins. The y values come
     from a known linear relationship with deliberately non-penguin
     coefficients (e.g. flipper-length contributes negatively),
     plus noise calibrated to match the unexplained-variance
     fraction Linear regression hits on real penguins.

  B. Synthetic X (matched marginals), synthetic y (linear with
     non-penguin coefficients): Samples fresh X rows from per-feature
     normal distributions matched to penguin moments, then generates
     y from the same non-penguin coefficients + matched noise. This
     removes any X-distribution memorization on top of B.

  C. Synthetic X (matched marginals), synthetic y (mild nonlinear):
     Same fresh X but a piecewise-linear-with-interaction y. Tests
     whether TabPFN handles a non-penguin functional form as well
     as the linear one.

Baselines on each synthetic dataset: Linear, Ridge, KNN3, RF.
Linear is the upper bound on the linear synthetic datasets (A, B);
RF / KNN3 should track at the level they hit on real penguins.

Reading:
  * If TabPFN's MAE on synthetic ≈ its MAE on real penguins (when
    normalised by Linear's MAE on each, which sets the achievable
    signal-to-noise floor) -> general-purpose in-context learner.
  * If TabPFN's MAE on synthetic is dramatically WORSE than its
    MAE on real penguins (relative to Linear's MAE on each), while
    Linear / KNN / RF transfer normally -> the prior is shaped to
    penguin-style data.

Usage:
    conda activate molactagent_env
    python scripts/test_tabpfn_prior_adaptation_penguins.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from run_loocv_metrics import (  # noqa: E402
    compute_regression_metrics,
    load_regression_csv,
)
from run_tabpfn_penguins_loocv import run_tabpfn_loocv  # noqa: E402

logger = logging.getLogger("tabpfn_prior_test")


def ml_loocv(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Standard ML-baseline LOOCV — Linear / Ridge / KNN3 / RF —
    so we have models with known generalisation behaviour to
    compare TabPFN against."""
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
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_te = X[i:i + 1]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)
        for name, model in models.items():
            try:
                m = type(model)(**model.get_params())
                m.fit(X_tr_s, y_tr)
                pred = float(m.predict(X_te_s)[0])
            except Exception as exc:
                logger.error(f"{name} fold {i} failed: {exc}")
                pred = float("nan")
            out[name].append(pred)
    metrics = {}
    for name, preds in out.items():
        m = compute_regression_metrics(y.tolist(), preds, intervals=None)
        m["predictions"] = [float(p) for p in preds]
        metrics[name] = m
    return metrics


def _summarise(metrics: dict, name: str) -> str:
    return (
        f"  {name:<12}: MAE={metrics['mae']:7.2f}  "
        f"RMSE={metrics['rmse']:7.2f}  "
        f"R^2={metrics['r2']:+7.3f}  "
        f"Pearson={metrics['pearson_r']:+.3f}"
    )


def _pretty_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    name_w = max(len(r["name"]) for r in rows)
    name_w = max(name_w, 12)
    header = f"  {'Method':<{name_w}}  " + "  ".join(
        f"{label:>8}" for label, _ in cols
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [header, sep]
    for r in rows:
        cells = []
        for label, key in cols:
            v = r.get(key)
            if v is None:
                cells.append(f"{'—':>8}")
            elif isinstance(v, float) and not np.isfinite(v):
                cells.append(f"{'—':>8}")
            else:
                cells.append(f"{v:>8.3f}")
        lines.append(f"  {r['name']:<{name_w}}  " + "  ".join(cells))
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3],
        help="Random seeds for each synthetic regime (each runs a "
             "full LOOCV on TabPFN + 4 ML baselines).",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "tabpfn_prior_test_penguins.json"
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

    # ── 0. Anchor: real penguins ─────────────────────────────────
    logger.info("Anchor — real penguins:")
    t0 = time.time()
    real_tabpfn_metrics, real_tabpfn_records = run_tabpfn_loocv(
        X, y, device=args.device, seed=42,
    )
    real_ml_metrics = ml_loocv(X, y)
    logger.info(_summarise(real_tabpfn_metrics, "TabPFN"))
    for name in ("Linear", "Ridge", "KNN3", "RF"):
        logger.info(_summarise(real_ml_metrics[name], name))
    logger.info(f"  (anchor took {time.time() - t0:.1f}s)")

    # Calibrate synthetic noise to match real penguins' Linear-MAE
    # signal-to-noise. Real penguins Linear LOOCV MAE / std ratio
    # gives us how much variance is residual.
    real_linear_residual_std = float(
        np.std(np.asarray(real_ml_metrics["Linear"]["predictions"]) - y)
    )
    logger.info(
        f"Calibrated noise std for synthetic regimes = "
        f"{real_linear_residual_std:.3f} (matches real-penguins "
        "Linear LOOCV residual std)"
    )

    # ── Synthetic dataset generators ─────────────────────────────
    # Pick coefficients that DELIBERATELY differ from the directions
    # any real penguins LinearRegression would learn:
    #   - bill_length: NEGATIVE slope (real penguins is positive)
    #   - bill_depth: NEGATIVE slope (real penguins is roughly zero/
    #     mildly negative)
    #   - flipper_length: NEGATIVE slope (real penguins is strongly
    #     positive — this is the biggest sign-flip)
    # Plus an offset so the synthetic y has a similar magnitude to
    # real body mass.
    X_centered = X - X.mean(axis=0)
    feat_stds = X.std(axis=0)
    # Choose beta so the noise-free signal has std comparable to y_std.
    raw_signal = (
        -150.0 * (X_centered[:, 0] / feat_stds[0])
        - 100.0 * (X_centered[:, 1] / feat_stds[1])
        - 200.0 * (X_centered[:, 2] / feat_stds[2])
    )
    # Scale so the noise-free signal has std == y_std (matches the
    # spread of real penguins targets).
    raw_signal *= (y_std / max(float(np.std(raw_signal)), 1e-9))
    syn_signal_realX = raw_signal + y_mean

    feat_means = X.mean(axis=0)

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
        # Piecewise-linear with one interaction: the "step" at
        # bill_depth = mean introduces a small nonlinearity that no
        # plain Linear model can fit perfectly.
        Xc = X_in - X_in.mean(axis=0)
        sds = X_in.std(axis=0)
        sds = np.where(sds < 1e-9, 1.0, sds)
        z1 = Xc[:, 0] / sds[0]
        z2 = Xc[:, 1] / sds[1]
        z3 = Xc[:, 2] / sds[2]
        sig = (
            -120.0 * z1
            + 80.0 * z2 * (z2 > 0)            # half-rectified
            - 200.0 * z3
            - 60.0 * z1 * z3                  # interaction
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

    # ── Run all synthetic regimes ─────────────────────────────────
    regime_summaries = []
    for label, x_kind, y_kind in regimes:
        logger.info(
            f"\n=== {label} (avg over {len(args.seeds)} seeds) ==="
        )
        per_seed = {
            "TabPFN": [],
            "Linear": [], "Ridge": [], "KNN3": [], "RF": [],
        }
        for seed in args.seeds:
            X_use = X if x_kind == "real" else gen_synthetic_X(seed)
            y_use = (
                gen_linear_y(X_use, seed) if y_kind == "linear"
                else gen_nonlinear_y(X_use, seed)
            )
            t0 = time.time()
            tabpfn_metrics, _tabpfn_records = run_tabpfn_loocv(
                X_use, y_use, device=args.device, seed=seed,
            )
            ml_metrics = ml_loocv(X_use, y_use)
            elapsed = time.time() - t0

            per_seed["TabPFN"].append(tabpfn_metrics)
            for name in ("Linear", "Ridge", "KNN3", "RF"):
                per_seed[name].append(ml_metrics[name])

            logger.info(
                f"  seed={seed} ({elapsed:.1f}s):  "
                + "  ".join(
                    f"{name}={metrics_list[-1]['mae']:.2f}"
                    for name, metrics_list in per_seed.items()
                )
            )

        avg_summary = {}
        for name, metrics_list in per_seed.items():
            avg_summary[name] = {
                "mae": float(np.mean([m["mae"] for m in metrics_list])),
                "rmse": float(np.mean([m["rmse"] for m in metrics_list])),
                "r2": float(np.mean([
                    m["r2"] for m in metrics_list if m["r2"] is not None
                ])) if any(m["r2"] is not None for m in metrics_list)
                else None,
                "pearson_r": float(np.mean([
                    m["pearson_r"] for m in metrics_list if m["pearson_r"] is not None
                ])) if any(m["pearson_r"] is not None for m in metrics_list)
                else None,
                "spearman_r": float(np.mean([
                    m["spearman_r"] for m in metrics_list
                    if m["spearman_r"] is not None
                ])) if any(m["spearman_r"] is not None for m in metrics_list)
                else None,
            }
        regime_summaries.append({"label": label, "avg": avg_summary})

    # ── Headline comparison: how does TabPFN's "lift over Linear"
    #    transfer from real penguins to each synthetic regime? ─────
    real_summary = {
        "TabPFN": {
            "mae": real_tabpfn_metrics["mae"],
            "rmse": real_tabpfn_metrics["rmse"],
            "r2": real_tabpfn_metrics.get("r2"),
            "pearson_r": real_tabpfn_metrics.get("pearson_r"),
            "spearman_r": real_tabpfn_metrics.get("spearman_r"),
        },
        **{
            name: {
                "mae": real_ml_metrics[name]["mae"],
                "rmse": real_ml_metrics[name]["rmse"],
                "r2": real_ml_metrics[name].get("r2"),
                "pearson_r": real_ml_metrics[name].get("pearson_r"),
                "spearman_r": real_ml_metrics[name].get("spearman_r"),
            }
            for name in ("Linear", "Ridge", "KNN3", "RF")
        },
    }

    print()
    print("=" * 96)
    print("TabPFN prior-adaptation test on penguins")
    print(
        f"n={n}, p={p}, target std={y_std:.2f}, "
        f"calibrated synthetic noise std={real_linear_residual_std:.2f}"
    )
    print("=" * 96)

    def _row(name: str, mae: float, ratio: float | None) -> dict:
        return {"name": name, "mae": mae, "ratio_vs_linear": ratio}

    for regime_name, summary_dict in [
        ("Real penguins (anchor)", real_summary),
        *[(s["label"], s["avg"]) for s in regime_summaries],
    ]:
        print()
        print(f"  {regime_name}")
        linear_mae = summary_dict["Linear"]["mae"]
        rows = []
        for name in ("Linear", "Ridge", "KNN3", "RF", "TabPFN"):
            mae = summary_dict[name]["mae"]
            ratio = (mae / linear_mae) if linear_mae > 0 else None
            rows.append(_row(name, mae, ratio))
        print(_pretty_table(
            rows, [("MAE", "mae"), ("MAE/Linear", "ratio_vs_linear")],
        ))

    print()
    print("How to read the comparison:")
    print(
        "  The 'MAE/Linear' column normalises each model's MAE by"
    )
    print(
        "  Linear regression's MAE on the same dataset, so we can"
    )
    print(
        "  compare 'lift over the linear baseline' across datasets"
    )
    print(
        "  with different noise floors. If TabPFN's MAE/Linear ratio"
    )
    print(
        "  on synthetic regimes ≈ its ratio on real penguins, the"
    )
    print(
        "  prior is general-purpose. If TabPFN's ratio jumps sharply"
    )
    print(
        "  on synthetic while ML baselines transfer flat, the prior"
    )
    print(
        "  is shaped to penguin-style data."
    )

    out_path = pathlib.Path(args.output).resolve()
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "real_summary": real_summary,
                "regimes": regime_summaries,
                "calibrated_noise_std": real_linear_residual_std,
                "y_std": y_std,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
