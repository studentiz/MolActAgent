"""Test whether TabPFN's strong penguins LOOCV result reflects
genuine in-context learning vs implicit memorization of
Palmer-Penguins-shaped (features -> body mass) patterns from its
pre-training.

Design (target-permutation test):
  For each random seed in ``--seeds``:
    1. Shuffle ``y`` across rows so (X[i], y_shuffled[i]) carries
       no real signal.
    2. Run LOOCV: train TabPFN on (X[~i], y_shuffled[~i]),
       predict on X[i], record ``pred_i``.
    3. Compute three errors per fold:
         - |pred_i - y_shuffled[i]|  (the in-context "truth"
           TabPFN was just trained on; near zero means it learned
           the shuffled signal — should not happen since there is no
           signal)
         - |pred_i - y[i]|           (the ORIGINAL true value;
           near zero means TabPFN ignored the shuffled training
           data and reproduced the unshuffled relationship — that
           IS the memorization signal)
         - |pred_i - mean(y_train_shuffled)|  (the floor any
           sensible in-context learner falls back to when the
           training data carries no signal)

Decision rule:
  * If MAE(pred, original-y) ≈ MAE(pred, shuffled-y) ≈ std(y)  ->
    no memorization. TabPFN treats the shuffled training set
    as noise and falls back to the mean.
  * If MAE(pred, original-y) is much SMALLER than MAE(pred,
    shuffled-y) AND much smaller than std(y)  -> TabPFN ignores
    the shuffled in-context training and reproduces the
    unshuffled relationship  -> memorization signal.

Caveats: this only probes row-level memorization. It does NOT
test "shape-level adaptation" where TabPFN's prior may be tuned to
small clean tabular datasets like Palmer Penguins via the
authors' development decisions; that would need a comparison
against a synthetic dataset of identical shape but a totally
different relationship.

Usage:
    conda activate molactagent_env
    python scripts/test_tabpfn_memorization_penguins.py
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

from run_loocv_metrics import load_regression_csv  # noqa: E402

logger = logging.getLogger("tabpfn_memorize")


def tabpfn_loocv(
    X: np.ndarray,
    y_train_signal: np.ndarray,
    device: str = "cpu",
    seed: int = 42,
) -> list[float]:
    """LOOCV over ``y_train_signal``: for each fold i, train TabPFN
    on (X[~i], y_train_signal[~i]) and predict on X[i].

    Returns the per-fold predictions in fold-index order.
    """
    from tabpfn import TabPFNRegressor

    n = len(y_train_signal)
    preds: list[float] = []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_tr = X[train_idx]
        y_tr = y_train_signal[train_idx]
        X_te = X[i:i + 1]
        try:
            reg = TabPFNRegressor(
                device=device,
                random_state=seed,
                ignore_pretraining_limits=True,
            )
            reg.fit(X_tr, y_tr)
            pred = float(reg.predict(X_te)[0])
        except Exception as exc:
            logger.error(f"TabPFN fold {i} failed: {exc}")
            pred = float("nan")
        preds.append(pred)
    return preds


def _mae(a: list[float], b: list[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(arr_a - arr_b)))


def _rmse(a: list[float], b: list[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((arr_a - arr_b) ** 2)))


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
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Random seeds for the y-shuffle. Each one runs a full "
             "LOOCV (24 folds) with that permutation.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "tabpfn_memorization_test_penguins.json"
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
    n = len(y)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_baseline_mae = float(np.mean(np.abs(y - y_mean)))
    logger.info(
        f"Loaded {n} samples, target range [{y.min():.4g}, "
        f"{y.max():.4g}], mean={y_mean:.4g}, std={y_std:.4g}; "
        f"mean-prediction MAE on this set = {y_baseline_mae:.4g}"
    )

    # ── Anchor: TabPFN with the REAL training relationship ─────
    logger.info("Anchor run — TabPFN LOOCV on real (X, y)...")
    t0 = time.time()
    real_preds = tabpfn_loocv(X, y, device=args.device, seed=42)
    real_elapsed = time.time() - t0
    real_mae_vs_y = _mae(real_preds, y.tolist())
    real_rmse_vs_y = _rmse(real_preds, y.tolist())
    logger.info(
        f"  Real (X, y) -> MAE={real_mae_vs_y:.3f}, "
        f"RMSE={real_rmse_vs_y:.3f}  ({real_elapsed:.1f}s)"
    )

    # ── Shuffle runs ──────────────────────────────────────────
    rng = np.random.default_rng(0)
    seed_results = []
    for seed in args.seeds:
        # Each random permutation gives a different shuffled-y
        # training signal. We use a separate seed for the
        # permutation (deterministic per --seed) AND pass the same
        # value to TabPFN's random_state so the inner randomness is
        # also reproducible per-run.
        local_rng = np.random.default_rng(seed)
        perm = local_rng.permutation(n)
        y_shuffled = y[perm]

        logger.info(
            f"Shuffle run (seed={seed}) — TabPFN LOOCV on "
            f"(X, shuffle(y))..."
        )
        t0 = time.time()
        sh_preds = tabpfn_loocv(
            X, y_shuffled, device=args.device, seed=seed,
        )
        sh_elapsed = time.time() - t0

        mae_vs_shuffled = _mae(sh_preds, y_shuffled.tolist())
        mae_vs_original = _mae(sh_preds, y.tolist())
        # MAE between TabPFN's predictions and the per-fold mean of
        # the shuffled training y — this is the "fall back to the
        # mean" floor an honest in-context learner should hit when
        # the training signal is destroyed.
        per_fold_means = np.array([
            float(np.mean(np.delete(y_shuffled, i)))
            for i in range(n)
        ])
        mae_vs_per_fold_mean = _mae(sh_preds, per_fold_means.tolist())

        logger.info(
            f"  seed={seed}: "
            f"MAE(pred, shuffled-y)={mae_vs_shuffled:.3f}  "
            f"MAE(pred, ORIGINAL-y)={mae_vs_original:.3f}  "
            f"MAE(pred, per-fold mean)={mae_vs_per_fold_mean:.3f}  "
            f"({sh_elapsed:.1f}s)"
        )

        seed_results.append({
            "seed": seed,
            "perm": perm.tolist(),
            "predictions": sh_preds,
            "y_shuffled": y_shuffled.tolist(),
            "mae_vs_shuffled": mae_vs_shuffled,
            "mae_vs_original": mae_vs_original,
            "mae_vs_per_fold_mean": mae_vs_per_fold_mean,
            "elapsed_s": round(sh_elapsed, 2),
        })

    # ── Summary ────────────────────────────────────────────────
    sh_vs_shuffled = float(np.mean(
        [s["mae_vs_shuffled"] for s in seed_results]
    ))
    sh_vs_original = float(np.mean(
        [s["mae_vs_original"] for s in seed_results]
    ))
    sh_vs_per_fold_mean = float(np.mean(
        [s["mae_vs_per_fold_mean"] for s in seed_results]
    ))

    print()
    print("=" * 72)
    print("TabPFN target-permutation memorization test on penguins")
    print(f"n={n}, target std={y_std:.3f}, mean-only MAE floor={y_baseline_mae:.3f}")
    print("=" * 72)
    print()
    print(f"  Anchor: Real (X, y)              MAE = {real_mae_vs_y:7.3f}")
    print(f"          (this is the 'genuine learning' baseline)")
    print()
    print(f"  Shuffled-y runs (avg over {len(args.seeds)} seeds):")
    print(f"    MAE(pred, shuffled-y)          = {sh_vs_shuffled:7.3f}")
    print(f"    MAE(pred, ORIGINAL-y)          = {sh_vs_original:7.3f}  <-- key number")
    print(f"    MAE(pred, per-fold-mean(y_tr)) = {sh_vs_per_fold_mean:7.3f}")
    print()
    print("Reading the result:")
    print(
        f"  * If MAE(pred, ORIGINAL-y) ≈ {real_mae_vs_y:.0f} (the anchor) AND much"
    )
    print(
        f"    smaller than MAE(pred, shuffled-y) ≈ {y_std:.0f}, that's a"
    )
    print("    STRONG memorization signal — TabPFN ignored the shuffled")
    print("    training data and reproduced the original (X -> y)")
    print("    relationship from somewhere other than the in-context data.")
    print()
    print(
        f"  * If MAE(pred, ORIGINAL-y) ≈ MAE(pred, shuffled-y) ≈ "
        f"{y_baseline_mae:.0f}"
    )
    print(
        f"    ≈ MAE(pred, mean) ≈ {y_baseline_mae:.0f}, that's the predicted-"
    )
    print("    near-mean signature of HONEST in-context learning that")
    print("    correctly identifies no signal in the shuffled data.")
    print()

    out_path = pathlib.Path(args.output).resolve()
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "n_samples": n,
                "target_mean": y_mean,
                "target_std": y_std,
                "mean_baseline_mae": y_baseline_mae,
                "anchor_real_mae": real_mae_vs_y,
                "anchor_real_rmse": real_rmse_vs_y,
                "anchor_real_predictions": real_preds,
                "shuffled_runs": seed_results,
                "shuffled_avg_mae_vs_shuffled": sh_vs_shuffled,
                "shuffled_avg_mae_vs_original": sh_vs_original,
                "shuffled_avg_mae_vs_per_fold_mean": sh_vs_per_fold_mean,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
