"""Confidence-vs-error calibration diagnostic + free recomputes for CCR.

Step 1 of the CCR ablation plan: takes an existing CCR LOOCV cache
(produced by ``run_loocv_BSLT_regression.py``) and:

  1. Computes per-phase contribution error per fold = |true target -
     mean of that phase's selected references' targets|.
  2. Reports the rank correlation between per-phase classification
     confidence and per-phase contribution error. A meaningfully
     negative correlation says "low confidence ↔ high error",
     i.e. confidence is calibrated against correctness, so
     confidence-weighted aggregation is likely to help.
  3. Recomputes the LOOCV metrics for two new aggregator variants
     that need per-phase information (and therefore can't be
     evaluated through the existing ``recompute_ccr_with_aggregator``
     path which only sees the deduped flat reference list):

       * ``CCR-conf``          : weight each ref by its source
                                  phase's classification confidence.
       * ``CCR-conf-distance`` : weight = phase_confidence *
                                  1 / (contextual_distance + eps).

     Refs picked by both phases get summed weights from each phase.
     If a phase failed to nominate a winner, only the surviving
     phase contributes refs (graceful degradation).

No new LLM calls. Pure analysis on the cached fold records.

Usage:
    python scripts/ccr_diagnose_and_recompute.py \
        --cache data/loocv_regression_BSLT_huigui_k5_ccr_only_v2.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from run_loocv_metrics import (  # noqa: E402
    compute_regression_metrics,
    format_regression_metrics_table,
)


def _phase_contribution_error(
    phase_neighbours: list[dict],
    y_train: list[float],
    train_to_global_idx: dict[int, int],
    true_value: float,
) -> tuple[float | None, float]:
    """Mean of the phase's selected refs' targets, and |that mean - truth|.

    Returns ``(phase_estimate, abs_err)``. ``phase_estimate`` is
    ``None`` when the phase nominated no neighbours (e.g. its
    classifier was inconclusive and no fallback class was chosen).
    """
    if not phase_neighbours:
        return None, float("nan")
    targets: list[float] = []
    for nb in phase_neighbours:
        ri = nb.get("ref_idx")
        # phase_neighbours store TRAIN-relative indices (because the
        # MolActAgent inside CCR was fit on train only). The
        # _ccr_one_fold runner remaps the dedup'd union to global
        # indices, but the per-phase neighbours stay train-relative.
        if ri is None:
            continue
        # Train-relative index used as-is into y_train.
        if 0 <= int(ri) < len(y_train):
            targets.append(float(y_train[int(ri)]))
    if not targets:
        return None, float("nan")
    est = float(np.mean(targets))
    return est, abs(est - true_value)


def _build_y_train_for_fold(
    fold_idx: int, y_all: list[float],
) -> list[float]:
    """Reconstruct the per-fold training y array used by CCR.

    LOOCV holds out index ``fold_idx`` and trains on the remaining
    samples in their original order. Phase neighbour ``ref_idx``
    fields are indices into this per-fold training array.
    """
    return [float(v) for i, v in enumerate(y_all) if i != fold_idx]


def calibration_diagnostic(records: list[dict], y_all: list[float]) -> dict:
    """Per-phase contribution error vs per-phase confidence.

    Returns a dict carrying the underlying arrays plus Pearson and
    Spearman rank correlations.
    """
    targets_conf = []
    targets_err = []
    feature_conf = []
    feature_err = []
    pooled_conf = []
    pooled_err = []

    print()
    print("=" * 96)
    print("Per-fold per-phase contribution analysis")
    print("=" * 96)
    print(
        f"{'fold':>4}  {'true':>7}  "
        f"{'P1.cls':>10} {'P1.conf':>7} {'P1.est':>8} {'P1.err':>7}  "
        f"{'P2.cls':>10} {'P2.conf':>7} {'P2.est':>8} {'P2.err':>7}"
    )
    print("-" * 96)

    for rec in records:
        if rec is None:
            continue
        fold = int(rec["fold"])
        true_v = float(rec["true"])
        y_train = _build_y_train_for_fold(fold, y_all)

        pt = rec.get("phase_target") or {}
        pf = rec.get("phase_feature") or {}
        pt_nbrs = pt.get("neighbours") or []
        pf_nbrs = pf.get("neighbours") or []
        pt_conf = float(pt.get("confidence", 0.0) or 0.0)
        pf_conf = float(pf.get("confidence", 0.0) or 0.0)
        pt_cls = pt.get("predicted_class", "?") or "?"
        pf_cls = pf.get("predicted_class", "?") or "?"

        pt_est, pt_err = _phase_contribution_error(
            pt_nbrs, y_train, {}, true_v,
        )
        pf_est, pf_err = _phase_contribution_error(
            pf_nbrs, y_train, {}, true_v,
        )

        if pt_est is not None and np.isfinite(pt_err):
            targets_conf.append(pt_conf)
            targets_err.append(pt_err)
            pooled_conf.append(pt_conf)
            pooled_err.append(pt_err)
        if pf_est is not None and np.isfinite(pf_err):
            feature_conf.append(pf_conf)
            feature_err.append(pf_err)
            pooled_conf.append(pf_conf)
            pooled_err.append(pf_err)

        pt_est_str = f"{pt_est:>8.3f}" if pt_est is not None else f"{'—':>8}"
        pt_err_str = f"{pt_err:>7.3f}" if np.isfinite(pt_err) else f"{'—':>7}"
        pf_est_str = f"{pf_est:>8.3f}" if pf_est is not None else f"{'—':>8}"
        pf_err_str = f"{pf_err:>7.3f}" if np.isfinite(pf_err) else f"{'—':>7}"
        print(
            f"{fold:>4d}  {true_v:>7.3f}  "
            f"{pt_cls:>10} {pt_conf:>7.3f} {pt_est_str} {pt_err_str}  "
            f"{pf_cls:>10} {pf_conf:>7.3f} {pf_est_str} {pf_err_str}"
        )

    def _corr(name: str, xs, ys) -> dict | None:
        if len(xs) < 3:
            print(f"  [{name}] n={len(xs)} too small for correlation")
            return None
        xa = np.asarray(xs, dtype=float)
        ya = np.asarray(ys, dtype=float)
        if xa.std() == 0 or ya.std() == 0:
            print(
                f"  [{name}] zero variance in confidence or error; "
                "no correlation possible"
            )
            return None
        pr = pearsonr(xa, ya)
        sr = spearmanr(xa, ya)
        print(
            f"  [{name}] n={len(xs)}  "
            f"Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.3f})  "
            f"Spearman r={sr.statistic:+.3f} (p={sr.pvalue:.3f})"
        )
        return {
            "n": len(xs),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
        }

    print()
    print("=" * 96)
    print("Calibration: does higher confidence predict lower error?")
    print("(Negative r = well-calibrated; r ≈ 0 = no signal)")
    print("=" * 96)
    target_corr = _corr("Phase 1 (target bin)", targets_conf, targets_err)
    feature_corr = _corr("Phase 2 (feat. cluster)", feature_conf, feature_err)
    pooled_corr = _corr("Pooled across phases", pooled_conf, pooled_err)

    return {
        "target_phase": target_corr,
        "feature_phase": feature_corr,
        "pooled": pooled_corr,
        "raw": {
            "target_conf": targets_conf,
            "target_err": targets_err,
            "feature_conf": feature_conf,
            "feature_err": feature_err,
        },
    }


def _aggregate_with_per_phase(
    pt_nbrs: list[dict],
    pf_nbrs: list[dict],
    pt_conf: float,
    pf_conf: float,
    y_train: list[float],
    method: str,
    eps: float = 1e-6,
) -> float | None:
    """Confidence-aware aggregation using per-phase neighbour data."""
    weighted_sum = 0.0
    total_weight = 0.0
    seen_index_weights: dict[int, float] = {}

    def _accumulate(neighbours: list[dict], phase_conf: float):
        for nb in neighbours:
            ri = nb.get("ref_idx")
            if ri is None or not (0 <= int(ri) < len(y_train)):
                continue
            ri_int = int(ri)
            d = float(nb.get("ctx_distance", 0.0) or 0.0)
            if method == "conf":
                w = float(phase_conf)
            elif method == "conf-distance":
                w = float(phase_conf) / (d + eps)
            else:
                raise ValueError(f"Unknown method {method!r}")
            seen_index_weights[ri_int] = (
                seen_index_weights.get(ri_int, 0.0) + w
            )

    if pt_nbrs:
        _accumulate(pt_nbrs, pt_conf)
    if pf_nbrs:
        _accumulate(pf_nbrs, pf_conf)

    for ri, w in seen_index_weights.items():
        weighted_sum += float(y_train[ri]) * w
        total_weight += w

    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def recompute_per_phase_aggregator(
    records: list[dict],
    y_all: list[float],
    method: str,
) -> tuple[dict, list[dict]]:
    new_records = []
    for rec in records:
        if rec is None:
            continue
        fold = int(rec["fold"])
        true_v = float(rec["true"])
        y_train = _build_y_train_for_fold(fold, y_all)

        pt = rec.get("phase_target") or {}
        pf = rec.get("phase_feature") or {}
        pred = _aggregate_with_per_phase(
            pt_nbrs=pt.get("neighbours") or [],
            pf_nbrs=pf.get("neighbours") or [],
            pt_conf=float(pt.get("confidence", 0.0) or 0.0),
            pf_conf=float(pf.get("confidence", 0.0) or 0.0),
            y_train=y_train,
            method=method,
        )
        new_records.append({
            "fold": fold,
            "true": true_v,
            "pred": float(pred) if pred is not None and np.isfinite(pred) else None,
            "aggregator": f"CCR-{method}",
        })

    y_true = [r["true"] for r in new_records]
    y_pred = [r["pred"] for r in new_records]
    metrics = compute_regression_metrics(y_true, y_pred, intervals=None)
    return metrics, new_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "loocv_regression_BSLT_huigui_k5_ccr_only_v2.json"
        ),
        help="Path to a CCR LOOCV cache JSON.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(
            _PROJECT_ROOT / "data" /
            "loocv_regression_BSLT_huigui_ccr_step1.json"
        ),
        help="Where to save the diagnostic + recomputed metrics.",
    )
    args = parser.parse_args()

    cache_path = pathlib.Path(args.cache).resolve()
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    y_all = [float(v) for v in raw.get("y_true") or []]
    records = (
        raw.get("predictions", {}).get("CCR-mean")
        or raw.get("predictions", {}).get("CCR-median")
        or raw.get("predictions", {}).get("CCR-weighted")
    )
    if records is None:
        raise SystemExit(f"No CCR records found in {cache_path}")
    print(
        f"Loaded {len(records)} CCR fold records from {cache_path.name}"
    )
    print(f"y_all has {len(y_all)} target values")

    diag = calibration_diagnostic(records, y_all)

    print()
    print("=" * 96)
    print("Recomputed metrics for confidence-aware aggregators")
    print("(no new LLM calls — recomputed from cached classification phases)")
    print("=" * 96)

    new_metrics: dict[str, dict] = {}
    new_predictions: dict[str, list[dict]] = {}
    for method in ("conf", "conf-distance"):
        m, recs = recompute_per_phase_aggregator(records, y_all, method)
        new_metrics[f"CCR-{method}"] = m
        new_predictions[f"CCR-{method}"] = recs

    # Side-by-side comparison with the original three aggregators.
    base_metrics = raw.get("metrics", {}) or {}
    side = {}
    for k in ("CCR-mean", "CCR-median", "CCR-weighted"):
        if k in base_metrics:
            side[k] = base_metrics[k]
    side.update(new_metrics)

    table = format_regression_metrics_table(side)
    print()
    print(table)

    out_path = pathlib.Path(args.output).resolve()
    out_path.write_text(
        json.dumps(
            {
                "source_cache": str(cache_path),
                "calibration_diagnostic": {
                    "target_phase": diag["target_phase"],
                    "feature_phase": diag["feature_phase"],
                    "pooled": diag["pooled"],
                },
                "metrics_baseline": {
                    k: side.get(k)
                    for k in ("CCR-mean", "CCR-median", "CCR-weighted")
                },
                "metrics_new": new_metrics,
                "predictions_new": new_predictions,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    print()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
