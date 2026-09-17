"""
MolActAgent evaluation with scaffold split, ML baselines, and pure-LLM baseline.

Usage:
    python scripts/run_demo.py                        # default: scaffold split, 3 test samples
    python scripts/run_demo.py --max-samples 0        # all test samples from scaffold split
    python scripts/run_demo.py --k 3 --use-reflection  # ablation: enable multi-round reflection

NOTE — only ``load_llm_config`` (used as a helper by other scripts)
has been kept usable; the standalone ``main()`` path is NOT yet
migrated to the multi-class label contract.

The scaffold-split, ML baselines, pure-LLM baseline, and
prediction-loop code in this file all hard-code the legacy
``"Pos"`` / ``"Neg"`` label strings. After the multi-class refactor
``prepare_features.build_dataset_from_dataframe`` returns ``y`` with
the original user labels (e.g. ``"active"`` / ``"inactive"``), so
running this script as-is would silently misclassify every sample.

For binary LOOCV evaluation use the migrated
``scripts/run_loocv_metrics.py`` (the validation path documented in
``data/refactor_validation.md``).
"""

import argparse
import json
import logging
import pathlib
import re
import sys


def _refuse_to_run_unmigrated_main() -> None:
    print(
        "\nERROR: scripts/run_demo.py main() has not been migrated to "
        "the multi-class label contract. The script still assumes "
        "``y == 'Pos'`` / ``y == 'Neg'`` throughout, but "
        "``prepare_features.build_dataset_from_dataframe`` now returns "
        "the original user labels (e.g. ``'active'`` / ``'inactive'``).\n\n"
        "Use scripts/run_loocv_metrics.py instead for binary LOOCV "
        "evaluation:\n\n"
        "  python scripts/run_loocv_metrics.py "
        "--csv data/antiTB_evidence_table.csv --k 3 --llm-profile deepseek\n\n"
        "(``load_llm_config`` from this file is still safe to import "
        "as a helper.)\n",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    _refuse_to_run_unmigrated_main()
import time

import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from molactagent import MolActAgent, LLMClient, setup_logging

logger = logging.getLogger("demo")


# ── data loading ─────────────────────────────────────────────────────

def load_data(project_root: pathlib.Path):
    data_dir = project_root / "data" / "processed"
    data = np.load(data_dir / "features.npz", allow_pickle=True)
    X, y = data["X"], data["y"]
    with open(data_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return X, y, meta


def _parse_llm_api_chunk(chunk: str) -> dict:
    cfg: dict[str, str] = {}
    for line in chunk.splitlines():
        line = line.strip()
        if not line or "：" not in line:
            continue
        key, val = line.split("：", 1)
        cfg[key.strip()] = val.strip()
    return cfg


def load_llm_config(project_root: pathlib.Path, profile: str | None = None) -> dict:
    """Load OpenAI-compatible API settings from ``LLM_API.txt``.

    The file may contain several blocks separated by blank lines (e.g. DeepSeek
    then DashScope). Duplicate keys in one block overwrite as before.

    ``profile``:
      - ``None`` / ``"last"``: use the **last** valid block (backward compatible).
      - ``"deepseek"`` / ``"ds"``: first block whose URL or model name mentions deepseek.
      - ``"qwen"`` / ``"dashscope"``: first block whose URL or model mentions qwen/dashscope.
    """
    path = project_root / "LLM_API.txt"
    text = path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text.strip()) if c.strip()]
    profiles: list[dict] = []
    for ch in chunks:
        d = _parse_llm_api_chunk(ch)
        if all(k in d for k in ("apiKey", "base_url", "Modelname")):
            profiles.append(d)
    if not profiles:
        d = _parse_llm_api_chunk(text)
        if not all(k in d for k in ("apiKey", "base_url", "Modelname")):
            raise ValueError("LLM_API.txt must contain apiKey, base_url, Modelname")
        profiles = [d]

    p = (profile or "last").strip().lower()
    if p in ("last", "legacy", "auto"):
        sel = profiles[-1]
    elif p in ("deepseek", "ds"):
        sel = next(
            (
                x for x in profiles
                if "deepseek" in x.get("base_url", "").lower()
                or "deepseek" in x.get("Modelname", "").lower()
            ),
            profiles[0],
        )
    elif p in ("qwen", "dashscope"):
        sel = next(
            (
                x for x in profiles
                if "dashscope" in x.get("base_url", "").lower()
                or "qwen" in x.get("Modelname", "").lower()
            ),
            profiles[-1],
        )
    else:
        raise ValueError(
            f"Unknown llm profile {profile!r}; use last, deepseek, or qwen."
        )

    return {
        "api_key": sel["apiKey"],
        "base_url": sel["base_url"],
        "model_name": sel["Modelname"],
    }


# ── scaffold split ───────────────────────────────────────────────────

def scaffold_split(y, scaffold_ids, test_ratio=0.3, seed=42):
    """Split data by scaffold: entire scaffolds go to train or test."""
    rng = np.random.RandomState(seed)

    scaffold_groups = defaultdict(list)
    for idx, sid in enumerate(scaffold_ids):
        scaffold_groups[sid].append(idx)

    # sort scaffolds by size (smaller scaffolds more likely to test)
    scaffold_order = sorted(scaffold_groups.keys(),
                            key=lambda s: len(scaffold_groups[s]))
    rng.shuffle(scaffold_order)

    n_total = len(y)
    n_test_target = int(n_total * test_ratio)

    test_indices = []
    train_indices = []
    for sid in scaffold_order:
        members = scaffold_groups[sid]
        if len(test_indices) < n_test_target:
            test_indices.extend(members)
        else:
            train_indices.extend(members)

    # ensure both sets have at least one Pos and one Neg
    y_train = y[train_indices]
    y_test = y[test_indices]
    if "Pos" not in y_train or "Neg" not in y_train:
        logger.warning("Train set missing a class — falling back to random split")
        return _random_split(y, test_ratio, seed)
    if "Pos" not in y_test or "Neg" not in y_test:
        logger.warning("Test set missing a class — falling back to random split")
        return _random_split(y, test_ratio, seed)

    return np.array(train_indices), np.array(test_indices)


def _random_split(y, test_ratio, seed):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    n_test = max(2, int(len(y) * test_ratio))
    return indices[n_test:], indices[:n_test]


# ── metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "Pos" and p == "Pos")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "Neg" and p == "Neg")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "Neg" and p == "Pos")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "Pos" and p == "Neg")
    n_unk = sum(1 for p in y_pred if p == "Unknown")
    n = len(y_true)

    accuracy = (tp + tn) / n if n else 0.0

    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0.0

    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec_pos + spec) / 2.0
    macro_f1 = (f1_pos + f1_neg) / 2.0

    # MCC
    denom = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "macro_f1": macro_f1,
        "precision_pos": prec_pos,
        "recall_pos": rec_pos,
        "f1_pos": f1_pos,
        "precision_neg": prec_neg,
        "recall_neg": rec_neg,
        "f1_neg": f1_neg,
        "specificity_neg": spec,
        "unknown_rate": n_unk / n if n else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "unknown": n_unk,
    }


def print_metrics_table(results: dict[str, dict]):
    """Print a comparison table of metrics for all methods."""
    methods = list(results.keys())
    metric_keys = [
        "accuracy", "balanced_accuracy", "mcc", "macro_f1",
        "precision_pos", "recall_pos", "f1_pos",
        "precision_neg", "recall_neg", "f1_neg",
        "specificity_neg", "unknown_rate",
    ]
    labels = [
        "Accuracy", "Balanced Acc", "MCC", "Macro F1",
        "Precision(Pos)", "Recall(Pos)", "F1(Pos)",
        "Precision(Neg)", "Recall(Neg)", "F1(Neg)",
        "Specificity(Neg)", "Unknown Rate",
    ]

    col_w = max(12, max(len(m) for m in methods) + 2)
    label_w = 20
    header = f"{'Metric':<{label_w}s}" + "".join(f"{m:>{col_w}s}" for m in methods)
    sep = "-" * len(header)

    print(f"\n{sep}\n  COMPARISON TABLE\n{sep}")
    print(header)
    print(sep)
    for key, label in zip(metric_keys, labels):
        row = f"{label:<{label_w}s}"
        for m in methods:
            val = results[m].get(key, 0.0)
            row += f"{val:>{col_w}.3f}"
        print(row)
    print(sep)

    for m in methods:
        r = results[m]
        print(f"  {m}: TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']} Unk={r.get('unknown',0)}")


# ── ML baselines ─────────────────────────────────────────────────────

def run_ml_baselines(X_train, y_train, X_test, y_test):
    """Run RF, SVM, LR baselines with balanced class weights."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    y_tr_binary = np.array([1 if l == "Pos" else 0 for l in y_train])
    pos_count = y_tr_binary.sum()
    neg_count = len(y_tr_binary) - pos_count
    xgb_scale = neg_count / pos_count if pos_count > 0 else 1.0

    classifiers = {
        "KNN1": KNeighborsClassifier(n_neighbors=1),
        "KNN3": KNeighborsClassifier(n_neighbors=3),
        "KNN5": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "LR": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "GBC": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGB": XGBClassifier(
            n_estimators=100, scale_pos_weight=xgb_scale,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }

    label_map = {"Pos": 1, "Neg": 0}
    inv_map = {1: "Pos", 0: "Neg"}
    needs_numeric = {"XGB"}

    results = {}
    for name, clf in classifiers.items():
        t0 = time.time()
        if name in needs_numeric:
            y_tr_enc = np.array([label_map[l] for l in y_train])
            clf.fit(X_tr, y_tr_enc)
            preds_raw = clf.predict(X_te)
            preds = [inv_map[int(p)] for p in preds_raw]
        else:
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te).tolist()
        elapsed = time.time() - t0
        metrics = compute_metrics(y_test.tolist(), preds)
        results[name] = metrics
        logger.info(
            f"[ML {name}] {elapsed:.2f}s  bal_acc={metrics['balanced_accuracy']:.3f}  "
            f"f1={metrics['f1_pos']:.3f}"
        )
    return results


# ── pure LLM baseline ───────────────────────────────────────────────

def run_pure_llm_baseline(
    llm, X_train, y_train, X_test, y_test,
    feature_names, feature_descriptions, background,
    max_samples=None,
):
    """Classify each test sample with a single LLM call (no contrastive triplets)."""
    test_indices = list(range(len(y_test)))
    if max_samples and max_samples < len(test_indices):
        test_indices = test_indices[:max_samples]

    ref_summary_lines = []
    for i in range(min(len(y_train), 10)):
        label = y_train[i]
        vals = ", ".join(
            f"{feature_names[j]}={X_train[i][j]:.2f}"
            for j in range(min(10, len(feature_names)))
        )
        ref_summary_lines.append(f"  [{label}] {vals} ...")
    ref_summary = "\n".join(ref_summary_lines)

    y_pred = []
    for idx in test_indices:
        test_vals = "\n".join(
            f"  {feature_names[j]}: {X_test[idx][j]:.4f}"
            for j in range(len(feature_names))
        )
        prompt = (
            f"You are a binary classifier. Background:\n{background}\n\n"
            f"Reference data (first 10 samples, showing first 10 features):\n"
            f"{ref_summary}\n\n"
            f"Total reference: {sum(1 for l in y_train if l=='Pos')} Pos, "
            f"{sum(1 for l in y_train if l=='Neg')} Neg\n\n"
            f"Test sample features:\n{test_vals}\n\n"
            f"Classify as exactly Pos or Neg. Reply with ONLY the label."
        )
        try:
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                tag=f"pureLLM_{idx}",
            )
            label = "Pos" if "pos" in resp.lower() else (
                "Neg" if "neg" in resp.lower() else "Unknown"
            )
        except Exception as e:
            logger.error(f"[PureLLM] sample {idx} failed: {e}")
            label = "Unknown"
        y_pred.append(label)

    y_true_sub = [y_test[i] for i in test_indices]
    return compute_metrics(y_true_sub, y_pred)


# ── MolActAgent evaluation ────────────────────────────────────────────

def run_molactagent(
    llm, X_train, y_train, X_test, y_test,
    feature_names, feature_descriptions, background,
    args,
    max_samples=None,
):
    n_test = len(y_test)
    if max_samples and max_samples < n_test:
        n_test = max_samples

    agent = MolActAgent(
        llm_client=llm,
        k=args.k,
        metric=args.metric,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        background=background,
        use_reflection=args.use_reflection,
        use_scaled_for_prompt=args.use_scaled_values,
        max_workers=args.max_workers,
    )
    agent.fit(X_train, y_train)

    y_pred = []
    all_results = []
    for i in range(n_test):
        logger.info(
            f"\n{'='*60}\n"
            f"MolActAgent sample {i+1}/{n_test}  true={y_test[i]}\n"
            f"{'='*60}"
        )
        t0 = time.time()
        res = agent.predict_one(X_test[i], test_id=i)
        elapsed = time.time() - t0
        pred = res.get("final_label", "Unknown")
        y_pred.append(pred)
        all_results.append(res)

        n_tri = len(res.get("triple_results", []))
        logger.info(
            f"  => pred={pred}  true={y_test[i]}  conf={res.get('confidence',0):.2f}  "
            f"triplets={n_tri}  time={elapsed:.1f}s"
        )

    y_true_sub = y_test[:n_test].tolist()
    metrics = compute_metrics(y_true_sub, y_pred)
    return metrics, all_results, y_pred


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MolActAgent evaluation")
    parser.add_argument("--max-samples", type=int, default=3,
                        help="Max test samples for LLM methods (0=all)")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--use-reflection", action="store_true",
                        help="Enable multi-round reflection (default: single-round)")
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--use-scaled-values", action="store_true",
                        help="Use scaled feature values in LLM prompts (default: raw)")
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Only run ML baselines (no LLM calls)")
    parser.add_argument("--test-indices", type=str, default=None,
                        help="Comma-separated global sample indices to use as test set "
                             "(e.g., '0,1,4,15,23,36'). Remaining samples become train set.")
    parser.add_argument("--train-pos-limit", type=int, default=0,
                        help="Limit Pos samples in training set to simulate imbalance (0=no limit)")
    parser.add_argument("--train-indices", type=str, default=None,
                        help="Comma-separated global sample indices to use as train set "
                             "(must be used together with --test-indices)")
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).resolve().parent.parent

    log_files = setup_logging(log_dir=project_root / "logs")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log:  {log_files['full']}")
    logger.info(f"Args: {vars(args)}")

    X, y, meta = load_data(project_root)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Pos: {(y == 'Pos').sum()}, Neg: {(y == 'Neg').sum()}")

    # ── split ──────────────────────────────────────────────────────
    if args.test_indices and args.train_indices:
        test_idx = np.array([int(x) for x in args.test_indices.split(",")])
        train_idx = np.array([int(x) for x in args.train_indices.split(",")])
        logger.info(f"Manual split (both): train={len(train_idx)} test={len(test_idx)}")
    elif args.test_indices:
        test_idx = np.array([int(x) for x in args.test_indices.split(",")])
        train_idx = np.array([i for i in range(len(y)) if i not in set(test_idx)])
        logger.info(f"Manual split: train={len(train_idx)} test={len(test_idx)}")
    elif meta.get("scaffold_ids"):
        train_idx, test_idx = scaffold_split(
            y, meta["scaffold_ids"], test_ratio=args.test_ratio
        )
        logger.info(
            f"Scaffold split: train={len(train_idx)} test={len(test_idx)}"
        )
    else:
        logger.warning("No scaffold info, using random split")
        train_idx, test_idx = _random_split(y, args.test_ratio, 42)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if args.train_pos_limit and args.train_pos_limit > 0:
        pos_mask = y_train == "Pos"
        neg_mask = y_train == "Neg"
        pos_indices = np.where(pos_mask)[0]
        rng = np.random.RandomState(42)
        if len(pos_indices) > args.train_pos_limit:
            keep_pos = rng.choice(pos_indices, args.train_pos_limit, replace=False)
            keep = np.sort(np.concatenate([keep_pos, np.where(neg_mask)[0]]))
            X_train, y_train = X_train[keep], y_train[keep]
            logger.info(
                f"Imbalance simulation: kept {args.train_pos_limit} Pos out of "
                f"{pos_mask.sum()} (ratio Neg:Pos = "
                f"{neg_mask.sum()}:{args.train_pos_limit} ≈ "
                f"{neg_mask.sum()/args.train_pos_limit:.0f}:1)"
            )

    logger.info(
        f"Train: {len(y_train)} (Pos={sum(y_train=='Pos')}, Neg={sum(y_train=='Neg')})"
    )
    logger.info(
        f"Test:  {len(y_test)} (Pos={sum(y_test=='Pos')}, Neg={sum(y_test=='Neg')})"
    )

    all_metrics = {}

    # ── ML baselines (always run, instant) ───────────────────────
    logger.info("\n" + "=" * 60 + "\n  ML BASELINES\n" + "=" * 60)
    ml_results = run_ml_baselines(X_train, y_train, X_test, y_test)
    all_metrics.update(ml_results)

    if args.skip_llm:
        print_metrics_table(all_metrics)
        return

    # ── LLM methods ──────────────────────────────────────────────
    llm_cfg = load_llm_config(project_root)
    llm = LLMClient(**llm_cfg, timeout=120.0)

    if args.test_indices:
        max_llm = len(y_test)
    else:
        max_llm = args.max_samples if args.max_samples > 0 else len(y_test)

    # ── Pure LLM baseline ────────────────────────────────────────
    logger.info("\n" + "=" * 60 + "\n  PURE LLM BASELINE\n" + "=" * 60)
    t0 = time.time()
    pure_metrics = run_pure_llm_baseline(
        llm, X_train, y_train, X_test, y_test,
        meta["feature_names"], meta["feature_descriptions"], meta["background"],
        max_samples=max_llm,
    )
    logger.info(f"Pure LLM done in {time.time()-t0:.1f}s")
    all_metrics["PureLLM"] = pure_metrics

    # ── MolActAgent ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60 + "\n  MOLACTAGENT\n" + "=" * 60)
    t0 = time.time()
    ma_metrics, ma_results, ma_preds = run_molactagent(
        llm, X_train, y_train, X_test, y_test,
        meta["feature_names"], meta["feature_descriptions"], meta["background"],
        args, max_samples=max_llm,
    )
    logger.info(f"MolActAgent done in {time.time()-t0:.1f}s")
    all_metrics["MolActAgent"] = ma_metrics

    # ── LLM stats ────────────────────────────────────────────────
    llm.print_stats()

    # ── comparison ───────────────────────────────────────────────
    print_metrics_table(all_metrics)

    # per-sample detail for LLM methods
    print(f"\nMolActAgent per-sample detail (n={len(ma_preds)}):")
    for i, (pred, res) in enumerate(zip(ma_preds, ma_results)):
        true = y_test[i]
        mark = "✓" if true == pred else ("?" if pred == "Unknown" else "✗")
        print(
            f"  [{mark}] true={true:<3s}  pred={pred:<7s}  "
            f"conf={res.get('confidence', 0):.2f}  "
            f"triplets={len(res.get('triple_results', []))}"
        )

    # ── save results ─────────────────────────────────────────────
    out_path = project_root / "data" / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": {
                    "method": "scaffold",
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                },
                "metrics": all_metrics,
                "molactagent_predictions": ma_results,
                "llm_stats": llm.stats,
                "llm_call_log": llm.call_log,
            },
            f, indent=2, ensure_ascii=False, default=str,
        )
    logger.info(f"Results saved to {out_path}")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log:  {log_files['full']}")


if __name__ == "__main__":
    main()
