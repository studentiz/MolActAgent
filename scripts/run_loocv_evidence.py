"""
Leave-one-out cross-validation (LOOCV) on evidence-table CSVs:
  MolActAgent (configurable K, default 3), pure LLM baseline, and ML baselines.

Example:
  python scripts/run_loocv_evidence.py --dataset both --k 3
  python scripts/run_loocv_evidence.py --csv data/antiTB_evidence_table.csv --k 3
  python scripts/run_loocv_evidence.py --dataset both --skip-llm   # ML only

NOTE — NOT YET MIGRATED TO THE MULTI-CLASS PIPELINE.

This script still hard-codes the legacy ``"Pos"`` / ``"Neg"`` label
contract throughout (the ML baseline encoders, the pure-LLM
baseline prompt, the per-fold metric counters). After the
multi-class refactor ``prepare_features.build_dataset_from_dataframe``
returns ``y`` with the original user labels (``"active"`` /
``"inactive"`` for these datasets), so running this file as-is would
silently produce nonsense (every classifier would see a single
class and predict accordingly).

For binary LOOCV with the same datasets, use
``scripts/run_loocv_metrics.py`` — it has been fully migrated, runs
the same ML baselines (KNN, NB, LR, SVM, RF, GBC, XGB), and
produces the same metric suite plus per-fold MolActAgent records.
That script is the validated path used in
``data/refactor_validation.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time


def _refuse_to_run_unmigrated_script() -> None:
    print(
        "\nERROR: scripts/run_loocv_evidence.py has not been migrated to "
        "the multi-class label contract. The script still assumes "
        "``y == 'Pos'`` / ``y == 'Neg'`` throughout, but "
        "``prepare_features.build_dataset_from_dataframe`` now returns "
        "the original user labels (e.g. ``'active'`` / ``'inactive'``).\n\n"
        "Use scripts/run_loocv_metrics.py instead — it runs the same "
        "ML baselines and MolActAgent LOOCV with the new schema:\n\n"
        "  python scripts/run_loocv_metrics.py "
        "--csv data/antiTB_evidence_table.csv --k 3 --llm-profile deepseek\n",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    _refuse_to_run_unmigrated_script()

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[misc, assignment]

try:
    from tabpfn import TabPFNClassifier as _TabPFNClassifier
    _HAS_TABPFN = True
except ImportError:
    _HAS_TABPFN = False

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from molactagent import MolActAgent, LLMClient, setup_logging  # noqa: E402

from prepare_features import build_dataset_from_dataframe  # noqa: E402
from run_demo import compute_metrics, load_llm_config, print_metrics_table  # noqa: E402

logger = logging.getLogger("loocv_evidence")

BACKGROUND_ANTITB = (
    "任务背景：环肽类化合物的抗结核分枝杆菌（Mycobacterium tuberculosis）活性二分类。\n"
    "标签语义：Pos 表示具有抗结核活性（文献/实验中记为 active）；"
    "Neg 表示无显著活性（inactive）。\n"
    "特征为 RDKit 分子描述符与 MACCS 结构指纹（与 MolActAgent 默认管线一致）。"
)

BACKGROUND_BSLT = (
    "任务背景：数据集包含环肽类与聚酮类化合物，以盐水虾（Artemia salina）"
    "致死毒性试验衡量与安全性相关的毒性风险。\n"
    "标签语义：Pos 表示在该毒性试验中呈阳性/有毒性（文献中记为 active）；"
    "Neg 表示阴性或低毒（inactive）。\n"
    "特征为 RDKit 分子描述符与 MACCS 结构指纹。"
)


def pure_llm_predict_one(
    llm: LLMClient,
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    background: str,
    tag: str,
) -> str:
    """Single-sample binary classification (same prompt pattern as run_demo.run_pure_llm_baseline)."""
    ref_summary_lines = []
    for i in range(min(len(y_train), 10)):
        label = y_train[i]
        vals = ", ".join(
            f"{feature_names[j]}={X_train[i][j]:.2f}"
            for j in range(min(10, len(feature_names)))
        )
        ref_summary_lines.append(f"  [{label}] {vals} ...")
    ref_summary = "\n".join(ref_summary_lines)

    test_vals = "\n".join(
        f"  {feature_names[j]}: {float(x_test[j]):.4f}"
        for j in range(len(feature_names))
    )
    prompt = (
        f"You are a binary classifier. Background:\n{background}\n\n"
        f"Reference data (first 10 samples, showing first 10 features):\n"
        f"{ref_summary}\n\n"
        f"Total reference: {sum(1 for l in y_train if l == 'Pos')} Pos, "
        f"{sum(1 for l in y_train if l == 'Neg')} Neg\n\n"
        f"Test sample features:\n{test_vals}\n\n"
        f"Classify as exactly Pos or Neg. Reply with ONLY the label."
    )
    try:
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            tag=tag,
        )
        lo = resp.lower()
        if "neg" in lo and "pos" not in lo:
            return "Neg"
        if "pos" in lo:
            return "Pos"
        if "neg" in lo:
            return "Neg"
        return "Unknown"
    except Exception as e:
        logger.error(f"[PureLLM] {tag} failed: {e}")
        return "Unknown"


def _make_ml_classifiers(y_train: np.ndarray):
    y_tr_bin = np.array([1 if l == "Pos" else 0 for l in y_train])
    pos_count = int(y_tr_bin.sum())
    neg_count = len(y_tr_bin) - pos_count
    xgb_scale = neg_count / pos_count if pos_count > 0 else 1.0

    out: dict = {
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
        "GBC": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if XGBClassifier is not None:
        out["XGB"] = XGBClassifier(
            n_estimators=100,
            scale_pos_weight=xgb_scale,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    return out


def run_ml_loocv(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    n = len(y)
    label_map = {"Pos": 1, "Neg": 0}
    inv_map = {1: "Pos", 0: "Neg"}
    needs_numeric = {"XGB"}

    dummy_train = np.setdiff1d(np.arange(n), [0], assume_unique=True)
    names = list(_make_ml_classifiers(y[dummy_train]).keys())
    all_preds: dict[str, list[str]] = {name: [] for name in names}

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i], dtype=int)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[i : i + 1]

        if len(np.unique(y_train)) < 2:
            logger.warning(
                f"LOOCV fold {i}: training set is single-class; "
                "ML predictions default to majority / arbitrary."
            )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        classifiers = _make_ml_classifiers(y_train)
        for name, clf in classifiers.items():
            try:
                if name in needs_numeric:
                    y_tr_enc = np.array([label_map[l] for l in y_train])
                    clf.fit(X_tr, y_tr_enc)
                    pred_raw = clf.predict(X_te)[0]
                    pred = inv_map[int(pred_raw)]
                else:
                    clf.fit(X_tr, y_train)
                    pred = str(clf.predict(X_te)[0])
            except Exception as e:
                logger.error(f"LOOCV fold {i} ML {name} failed: {e}")
                pred = "Unknown"
            all_preds[name].append(pred)

    y_list = y.tolist()
    results = {name: compute_metrics(y_list, preds) for name, preds in all_preds.items()}

    if _HAS_TABPFN:
        results.update(_run_tabpfn_loocv(X, y))

    return results


def _run_tabpfn_loocv(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Run TabPFN LOOCV separately (heavier per-fold cost)."""
    n = len(y)
    label_map = {"Pos": 1, "Neg": 0}
    inv_map = {1: "Pos", 0: "Neg"}
    preds: list[str] = []

    logger.info("Running TabPFN LOOCV ...")
    t0 = time.time()
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i], dtype=int)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[i : i + 1]
        y_tr_enc = np.array([label_map[l] for l in y_train])

        try:
            clf = _TabPFNClassifier()
            clf.fit(X_train, y_tr_enc)
            pred_raw = clf.predict(X_test)[0]
            pred = inv_map[int(pred_raw)]
        except Exception as e:
            logger.error(f"TabPFN LOOCV fold {i} failed: {e}")
            pred = "Unknown"
        preds.append(pred)

    elapsed = time.time() - t0
    logger.info(f"TabPFN LOOCV finished in {elapsed:.1f}s")
    return {"TabPFN": compute_metrics(y.tolist(), preds)}


def run_molactagent_loocv(
    llm: LLMClient,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_descriptions: dict[str, str],
    background: str,
    k: int,
    metric: str,
    max_workers: int,
    use_scaled_for_prompt: bool,
) -> tuple[dict, list[dict], list[str]]:
    n = len(y)
    y_pred: list[str] = []
    details: list[dict] = []

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i], dtype=int)
        X_train, y_train = X[train_idx], y[train_idx]
        if len(np.unique(y_train)) < 2:
            logger.warning(
                f"LOOCV fold {i}: training set single-class; "
                "MolActAgent may not form Pos/Neg neighbor sets."
            )

        agent = MolActAgent(
            llm_client=llm,
            k=k,
            metric=metric,
            feature_names=list(feature_names),
            feature_descriptions=dict(feature_descriptions),
            background=background,
            use_scaled_for_prompt=use_scaled_for_prompt,
            max_workers=max_workers,
        )
        agent.fit(X_train, y_train)
        logger.info(
            f"\n{'='*60}\nLOOCV MolActAgent fold {i + 1}/{n}  held_out true={y[i]}\n{'='*60}"
        )
        t0 = time.time()
        res = agent.predict_one(X[i], test_id=i)
        elapsed = time.time() - t0
        pred = res.get("final_label", "Unknown")
        y_pred.append(pred)
        details.append(res)
        logger.info(
            f"  => pred={pred}  true={y[i]}  conf={res.get('confidence', 0):.2f}  "
            f"time={elapsed:.1f}s"
        )

    metrics = compute_metrics(y.tolist(), y_pred)
    return metrics, details, y_pred


def run_one_csv(
    csv_path: pathlib.Path,
    background: str,
    args: argparse.Namespace,
    llm: LLMClient | None,
) -> dict:
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    X, y, meta = build_dataset_from_dataframe(
        df,
        background,
        verbose=True,
        add_morgan=getattr(args, "add_morgan", False),
        morgan_radius=getattr(args, "morgan_radius", 2),
        morgan_n_bits=getattr(args, "morgan_n_bits", 2048),
    )

    out: dict = {
        "csv": str(csv_path),
        "n_samples": int(len(y)),
        "pos": int((y == "Pos").sum()),
        "neg": int((y == "Neg").sum()),
    }

    all_metrics: dict[str, dict] = {}

    logger.info("\n" + "=" * 60 + "\n  ML baselines (LOOCV)\n" + "=" * 60)
    t0 = time.time()
    ml_metrics = run_ml_loocv(X, y)
    all_metrics.update(ml_metrics)
    logger.info(f"ML LOOCV finished in {time.time() - t0:.1f}s")

    if args.skip_llm:
        out["metrics"] = all_metrics
        return out

    assert llm is not None
    pure_preds: list[str] | None = None
    if not getattr(args, "skip_pure_llm", False):
        logger.info("\n" + "=" * 60 + "\n  Pure LLM (LOOCV)\n" + "=" * 60)
        t0 = time.time()
        pure_preds = []
        n = len(y)
        for i in range(n):
            train_idx = np.array([j for j in range(n) if j != i], dtype=int)
            X_train, y_train = X[train_idx], y[train_idx]
            pred = pure_llm_predict_one(
                llm,
                X_train,
                y_train,
                X[i],
                meta["feature_names"],
                background,
                tag=f"pureLLM_loocv_{csv_path.stem}_{i}",
            )
            pure_preds.append(pred)

        pure_metrics = compute_metrics(y.tolist(), pure_preds)
        all_metrics["PureLLM"] = pure_metrics
        logger.info(f"Pure LLM LOOCV finished in {time.time() - t0:.1f}s")
    else:
        logger.info("Skipping Pure LLM (--skip-pure-llm).")

    logger.info("\n" + "=" * 60 + "\n  MolActAgent (LOOCV)\n" + "=" * 60)
    t0 = time.time()
    ca_metrics, ca_details, ca_preds = run_molactagent_loocv(
        llm,
        X,
        y,
        meta["feature_names"],
        meta["feature_descriptions"],
        background,
        k=args.k,
        metric=args.metric,
        max_workers=args.max_workers,
        use_scaled_for_prompt=args.use_scaled_values,
    )
    all_metrics["MolActAgent"] = ca_metrics
    logger.info(f"MolActAgent LOOCV finished in {time.time() - t0:.1f}s")

    out["metrics"] = all_metrics
    out["molactagent_predictions"] = ca_details
    if pure_preds is not None:
        out["pure_llm_predictions"] = pure_preds
    return out


def main():
    parser = argparse.ArgumentParser(description="LOOCV on evidence CSVs")
    parser.add_argument(
        "--dataset",
        choices=["antiTB", "BSLT", "both"],
        default="both",
        help="Preset CSVs under data/",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Single CSV path (overrides --dataset). Use --background with antitb|bslt.",
    )
    parser.add_argument(
        "--background",
        choices=["antitb", "bslt"],
        default="antitb",
        help="Task description for LLM when using --csv (default: antitb)",
    )
    parser.add_argument("--k", type=int, default=3, help="MolActAgent neighbors per class")
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Thread pool size for parallel triplet LLM calls (default: 100, same as run_demo).",
    )
    parser.add_argument("--use-scaled-values", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--skip-pure-llm",
        action="store_true",
        help="Run ML + MolActAgent only (skip pure-LLM baseline; saves API calls).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results JSON to this path (default: data/loocv_evidence_results.json).",
    )
    parser.add_argument(
        "--llm-profile",
        type=str,
        default="last",
        metavar="PROFILE",
        help=(
            "Provider block in LLM_API.txt: last (default, same as legacy), "
            "deepseek / ds, or qwen / dashscope."
        ),
    )
    parser.add_argument(
        "--add-morgan",
        action="store_true",
        help="Append RDKit Morgan fingerprint bits (after MACCS; variance-filtered).",
    )
    parser.add_argument(
        "--morgan-radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2, ECFP4-like).",
    )
    parser.add_argument(
        "--morgan-n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint length in bits (default: 2048).",
    )
    args = parser.parse_args()

    project_root = _PROJECT_ROOT
    log_files = setup_logging(log_dir=project_root / "logs")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log:  {log_files['full']}")

    llm: LLMClient | None = None
    if not args.skip_llm:
        prof = args.llm_profile.strip().lower()
        if prof in ("ds",):
            prof = "deepseek"
        if prof in ("dashscope",):
            prof = "qwen"
        llm_cfg = load_llm_config(project_root, profile=prof if prof != "last" else None)
        logger.info(
            f"LLM profile={args.llm_profile!r} -> model={llm_cfg['model_name']!r} "
            f"base_url={llm_cfg['base_url']!r}"
        )
        llm = LLMClient(**llm_cfg, timeout=120.0)

    runs: list[dict] = []

    if args.csv:
        csv_paths = [pathlib.Path(args.csv).resolve()]
        backgrounds = [
            BACKGROUND_ANTITB if args.background == "antitb" else BACKGROUND_BSLT
        ]
    elif args.dataset == "antiTB":
        csv_paths = [project_root / "data" / "antiTB_evidence_table.csv"]
        backgrounds = [BACKGROUND_ANTITB]
    elif args.dataset == "BSLT":
        csv_paths = [project_root / "data" / "BSLT_evidence_table.csv"]
        backgrounds = [BACKGROUND_BSLT]
    else:
        csv_paths = [
            project_root / "data" / "antiTB_evidence_table.csv",
            project_root / "data" / "BSLT_evidence_table.csv",
        ]
        backgrounds = [BACKGROUND_ANTITB, BACKGROUND_BSLT]

    for csv_path, bg in zip(csv_paths, backgrounds):
        logger.info(f"\n########## {csv_path.name} ##########")
        run_result = run_one_csv(csv_path, bg, args, llm)
        runs.append(run_result)
        print_metrics_table(run_result["metrics"])
        if llm is not None:
            llm.print_stats()

    out_path = (
        pathlib.Path(args.output_json).resolve()
        if args.output_json
        else project_root / "data" / "loocv_evidence_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "runs": runs,
                "llm_stats": llm.stats if llm else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    logger.info(f"Wrote aggregated results to {out_path}")
    logger.info(f"Brief log: {log_files['brief']}")
    logger.info(f"Full log:  {log_files['full']}")


if __name__ == "__main__":
    main()
