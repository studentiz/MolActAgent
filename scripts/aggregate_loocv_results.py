#!/usr/bin/env python3
"""Merge LOOCV JSON outputs into data/loocv_all_results_consolidated.json."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SOURCES: list[tuple[str, str]] = [
    ("loocv_evidence_results.json", "仅 ML，描述符+MACCS，两数据集"),
    ("loocv_ml_desc_maccs_morgan_loocv.json", "仅 ML，描述符+MACCS+Morgan，两数据集"),
    ("loocv_deepseek_both_k3.json", "DeepSeek：ML + Pure LLM + MolActAgent(K=3)，描述符+MACCS，两数据集"),
    ("loocv_BSLT_molactagent_k3.json", "BSLT：ML + MolActAgent，跳过 Pure LLM，描述符+MACCS"),
]

METHOD_ORDER = [
    "KNN1", "KNN3", "KNN5", "NB", "RF", "SVM", "LR", "GBC", "XGB",
    "PureLLM", "MolActAgent",
]

KEY_METHODS = ["KNN3", "SVM", "RF", "PureLLM", "MolActAgent"]

DATASET_ORDER = {"antiTB": 0, "BSLT": 1}


def dataset_key(csv_path: str) -> str:
    stem = Path(csv_path).stem.lower()
    if "antitb" in stem:
        return "antiTB"
    if "bslt" in stem:
        return "BSLT"
    return Path(csv_path).stem


def feature_id_from_args(args: dict) -> tuple[str, str]:
    if args.get("add_morgan"):
        return (
            "desc_MACCS_Morgan",
            "RDKit 描述符 + MACCS（零方差）+ Morgan 指纹（radius=2, 2048 bit，零方差）",
        )
    return ("desc_MACCS", "RDKit 描述符 + MACCS（零方差）")


def round_metrics(m: dict) -> dict:
    out: dict = {}
    for k, v in m.items():
        if k in ("tp", "tn", "fp", "fn", "unknown"):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v
    return out


def sort_methods(keys: list[str]) -> list[str]:
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(keys, key=lambda x: (order.get(x, 100), x))


def build_summary_text(rows_out: list[dict]) -> str:
    def fmt_block(dkey: str, fid: str, label_short: str) -> str:
        r = next(
            (x for x in rows_out if x["dataset"] == dkey and x["feature_set_id"] == fid),
            None,
        )
        if not r:
            return f"{dkey} | {label_short}: (无数据)"
        m = r["methods"]
        parts = [f"{dkey} | {label_short} | n={r['n_samples']}"]
        for km in KEY_METHODS:
            if km not in m:
                continue
            mv = m[km]
            parts.append(
                f"  {km}: Acc={mv['accuracy']:.3f} MCC={mv['mcc']:.3f} MacroF1={mv['macro_f1']:.3f} "
                f"(TP/TN/FP/FN={mv['tp']}/{mv['tn']}/{mv['fp']}/{mv['fn']})"
            )
        return "\n".join(parts)

    lines = ["=== LOOCV 汇总（关键方法：KNN3 / SVM / RF / PureLLM / MolActAgent）==="]
    for dkey in ["antiTB", "BSLT"]:
        lines.append("")
        lines.append(fmt_block(dkey, "desc_MACCS", "描述符+MACCS"))
        lines.append("")
        lines.append(fmt_block(dkey, "desc_MACCS_Morgan", "描述符+MACCS+Morgan（仅 ML）"))
    lines.append("")
    lines.append("完整方法列表与来源见 JSON 字段 results / source_catalog。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <project>/data/loocv_all_results_consolidated.json",
    )
    args_p = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    out_path = args_p.output or (data_dir / "loocv_all_results_consolidated.json")

    bucket: dict = defaultdict(lambda: defaultdict(lambda: {"metrics": None, "sources": []}))
    run_meta: dict = {}

    for fname, desc_zh in DEFAULT_SOURCES:
        path = data_dir / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        a = blob.get("args") or {}
        fid, flabel = feature_id_from_args(a)
        for run in blob.get("runs") or []:
            dkey = dataset_key(run.get("csv", ""))
            nk = (dkey, fid)
            run_meta[nk] = {
                "n_samples": run.get("n_samples"),
                "pos": run.get("pos"),
                "neg": run.get("neg"),
                "feature_label": flabel,
            }
            for method, mvals in (run.get("metrics") or {}).items():
                cell = bucket[nk][method]
                rv = round_metrics(mvals)
                if cell["metrics"] is None:
                    cell["metrics"] = rv
                elif cell["metrics"] != rv:
                    cell.setdefault("conflicts", []).append({"file": fname, "metrics": rv})
                cell["sources"].append(fname)

    rows_out: list[dict] = []
    for (dkey, fid) in sorted(
        bucket.keys(), key=lambda x: (DATASET_ORDER.get(x[0], 99), x[1])
    ):
        methods_block: dict = {}
        prov_block: dict = {}
        for method in sort_methods(list(bucket[(dkey, fid)].keys())):
            cell = bucket[(dkey, fid)][method]
            methods_block[method] = cell["metrics"]
            prov_block[method] = sorted(set(cell["sources"]))
            if cell.get("conflicts"):
                methods_block[method] = {
                    **cell["metrics"],
                    "_warning": "multiple source files disagree; see conflicts",
                    "_conflicts": cell["conflicts"],
                }
        meta = run_meta.get((dkey, fid), {})
        rows_out.append(
            {
                "dataset": dkey,
                "feature_set_id": fid,
                "feature_set_description": meta.get("feature_label", ""),
                "n_samples": meta.get("n_samples"),
                "n_pos": meta.get("pos"),
                "n_neg": meta.get("neg"),
                "methods": methods_block,
                "provenance": prov_block,
            }
        )

    catalog = [
        {"file": name, "path": str((data_dir / name).resolve()), "description_zh": zh}
        for name, zh in DEFAULT_SOURCES
    ]

    out = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "title": "MolActAgent 项目 LOOCV 结果整合（两数据集 × 特征 × 方法）",
            "evaluation": (
                "留一法（LOOCV）；标签 active→Pos，inactive→Neg；"
                "BSLT 表头 class_lable；MolActAgent / Pure LLM 使用描述符+MACCS（未加 Morgan）"
            ),
        },
        "source_catalog": catalog,
        "results": rows_out,
        "summary_text_zh": build_summary_text(rows_out),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
