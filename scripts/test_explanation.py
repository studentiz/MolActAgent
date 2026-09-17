"""Effectiveness test for the rebuilt explanation pipeline.

Uses the FIRST row of ``data/antiTB_evidence_table.csv`` as the test
sample and ALL remaining rows as the reference set, then runs the
MolActAgent end-to-end, generates the Markdown report, and renders
the HTML.

Usage:
    conda activate molactagent_env
    python scripts/test_explanation.py
"""
from __future__ import annotations

import pathlib
import sys

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd

from molactagent import MolActAgent, LLMClient, render_html_report, setup_logging
from prepare_features import build_dataset_from_dataframe
from run_demo import load_llm_config

BACKGROUND = (
    "Task: classification of cyclic peptide compounds for "
    "anti-tuberculosis (Mycobacterium tuberculosis) activity.\n"
    "Class labels: ``active`` (compound exhibits anti-TB activity in "
    "the literature/experiment) vs ``inactive`` (no significant "
    "activity).\n"
    "Features: RDKit molecular descriptors and MACCS structural "
    "fingerprints. For peptide-like compounds, key drivers of "
    "anti-TB activity tend to involve membrane permeability "
    "(TPSA, MolLogP, MolWt), conformational rigidity (rotatable "
    "bond count, ring count), and specific structural motifs "
    "(MACCS fingerprint bits)."
)

K_NEIGHBORS = 3
LLM_PROFILE = "deepseek"
TEST_ROW_INDEX = 0


def main():
    log_files = setup_logging(log_dir=_PROJECT_ROOT / "logs")
    print(f"Brief log: {log_files['brief']}")
    print(f"Full log:  {log_files['full']}")

    csv_path = _PROJECT_ROOT / "data" / "antiTB_evidence_table.csv"
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples from {csv_path.name}")

    X, y, meta = build_dataset_from_dataframe(df, BACKGROUND, verbose=True)
    feature_names = meta["feature_names"]
    feature_descriptions = meta["feature_descriptions"]

    test_idx = TEST_ROW_INDEX
    true_label = str(y[test_idx])
    print(
        f"\nTest sample: row {test_idx}  "
        f"(SMILES: {meta['smiles'][test_idx][:60]}{'...' if len(meta['smiles'][test_idx]) > 60 else ''})"
    )
    print(f"True label:  {true_label}")

    train_mask = np.array([i != test_idx for i in range(len(y))])
    X_train, y_train = X[train_mask], y[train_mask]
    x_test = X[test_idx]
    classes_in_train, counts_in_train = np.unique(y_train, return_counts=True)
    per_class_text = ", ".join(
        f"{int(c)} {cls}" for cls, c in zip(classes_in_train, counts_in_train)
    )
    print(f"Reference set: {len(y_train)} samples ({per_class_text})")

    llm_cfg = load_llm_config(_PROJECT_ROOT, profile=LLM_PROFILE)
    print(
        f"\nLLM: model={llm_cfg['model_name']}  "
        f"base_url={llm_cfg['base_url']}"
    )
    llm = LLMClient(**llm_cfg, timeout=120.0)

    agent = MolActAgent(
        llm_client=llm,
        k=K_NEIGHBORS,
        feature_names=list(feature_names),
        feature_descriptions=dict(feature_descriptions),
        background=BACKGROUND,
        max_workers=20,
    )
    agent.fit(X_train, y_train)

    print(
        f"\n--- Running prediction (k={K_NEIGHBORS}) ---"
    )
    result, explanation = agent.analyze(
        x_test,
        test_id=test_idx,
        class_labels=agent.class_labels,
    )

    pred_label = result.get("final_label", "Inconclusive")
    print("\n========== PREDICTION ==========")
    print(f"  True:        {true_label}")
    print(f"  Predicted:   {pred_label}")
    print(f"  Confidence:  {result['confidence']:.0%}")
    print(f"  Basis:       {result['confidence_basis'].get('basis_text', '')}")
    print(f"  Rationale:   {result['confidence_basis'].get('rationale', '')}")

    vs = result.get("vote_summary", {}) or {}
    per = vs.get("per_class", {}) or {}
    tally_parts = [
        f"{cls}={info.get('votes', 0)} (w={info.get('weighted', 0):.2f})"
        for cls, info in per.items()
    ]
    tally_parts.append(f"Unknown={vs.get('num_unknown', 0)}")
    print(f"  Vote tally:  {'  '.join(tally_parts)}")

    print("\n  Top driver features (cited as drivers, weighted):")
    for fa in result.get("driver_attribution", [])[:6]:
        dom = fa.get("dominant_class") or "Mixed"
        print(
            f"    -> {fa['feature']:<22s} "
            f"cited={fa.get('n_cited', 0):>2d} "
            f"weighted_score={fa.get('weighted_score', 0):+.2f}  "
            f"(favours {dom})"
        )

    print("\n========== MARKDOWN EXPLANATION (first 800 chars) ==========")
    print(explanation[:800])
    print("..." if len(explanation) > 800 else "")
    print(f"[Full Markdown length: {len(explanation)} chars]")

    print("\n--- Rendering HTML report ---")
    report_html = render_html_report(
        result,
        explanation=explanation,
        class_labels=agent.class_labels,
        title="MolActAgent Anti-TB Report",
    )
    out_path = _PROJECT_ROOT / "data" / f"report_v2_sample{test_idx}.html"
    out_path.write_text(report_html, encoding="utf-8")
    print(f"Report saved to: {out_path}")
    print(f"Open in browser: file://{out_path}")

    md_path = _PROJECT_ROOT / "data" / f"report_v2_sample{test_idx}.md"
    md_path.write_text(explanation, encoding="utf-8")
    print(f"Markdown saved to: {md_path}")

    print()
    llm.print_stats()


if __name__ == "__main__":
    main()
