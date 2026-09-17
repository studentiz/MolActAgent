"""Build the Penguins regression example from the vendored CC0 CSV.

The Palmer Penguins dataset is small but classic, with one obvious
continuous target (``body_mass_g``) and four numeric predictors
(bill length / depth, flipper length, plus a non-numeric
species/island/sex). We strip categorical columns and rows with
missing values, then take a deterministic ~30-row subsample for the
example so the LLM-driven flow stays cheap to demo.

Usage:

    python scripts/prepare_penguins_subsample.py \
        --raw data/raw/penguins.csv \
        --out-dir examples/penguins/

The script writes:

    examples/penguins/reference.csv   (24 rows; LOOCV-ready)
    examples/penguins/test.csv         (6 rows; held-out for demos)
    examples/penguins/feature_descriptions.json
    examples/penguins/background.txt
    examples/penguins/llm_config.jsonc.template
"""
from __future__ import annotations

import argparse
import json
import pathlib
from textwrap import dedent

import numpy as np
import pandas as pd

DEFAULT_NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
]
DEFAULT_TARGET = "body_mass_g"
DEFAULT_ID_COL = "penguin_id"
DEFAULT_SAMPLE_ID_COL = "sample_id"
DEFAULT_SEED = 2026
DEFAULT_TOTAL = 30
DEFAULT_TEST = 6


FEATURE_DESCRIPTIONS = {
    "bill_length_mm": (
        "Length of the penguin's bill from base to tip, in "
        "millimetres. Generally rises with overall body size; "
        "Gentoos and Chinstraps have noticeably longer bills than "
        "Adelies."
    ),
    "bill_depth_mm": (
        "Vertical thickness of the bill at its base, in "
        "millimetres. Adelies and Chinstraps have deep bills "
        "(usually 17-21 mm); Gentoos are shallower (typically "
        "13-16 mm), so deeper bills tend to come with smaller "
        "bodies in this dataset."
    ),
    "flipper_length_mm": (
        "Length of the flipper, in millimetres. The single best "
        "monotone proxy for body mass in penguins — larger penguins "
        "have longer flippers."
    ),
    "body_mass_g": (
        "Body mass in grams. The continuous target value to "
        "predict. Adult Adelies and Chinstraps cluster around "
        "3.5-4.5 kg; Gentoos cluster around 4.7-6.0 kg."
    ),
}

BACKGROUND_TEXT = dedent("""
Palmer Archipelago penguins, three species (Adelie, Chinstrap, Gentoo),
measured between 2007 and 2009 by Dr. Kristen Gorman at the Long-Term
Ecological Research site near Palmer Station. Body mass varies smoothly
with structural size: birds with longer flippers and longer bills are
heavier on average. Bill DEPTH is the only feature that runs the
opposite way — Gentoos are heavier but have shallower bills than the
two Adelie-sized species. The reference set is deliberately small (24
adults across all three species) so the example is cheap to run; the
agent should treat large flipper / long bill as evidence of a
heavier bird, while reading bill depth in context.

The CSV files also carry a ``species`` column for human readability
and a ``sample_id`` column for unique row identifiers. Neither is
used as a feature in the regression run — drop ``species`` before
passing the data to ``run_regression`` and pass ``sample_id`` via
``id_column``.
""").strip()


def _drop_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = df.copy()
    for c in cols:
        keep = keep[keep[c].notna()]
    return keep.reset_index(drop=True)


def _stratified_subsample(
    df: pd.DataFrame,
    n_total: int,
    n_test: int,
    seed: int,
    strat_col: str = "species",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reference_df, test_df) with the test split pulled
    proportionally from each species so the demo always covers the
    multi-cluster nature of the dataset."""
    rng = np.random.default_rng(seed)
    if strat_col not in df.columns:
        sampled = df.sample(n=n_total, random_state=seed).reset_index(drop=True)
    else:
        groups = df.groupby(strat_col)
        per_group_total = max(1, n_total // len(groups))
        picked: list[pd.DataFrame] = []
        for _, g in groups:
            take = min(per_group_total, len(g))
            picked.append(g.sample(n=take, random_state=seed).reset_index(drop=True))
        sampled = pd.concat(picked).reset_index(drop=True)
        if len(sampled) > n_total:
            sampled = sampled.sample(n=n_total, random_state=seed).reset_index(drop=True)

    perm = rng.permutation(len(sampled))
    test_idx = perm[:n_test]
    ref_idx = perm[n_test:]
    test_df = sampled.iloc[test_idx].sort_index().reset_index(drop=True)
    ref_df = sampled.iloc[ref_idx].sort_index().reset_index(drop=True)
    return ref_df, test_df


def build(
    raw_csv: pathlib.Path,
    out_dir: pathlib.Path,
    target: str = DEFAULT_TARGET,
    features: list[str] | None = None,
    n_total: int = DEFAULT_TOTAL,
    n_test: int = DEFAULT_TEST,
    seed: int = DEFAULT_SEED,
) -> None:
    if features is None:
        features = DEFAULT_NUMERIC_FEATURES
    raw = pd.read_csv(raw_csv)
    keep_cols = features + [target, "species"]
    raw = raw[keep_cols]
    clean = _drop_missing(raw, features + [target])

    ref_df, test_df = _stratified_subsample(
        clean, n_total=n_total, n_test=n_test, seed=seed,
    )

    ref_df.insert(0, DEFAULT_SAMPLE_ID_COL, [
        f"penguin_R{i:02d}" for i in range(len(ref_df))
    ])
    test_df.insert(0, DEFAULT_SAMPLE_ID_COL, [
        f"penguin_T{i:02d}" for i in range(len(test_df))
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(out_dir / "reference.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # Only ship descriptions for keys that survive into the agent's
    # feature set. ``species`` and ``sample_id`` are present in the
    # CSVs but stripped before run_regression is called, so listing
    # them here would just trigger the validator's "feature
    # description references missing column" warning.
    desc = {f: FEATURE_DESCRIPTIONS[f] for f in features if f in FEATURE_DESCRIPTIONS}
    desc[target] = FEATURE_DESCRIPTIONS.get(target, "Continuous regression target.")
    (out_dir / "feature_descriptions.json").write_text(
        json.dumps(desc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "background.txt").write_text(
        BACKGROUND_TEXT + "\n", encoding="utf-8",
    )
    (out_dir / "llm_config.jsonc.template").write_text(
        dedent("""\
        {
          // Replace with your own API credentials and model.
          "apiKey": "sk-REPLACE_ME",
          "base_url": "https://api.deepseek.com/v1",
          "Modelname": "deepseek-chat",
          "temperature": 0.0
        }
        """),
        encoding="utf-8",
    )

    print(
        f"Wrote {len(ref_df)} reference rows and {len(test_df)} test "
        f"rows to {out_dir}/."
    )
    print(
        f"Target column: {target!r}, range "
        f"[{ref_df[target].min():.0f}, {ref_df[target].max():.0f}], "
        f"distinct = {ref_df[target].nunique()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default="data/raw/penguins.csv", type=pathlib.Path,
        help="Path to the vendored Palmer Penguins CSV.",
    )
    parser.add_argument(
        "--out-dir", default="examples/penguins/", type=pathlib.Path,
        help="Where to write the example reference / test / config files.",
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET,
        help="Continuous target column to regress.",
    )
    parser.add_argument(
        "--features", nargs="+", default=DEFAULT_NUMERIC_FEATURES,
        help="Numeric feature columns to keep.",
    )
    parser.add_argument(
        "--total", type=int, default=DEFAULT_TOTAL,
        help="Total subsample size (reference + test).",
    )
    parser.add_argument(
        "--test", type=int, default=DEFAULT_TEST,
        help="How many rows to hold out as the test split.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for the subsample (use the default for "
        "the reproducible example).",
    )
    args = parser.parse_args()
    build(
        raw_csv=args.raw,
        out_dir=args.out_dir,
        target=args.target,
        features=args.features,
        n_total=args.total,
        n_test=args.test,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
