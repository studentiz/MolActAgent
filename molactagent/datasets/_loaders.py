"""Concrete dataset loaders.

Each loader resolves its files relative to the *project root* (the
parent directory of the ``molactagent`` package). This is the right
thing for editable installs and for the bundled examples; for users
who installed the package from a wheel without the ``examples/``
directory, the loaders raise ``FileNotFoundError`` with an explicit
"the dataset lives in the source repo, not the installed wheel"
note.

This is the only place in the package that touches files outside
the ``molactagent`` source tree, and it does so as **read-only data
access** — the package never writes to ``examples/`` / ``data/``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pandas as pd


def _project_root() -> pathlib.Path:
    """Resolve the project root by walking up from this file.

    Layout:
        <root>/molactagent/datasets/_loaders.py
        <root>/examples/
        <root>/data/

    Editable installs preserve this layout; wheel installs do not
    ship the example / data directories. The loaders raise a
    descriptive error in the latter case.
    """
    return pathlib.Path(__file__).resolve().parent.parent.parent


def _missing_dataset_msg(path: pathlib.Path, hint: str) -> str:
    return (
        f"Bundled dataset path not found: {path}\n"
        f"Hint: {hint}\n"
        f"This dataset ships with the MolActAgent source repository "
        "but is not part of the wheel install. To make the bundled "
        "examples available, clone the project repository and install "
        "it editably (`pip install -e .` from the checkout). See the "
        "project README for installation details."
    )


# ── iris flowers (multi-class classification) ─────────────────────


def load_iris_flowers():
    """Load the bundled iris-flowers classification example.

    Returns a :class:`~molactagent.pp.tabular.TabularBundle` with:

    - ``X_train, y_train`` (the reference set; ~12 rows × 3
      species).
    - ``X_test, y_test`` (4-6 held-out flowers with known truth).
    - ``feature_descriptions``, ``background`` populated from the
      example's JSON / TXT files.
    - ``class_labels = ["setosa", "versicolor", "virginica"]``.
    """
    from .. import pp as _pp

    iris_dir = _project_root() / "examples" / "iris_flowers"
    if not iris_dir.exists():
        raise FileNotFoundError(_missing_dataset_msg(
            iris_dir,
            "Expected 'examples/iris_flowers/' next to the package "
            "source root.",
        ))
    return _pp.tabular.load_dataset(
        reference=iris_dir / "reference.csv",
        test=iris_dir / "test.csv",
        label_column="species",
        id_column="flower_id",
        feature_descriptions=iris_dir / "feature_descriptions.json",
        background=f"@{iris_dir / 'background.txt'}",
    )


# ── penguins (regression) ────────────────────────────────────────


def load_penguins():
    """Load the bundled Palmer Penguins regression example.

    Returns a :class:`~molactagent.pp.tabular.TabularBundle` with
    body mass (g) as the continuous target.

    The bundled CSVs include a ``species`` column that the
    ``background.txt`` explicitly says to drop before regression
    ("The CSV files also carry a ``species`` column for human
    readability ... not used as a feature in the regression run —
    drop ``species`` before passing the data to
    ``run_regression``"). We honour that here by reading the CSVs
    into DataFrames, dropping ``species``, and then handing them
    to the validator. This keeps the on-disk fixtures
    human-readable while the loader does the right thing for the
    estimator path.

    The example must be built once with
    ``python scripts/prepare_penguins_subsample.py``; this loader
    raises a ``FileNotFoundError`` that points at the script when
    the files are missing.
    """
    from .. import pp as _pp

    pen_dir = _project_root() / "examples" / "penguins"
    if not (pen_dir / "reference.csv").exists():
        raise FileNotFoundError(_missing_dataset_msg(
            pen_dir / "reference.csv",
            "Build the penguins example first by running "
            "`python scripts/prepare_penguins_subsample.py` "
            "from the project root.",
        ))

    # Read DataFrames directly so we can drop the metadata-only
    # ``species`` column before validation. Passing path-strings
    # straight through would let validation see ``species`` as a
    # feature column and reject it (non-numeric).
    ref_df = pd.read_csv(pen_dir / "reference.csv")
    test_df = pd.read_csv(pen_dir / "test.csv")
    for df in (ref_df, test_df):
        if "species" in df.columns:
            df.drop(columns=["species"], inplace=True)

    return _pp.tabular.load_dataset(
        reference=ref_df,
        test=test_df,
        label_column="body_mass_g",
        id_column="sample_id",
        feature_descriptions=pen_dir / "feature_descriptions.json",
        background=f"@{pen_dir / 'background.txt'}",
    )


# ── BSLT (SMILES → regression) ───────────────────────────────────


def load_bslt_regression(
    *,
    descriptors: bool = True,
    maccs: bool = True,
    morgan: bool = False,
):
    """Load the BSLT (brine-shrimp lethality) regression dataset.

    Returns a :class:`~molactagent.pp.chemistry.FeatureBundle` —
    SMILES are featurised on the fly via
    :func:`molactagent.pp.smiles_to_features`. Requires RDKit
    (``pip install molactagent[chem]``).

    The default feature set (26 RDKit descriptors + 167-bit MACCS
    block, variance-filtered) matches every cached benchmark in the
    project's research output. Pass ``morgan=True`` to additionally
    append a Morgan / ECFP fingerprint block.

    The bundled background string is
    :data:`molactagent.pp.chemistry.BSLT_BACKGROUND` — pass it into
    your estimator's ``background=`` argument when fitting so the
    LLM gets the right domain context.

    File search order:

    1. ``examples/bslt/bslt.csv`` — the canonical user-facing copy
       that ships in the repository.
    2. ``data/BSLT_huigui.csv`` — the legacy research-output copy
       (for back-compat with checkouts that still have it).
    """
    from .. import pp as _pp

    root = _project_root()
    candidates = [
        root / "examples" / "bslt" / "bslt.csv",
        root / "data" / "BSLT_huigui.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(_missing_dataset_msg(
            candidates[0],
            "The BSLT dataset ships at `examples/bslt/bslt.csv`. "
            "It is bundled with the source repository but not with "
            "wheel installs.",
        ))
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "standard_value" not in df.columns:
        raise ValueError(
            f"BSLT CSV must have 'smiles' and 'standard_value' "
            f"columns; got {list(df.columns)!r}."
        )
    return _pp.smiles_to_features(
        df["smiles"], y=df["standard_value"],
        descriptors=descriptors, maccs=maccs, morgan=morgan,
    )
