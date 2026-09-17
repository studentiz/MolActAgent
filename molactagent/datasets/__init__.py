"""Bundled example datasets.

Thin accessors over the on-disk fixtures in ``examples/``. Each
loader returns a :class:`~molactagent.pp.tabular.TabularBundle` (or a
:class:`~molactagent.pp.chemistry.FeatureBundle` for SMILES-based
datasets), so a user can go from "give me an example" to
"feed it into the estimator" in one line.

These do not duplicate any data: every dataset reads the canonical
files under ``examples/`` (or, for the BSLT chemistry dataset,
``data/BSLT_huigui.csv``). Loaders raise ``FileNotFoundError`` with
a helpful message when a missing dataset needs to be built first
(e.g. via ``scripts/prepare_penguins_subsample.py``).
"""

from ._loaders import (
    load_bslt_regression,
    load_iris_flowers,
    load_penguins,
)

__all__ = [
    "load_iris_flowers",
    "load_penguins",
    "load_bslt_regression",
]
