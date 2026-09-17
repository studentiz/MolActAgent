"""Tests for ``molactagent.pp`` — chemistry + tabular preprocessing.

Chemistry tests are auto-skipped when RDKit is not installed; the
tabular tests rely only on the existing IO layer and always run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import molactagent as ma

# RDKit availability is stable for the lifetime of the test
# session; check once.
try:
    import rdkit  # noqa: F401
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

requires_rdkit = pytest.mark.skipif(
    not _HAS_RDKIT, reason="RDKit not installed",
)


# ── descriptors module ─────────────────────────────────────────────


def test_descriptors_catalog_has_26_entries():
    """Catalogue size must stay at 26 — adding a new descriptor
    silently would fork the canonical X column order."""
    from molactagent.pp import descriptors as d
    assert len(d.list_available()) == 26
    assert "MolWt" in d.list_available()
    assert d.describe("MolLogP").startswith("Wildman-Crippen LogP")


def test_descriptors_describe_unknown_raises():
    from molactagent.pp import descriptors as d
    with pytest.raises(KeyError):
        d.describe("NotAReal_Descriptor_XYZ")


@requires_rdkit
def test_descriptors_compute_for_aspirin():
    """Aspirin should give plausible MolWt."""
    from rdkit import Chem
    from molactagent.pp import descriptors as d
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    out = d.compute(mol)
    assert "MolWt" in out
    assert 100 < out["MolWt"] < 200, (
        f"Aspirin MolWt should be ~180; got {out['MolWt']}"
    )


# ── fingerprints module ───────────────────────────────────────────


def test_maccs_keys_dict_has_167_entries():
    """MACCS dictionary must cover indices 0-166."""
    from molactagent.pp import fingerprints as f
    assert len(f.MACCS_KEY_DESCRIPTIONS) == 167
    assert 0 in f.MACCS_KEY_DESCRIPTIONS
    assert 166 in f.MACCS_KEY_DESCRIPTIONS
    assert f.describe_maccs_key(108) == "Hydroxyl group (OH)"


def test_maccs_describe_out_of_range_raises():
    from molactagent.pp import fingerprints as f
    with pytest.raises(KeyError):
        f.describe_maccs_key(200)


@requires_rdkit
def test_maccs_extractor_returns_167_bits():
    from rdkit import Chem
    from molactagent.pp import fingerprints as f
    mol = Chem.MolFromSmiles("CCO")  # ethanol
    fp = f.maccs(mol)
    assert fp.shape == (167,)
    assert fp.dtype == np.int8


@requires_rdkit
def test_morgan_extractor_returns_n_bits():
    from rdkit import Chem
    from molactagent.pp import fingerprints as f
    mol = Chem.MolFromSmiles("CCO")
    fp = f.morgan(mol, radius=2, n_bits=512)
    assert fp.shape == (512,)


# ── chemistry main entry point ────────────────────────────────────


@requires_rdkit
def test_smiles_to_features_basic():
    from molactagent.pp import smiles_to_features

    smiles = ["CCO", "CC(=O)O", "c1ccccc1"]  # ethanol, acetic acid, benzene
    bundle = smiles_to_features(smiles, descriptors=True, maccs=True)
    assert bundle.X.shape[0] == 3
    # 26 descriptors + at least one variant MACCS bit.
    assert bundle.X.shape[1] >= 27
    assert len(bundle.feature_names) == bundle.X.shape[1]
    assert "MolWt" in bundle.feature_names
    assert "MolWt" in bundle.feature_descriptions
    assert all(name in bundle.feature_descriptions for name in bundle.feature_names)
    assert bundle.metadata["n_descriptors"] == 26


@requires_rdkit
def test_smiles_to_features_with_y():
    from molactagent.pp import smiles_to_features
    smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    y = ["a", "b", "a"]
    bundle = smiles_to_features(smiles, y=y, descriptors=True, maccs=True)
    assert bundle.y is not None
    assert list(bundle.y) == y


@requires_rdkit
def test_smiles_to_features_skips_invalid():
    from molactagent.pp import smiles_to_features
    smiles = ["CCO", "this-is-garbage", "CC(=O)O"]
    bundle = smiles_to_features(
        smiles, descriptors=True, maccs=False, skip_invalid=True,
    )
    assert bundle.metadata["n_invalid_smiles"] == 1
    assert bundle.metadata["n_kept_smiles"] == 2
    assert "this-is-garbage" not in bundle.smiles


@requires_rdkit
def test_smiles_to_features_morgan_optional():
    from molactagent.pp import smiles_to_features
    smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    bundle = smiles_to_features(
        smiles,
        descriptors=False, maccs=False,
        morgan=True, morgan_n_bits=128,
    )
    # Morgan-only path; column count is at most 128 (variance
    # filter may drop some).
    assert bundle.X.shape[1] <= 128
    assert bundle.metadata["n_morgan"] == bundle.X.shape[1]


@requires_rdkit
def test_smiles_to_features_to_dataframe_round_trip():
    from molactagent.pp import smiles_to_features
    bundle = smiles_to_features(
        ["CCO", "CC(=O)O"], y=[1.0, 2.0],
        descriptors=True, maccs=False,
    )
    df = bundle.to_dataframe()
    assert "smiles" in df.columns
    assert "MolWt" in df.columns
    assert "target" in df.columns
    assert len(df) == 2


@requires_rdkit
def test_smiles_to_features_drops_invariant():
    """Drop-invariant should remove constant MACCS bits."""
    from molactagent.pp import smiles_to_features
    smiles = ["CCO", "CCO"]  # identical → all MACCS columns constant
    bundle = smiles_to_features(
        smiles, descriptors=False, maccs=True,
        drop_invariant=True,
    )
    # All MACCS bits identical → variance filter drops everything.
    # The metadata reflects that we asked for MACCS but every bit
    # was invariant.
    assert bundle.X.shape[1] == 0


@requires_rdkit
def test_smiles_to_features_requires_at_least_one_block():
    from molactagent.pp import smiles_to_features
    with pytest.raises(ValueError, match="At least one"):
        smiles_to_features(
            ["CCO"], descriptors=False, maccs=False, morgan=False,
        )


# ── scaffold split ───────────────────────────────────────────────


@requires_rdkit
def test_scaffold_split_basic():
    from molactagent.pp import scaffold_split
    # Two scaffold groups: benzene-derived vs acetone-derived.
    smiles = [
        "c1ccccc1",  # benzene
        "Cc1ccccc1",  # toluene
        "Oc1ccccc1",  # phenol — same Murcko scaffold as benzene
        "CC(=O)C",  # acetone
        "CCC(=O)C",  # 2-pentanone
    ]
    split = scaffold_split(smiles, test_size=0.4, random_state=0)
    # Indices partition every row exactly once.
    union = set(split.train_idx.tolist()) | set(split.test_idx.tolist())
    assert union == set(range(len(smiles)))
    # Scaffold-disjoint: no scaffold appears in both.
    assert not (
        set(split.train_scaffolds) & set(split.test_scaffolds)
    )


@requires_rdkit
def test_scaffold_split_seeded_deterministic():
    from molactagent.pp import scaffold_split
    smiles = ["c1ccccc1", "CCO", "CC(=O)O", "Cc1ccccc1", "CCC"]
    a = scaffold_split(smiles, test_size=0.4, random_state=42)
    b = scaffold_split(smiles, test_size=0.4, random_state=42)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert np.array_equal(a.test_idx, b.test_idx)


# ── tabular load_dataset ─────────────────────────────────────────


def test_load_dataset_iris(tmp_workspace, project_root):
    """Smoke-test the tabular convenience wrapper on the bundled
    iris example."""
    iris_dir = project_root / "examples" / "iris_flowers"
    bundle = ma.pp.load_dataset(
        reference=iris_dir / "reference.csv",
        test=iris_dir / "test.csv",
        label_column="species",
        id_column="flower_id",
        feature_descriptions=iris_dir / "feature_descriptions.json",
        background=f"@{iris_dir / 'background.txt'}",
    )
    assert bundle.task == "classification"
    assert bundle.X_train.shape[0] > 0
    assert bundle.X_test.shape[0] > 0
    assert "sepal_length_cm" in bundle.feature_names
    assert bundle.background.startswith("We are studying")
    assert set(bundle.class_labels).issuperset({"setosa", "versicolor", "virginica"})


def test_load_dataset_validation_failure_raises(tmp_workspace, simple_reference):
    """Bad inputs raise ``ValueError`` by default."""
    bad_test = simple_reference.iloc[:0]  # empty test frame
    with pytest.raises(Exception):
        ma.pp.load_dataset(
            reference=simple_reference,
            test=bad_test,
            label_column="activity",
            id_column=None,
        )


# ── lazy import sanity ──────────────────────────────────────────


def test_pp_namespace_lazy_attrs():
    """``ma.pp.smiles_to_features`` must resolve through the lazy
    re-export."""
    func = ma.pp.smiles_to_features
    assert callable(func)


def test_top_level_namespace_aliases():
    """``ma.tools`` is an alias for ``ma.tl``. ``ma.preprocessing``
    is intentionally NOT aliased to ``ma.pp`` — the package has an
    internal ``preprocessing`` submodule that the alias would
    shadow."""
    assert ma.tl is ma.tools


def test_preprocessing_legacy_module_still_works():
    """The legacy internal ``molactagent.preprocessing`` module —
    used by ``molactagent.agent`` — continues to expose
    ``DataPreprocessor``."""
    from molactagent.preprocessing import DataPreprocessor
    pre = DataPreprocessor("standard")
    assert pre.scaler_type == "standard"


def test_top_level_namespace_resolution_in_fresh_process():
    """Regression test: ``ma.pp`` must resolve from a fresh interpreter.

    The Scanpy-style namespace lookup goes through
    ``molactagent.__init__.__getattr__``. An earlier implementation
    used ``from . import pp`` which recursively re-triggered
    ``__getattr__`` and crashed. We now use ``importlib.import_module``
    to bypass that loop.

    Pytest hides the bug because test collection imports the
    sub-packages directly, populating ``sys.modules`` before any
    ``__getattr__`` is called. This test exercises the cold-start
    code path explicitly.
    """
    import subprocess
    import sys

    # Run a tiny one-shot script in a fresh interpreter so we hit
    # the cold-start lazy-import path. The script touches every
    # Scanpy-style namespace.
    code = (
        "import molactagent as ma; "
        "_ = ma.pp; _ = ma.tl; _ = ma.tools; "
        "_ = ma.metrics; _ = ma.datasets; "
        "print('ok')"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code], text=True, timeout=30,
    ).strip()
    assert out == "ok"
