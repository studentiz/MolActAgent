"""Preprocessing — the Scanpy-style ``ma.pp`` namespace.

Two families of helpers live here:

- **Chemistry** (:mod:`chemistry`, :mod:`descriptors`,
  :mod:`fingerprints`, :mod:`scaffold`) — convert SMILES strings
  into MolActAgent-ready feature matrices, plus the curated
  RDKit-descriptor catalogue and the 167-key MACCS dictionary.
  Requires the optional ``rdkit`` dependency. Import any of these
  modules without RDKit installed and you will get a clear
  ``ImportError`` pointing at ``pip install molactagent[chem]``.

- **Tabular** (:mod:`tabular`) — a thin convenience wrapper that
  combines the existing :mod:`molactagent.io` validators and
  preparators into a single one-call entry point for the common
  numeric-CSV case.

The split is intentional: a non-chemistry user importing
``ma.pp.tabular`` should not pay the RDKit import cost; a
chemistry user importing ``ma.pp.chemistry`` should not have to
remember which scripts to copy from.
"""

from . import chemistry, descriptors, fingerprints, scaffold, tabular

__all__ = [
    "chemistry",
    "descriptors",
    "fingerprints",
    "scaffold",
    "tabular",
    # Re-exports for the most-used helpers (so ``from
    # molactagent.pp import smiles_to_features, scaffold_split``
    # works without poking into submodules).
    "smiles_to_features",
    "scaffold_split",
    "load_dataset",
]


def __getattr__(name: str):
    """Lazy re-exports — keeps `import molactagent.pp` cheap when
    only a subset of submodules is touched."""
    if name == "smiles_to_features":
        from .chemistry import smiles_to_features
        return smiles_to_features
    if name == "scaffold_split":
        from .scaffold import scaffold_split
        return scaffold_split
    if name == "load_dataset":
        from .tabular import load_dataset
        return load_dataset
    raise AttributeError(f"module 'molactagent.pp' has no attribute {name!r}")
