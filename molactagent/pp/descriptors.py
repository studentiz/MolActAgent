"""Curated RDKit molecular-descriptor catalogue.

Direct port of the ``DESCRIPTOR_FUNCS`` / ``DESCRIPTOR_DESCRIPTIONS``
constants from ``scripts/prepare_features.py``. The catalogue is the
26-descriptor selection that the project's compound-classification
benchmarks have used since the antiTB / BSLT runs.

Why these 26 specifically? They are LLM-interpretable: each has a
short, plain-English description that can be surfaced to the LLM
prompt builder so the model knows "MolLogP is lipophilicity" rather
than guessing from a numeric column header. That descriptions
dictionary is the curation layer; the function dictionary is just
the RDKit dispatch.

The module imports RDKit lazily inside :func:`compute` so users who
only want to inspect the catalogue (e.g. via :func:`describe`) do
not pay the RDKit import cost.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# ── descriptor catalogue ────────────────────────────────────────────


# The descriptions live as a module constant so users can introspect
# them without RDKit installed (``ca.pp.descriptors.DESCRIPTIONS``).
DESCRIPTIONS: dict[str, str] = {
    "MolWt": "Molecular weight (Da)",
    "MolLogP": "Wildman-Crippen LogP – lipophilicity (higher = more hydrophobic)",
    "TPSA": "Topological Polar Surface Area (Å²) – related to membrane permeability",
    "NumHDonors": "Number of hydrogen-bond donors",
    "NumHAcceptors": "Number of hydrogen-bond acceptors",
    "NumRotatableBonds": "Number of rotatable bonds – molecular flexibility",
    "RingCount": "Total number of rings",
    "NumAromaticRings": "Number of aromatic rings",
    "NumAliphaticRings": "Number of non-aromatic (aliphatic) rings",
    "FractionCSP3": "Fraction of sp3-hybridized carbons – 3D character / saturation",
    "HeavyAtomCount": "Number of non-hydrogen atoms",
    "NumHeteroatoms": "Number of heteroatoms (N, O, S, halogens, etc.)",
    "NumValenceElectrons": "Total number of valence electrons",
    "LabuteASA": "Labute approximate surface area (Å²)",
    "BertzCT": "Bertz topological complexity index",
    "HallKierAlpha": "Hall-Kier alpha – encodes molecular connectivity / branching",
    "Kappa1": "1st-order Kappa shape index (encodes overall molecular shape)",
    "Kappa2": "2nd-order Kappa shape index",
    "Kappa3": "3rd-order Kappa shape index",
    "Chi0v": "0th-order chi connectivity index (valence-weighted)",
    "Chi1v": "1st-order chi connectivity index (valence-weighted)",
    "MaxEStateIndex": "Maximum electrotopological state index – most electrophilic site",
    "MinEStateIndex": "Minimum electrotopological state index – most nucleophilic site",
    "qed": "Quantitative Estimate of Drug-likeness (0–1, higher = more drug-like)",
    "NHOHCount": "Count of NH and OH groups",
    "NOCount": "Count of nitrogen and oxygen atoms",
}


def list_available() -> list[str]:
    """Return the catalogued descriptor names in canonical order.

    Order matches the legacy ``DESCRIPTOR_FUNCS`` dict-iteration
    order so ``X`` matrices produced by the new chemistry module
    line up column-for-column with files written by
    ``scripts/prepare_features.py`` (and therefore with every
    cached benchmark in ``data/``).
    """
    return list(DESCRIPTIONS.keys())


def describe(name: str) -> str:
    """Return the human-readable description for ``name``.

    Raises ``KeyError`` for unknown descriptors. The catalogue is
    intentionally fixed; if you need a descriptor that is not
    listed here, compute it yourself with ``rdkit.Chem.Descriptors``
    and stitch it into the feature matrix manually — adding a new
    descriptor here forks the canonical ``X`` ordering.
    """
    if name not in DESCRIPTIONS:
        raise KeyError(
            f"Unknown descriptor {name!r}. Catalogued descriptors: "
            f"{list_available()}"
        )
    return DESCRIPTIONS[name]


# ── compute (RDKit-backed) ─────────────────────────────────────────


def _rdkit_descriptor_funcs() -> dict[str, Callable[[Any], float]]:
    """Resolve the RDKit ``Descriptors.*`` callables for our catalogue.

    Imported lazily so the module-level catalogue stays accessible
    without RDKit. Raises a friendly ``ImportError`` if RDKit is
    missing.
    """
    try:
        from rdkit.Chem import Descriptors
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "RDKit is required for descriptor computation. Install with "
            "`pip install molactagent[chem]` (or "
            "`conda install -c conda-forge rdkit`)."
        ) from e

    funcs: dict[str, Callable[[Any], float]] = {}
    for name in DESCRIPTIONS.keys():
        # ``getattr`` so any catalogue addition that's not a public
        # ``Descriptors`` attribute fails fast at this lookup, not
        # silently mid-loop.
        funcs[name] = getattr(Descriptors, name)
    return funcs


def compute(mol: Any) -> dict[str, float]:
    """Compute every catalogued descriptor for an RDKit Mol.

    Returns a dict ``{descriptor_name: float}``. Failures on a
    single descriptor become ``np.nan`` rather than raising — most
    benchmark CSVs have at least one molecule that triggers an
    edge case (charged species, exotic heteroatoms) on at least
    one descriptor. Mirrors the policy in
    ``scripts/prepare_features.compute_descriptors``.
    """
    funcs = _rdkit_descriptor_funcs()
    out: dict[str, float] = {}
    for name, func in funcs.items():
        try:
            v = func(mol)
            out[name] = float(v) if v is not None else float("nan")
        except Exception:
            out[name] = float("nan")
    return out


def compute_matrix(mols: list[Any]) -> np.ndarray:
    """Stack :func:`compute` over many molecules into ``(n, 26)``."""
    rows = [compute(m) for m in mols]
    names = list_available()
    return np.array(
        [[row[n] for n in names] for row in rows], dtype=np.float64,
    )
