"""SMILES → MolActAgent-ready feature matrix in one call.

This module is the user-facing chemistry entry point. It composes
:mod:`molactagent.pp.descriptors`, :mod:`molactagent.pp.fingerprints`,
and :mod:`molactagent.pp.scaffold` into a single
:func:`smiles_to_features` call that:

1. Parses every SMILES string into an RDKit Mol.
2. Computes the catalogued descriptor block (26 features).
3. Computes the MACCS bitvector (167 features).
4. Optionally computes a Morgan / ECFP-like fingerprint
   (``n_bits`` features, default 2048).
5. Drops zero-variance columns from the fingerprint blocks (the
   variance filter is necessary because most MACCS bits are
   uninformative on any single dataset; without it the LLM prompt
   bloats with constant columns).
6. Computes Murcko scaffolds for downstream scaffold-aware
   splitting.

Returns a :class:`FeatureBundle` carrying ``X``, ``y`` (when the
caller passed labels), feature names, the human-readable
``feature_descriptions`` dict MolActAgent's prompt builder consumes,
the original SMILES, and the scaffold metadata.

A small number of pre-curated domain-background strings (EGFR,
antiTB, BSLT) are exported as module constants — they're the
backgrounds used by the project's own benchmarks and are well-
suited to dropping into ``MolActAgent(background=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import descriptors as _descriptors
from . import fingerprints as _fingerprints
from . import scaffold as _scaffold


# ── pre-curated domain backgrounds (verbatim from scripts/) ─────────


EGFR_BACKGROUND = (
    "This study focuses on classifying small-molecule compounds targeting the "
    "Epidermal Growth Factor Receptor (EGFR). EGFR is a transmembrane tyrosine "
    "kinase receptor critically involved in cell proliferation, survival, and "
    "differentiation. Overexpression or activating mutations of EGFR are implicated "
    "in multiple cancers (non-small-cell lung cancer, colorectal cancer, etc.), "
    "making it a key therapeutic target.\n\n"
    "Classification task:\n"
    "  Pos (active)  – compound shows inhibitory activity against EGFR\n"
    "  Neg (inactive) – compound does not show significant EGFR inhibition\n\n"
    "Features include physicochemical molecular descriptors and MACCS structural "
    "fingerprints (binary indicators of specific chemical substructure patterns).\n\n"
    "Active EGFR inhibitors typically share:\n"
    "  - Quinazoline or pyrimidine core scaffolds that bind the ATP-binding pocket\n"
    "  - Hydrogen-bond acceptors/donors for kinase hinge-region interaction\n"
    "  - Hydrophobic substituents for binding-pocket complementarity\n"
    "  - Halogen substitution (F, Cl, Br) for potency and selectivity tuning"
)

ANTITB_BACKGROUND = (
    "Task: binary classification of cyclic peptide compounds for "
    "anti-tuberculosis (Mycobacterium tuberculosis) activity.\n"
    "Label semantics: Pos = exhibits anti-TB activity (active in "
    "literature/experiment); Neg = no significant activity (inactive).\n"
    "Features: RDKit molecular descriptors and MACCS structural "
    "fingerprints. For peptide-like compounds, drivers of anti-TB "
    "activity tend to involve membrane permeability (TPSA, MolLogP, "
    "MolWt), conformational rigidity (rotatable bonds, ring count), "
    "and specific structural motifs (MACCS bits)."
)

BSLT_BACKGROUND = (
    "Task: regression / classification of cyclic peptide and polyketide "
    "compounds for brine-shrimp (Artemia salina) lethality, a "
    "general toxicity screen.\n"
    "Target: bioactivity value (continuous regression target) OR "
    "binary toxicity label (Pos = positive in the lethality assay; "
    "Neg = negative or low toxicity).\n"
    "Features: RDKit molecular descriptors and MACCS fingerprints."
)


# ── feature bundle dataclass ────────────────────────────────────────


@dataclass
class FeatureBundle:
    """Container for the output of :func:`smiles_to_features`.

    Designed to drop directly into either the high-level pipeline
    (``run_classification(reference=df, ...)``) or the new
    estimator wrappers (``MolActAgentClassifier(feature_names=...,
    feature_descriptions=...).fit(bundle.X, bundle.y)``).
    """

    X: np.ndarray
    feature_names: list[str]
    feature_descriptions: dict[str, str]
    smiles: list[str]
    y: np.ndarray | None = None

    # Provenance — useful both for debugging and for downstream
    # scaffold-aware CV.
    scaffolds: list[str] = field(default_factory=list)
    scaffold_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame ``[smiles, *features, target?]`` for
        users who prefer the legacy CSV-shape API."""
        out = pd.DataFrame(self.X, columns=self.feature_names)
        out.insert(0, "smiles", self.smiles)
        if self.y is not None:
            out["target"] = self.y
        return out


# ── main entry point ────────────────────────────────────────────────


def smiles_to_features(
    smiles: Iterable[str] | pd.Series,
    *,
    y: Iterable[Any] | pd.Series | None = None,
    descriptors: bool = True,
    maccs: bool = True,
    morgan: bool = False,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    drop_invariant: bool = True,
    skip_invalid: bool = True,
    verbose: bool = False,
) -> FeatureBundle:
    """Convert SMILES (and optional labels) into a feature matrix.

    Parameters
    ----------
    smiles :
        Iterable of SMILES strings.
    y :
        Optional aligned labels (any dtype). Coerced to a numpy
        array; row order matches the surviving (parsed-OK) SMILES.
    descriptors :
        Include the 26-descriptor block.
    maccs :
        Include the 167-bit MACCS block.
    morgan :
        Append a Morgan fingerprint block. Off by default —
        Morgan + MACCS + descriptors gets prompt-bloaty quickly.
    morgan_radius :
        Morgan radius (2 = ECFP4-like; 3 = ECFP6-like).
    morgan_n_bits :
        Morgan bitvector length.
    drop_invariant :
        Drop fingerprint columns whose values are constant across
        the input dataset. Default ``True``: MolActAgent's variance
        filter would drop them anyway, and dropping early shrinks
        the prompt.
    skip_invalid :
        Skip SMILES that fail to parse (with a printed warning if
        ``verbose=True``). When ``False``, the first invalid SMILES
        raises ``ValueError``.
    verbose :
        Print per-row progress/warnings to stdout. Default off.

    Returns
    -------
    :class:`FeatureBundle`

    Notes
    -----
    The column order is **deterministic and append-only**:
    descriptor block first (in the catalogue's canonical order),
    then MACCS bits (in ascending bit index), then Morgan bits if
    requested. When ``descriptors=True, maccs=True`` (with or
    without ``morgan``), this matches the legacy ``X`` written by
    ``scripts/prepare_features.py`` byte-for-byte and therefore
    matches every cached benchmark in ``data/``. Setting any of
    the three flags to ``False`` simply drops that block — the
    surviving blocks keep their canonical positions, so swapping
    flags never reorders the remaining columns.
    """
    if not (descriptors or maccs or morgan):
        raise ValueError(
            "At least one of descriptors / maccs / morgan must be True."
        )

    # Parse SMILES → Mol.
    try:
        from rdkit import Chem
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "RDKit is required for SMILES featurisation. Install with "
            "`pip install molactagent[chem]` (or "
            "`conda install -c conda-forge rdkit`)."
        ) from e

    smiles_list = [str(s) for s in smiles]
    n_total = len(smiles_list)
    if n_total == 0:
        raise ValueError("`smiles` is empty.")
    y_array = None if y is None else np.asarray(list(y))
    if y_array is not None and len(y_array) != n_total:
        raise ValueError(
            f"`smiles` has {n_total} entries but `y` has {len(y_array)}."
        )

    mols: list[Any] = []
    valid_idx: list[int] = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            if skip_invalid:
                if verbose:
                    print(f"  [smiles_to_features] skipping invalid SMILES at row {i}: {smi!r}")
                continue
            raise ValueError(
                f"Invalid SMILES at row {i}: {smi!r}. Use "
                "`skip_invalid=True` to drop bad rows instead."
            )
        mols.append(mol)
        valid_idx.append(i)

    if not mols:
        raise ValueError(
            "Every SMILES row failed to parse; nothing to featurise."
        )

    surviving_smiles = [smiles_list[i] for i in valid_idx]
    surviving_y = None if y_array is None else y_array[valid_idx]

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_descriptions: dict[str, str] = {}
    n_descriptors = n_maccs = n_morgan = 0

    if descriptors:
        X_desc = _descriptors.compute_matrix(mols)
        desc_names = _descriptors.list_available()
        blocks.append(X_desc)
        feature_names.extend(desc_names)
        for name in desc_names:
            feature_descriptions[name] = _descriptors.DESCRIPTIONS[name]
        n_descriptors = len(desc_names)

    if maccs:
        X_maccs = _fingerprints.maccs_matrix(mols)
        kept = _filter_invariant(X_maccs) if drop_invariant else np.arange(X_maccs.shape[1])
        X_maccs = X_maccs[:, kept]
        maccs_names = [f"MACCS_{int(k)}" for k in kept]
        blocks.append(X_maccs)
        feature_names.extend(maccs_names)
        for k, name in zip(kept, maccs_names):
            feature_descriptions[name] = (
                _fingerprints.MACCS_KEY_DESCRIPTIONS.get(
                    int(k), f"MACCS structural key {int(k)}",
                )
            )
        n_maccs = len(maccs_names)
        if verbose:
            print(
                f"  [smiles_to_features] MACCS keys retained "
                f"(non-zero variance): {n_maccs} / 167"
            )

    if morgan:
        X_morg = _fingerprints.morgan_matrix(
            mols, radius=morgan_radius, n_bits=morgan_n_bits,
        )
        kept = _filter_invariant(X_morg) if drop_invariant else np.arange(X_morg.shape[1])
        X_morg = X_morg[:, kept]
        morgan_names = [f"Morgan_r{morgan_radius}_{int(k)}" for k in kept]
        blocks.append(X_morg)
        feature_names.extend(morgan_names)
        for name in morgan_names:
            feature_descriptions[name] = (
                f"Morgan circular fingerprint bit (radius={morgan_radius}, "
                f"ECFP-like; bit index in full {morgan_n_bits}-dim vector)"
            )
        n_morgan = len(morgan_names)
        if verbose:
            print(
                f"  [smiles_to_features] Morgan bits retained: "
                f"{n_morgan} / {morgan_n_bits} (radius={morgan_radius})"
            )

    X = np.hstack(blocks).astype(np.float64)

    # Scaffold computation — useful for downstream split / GroupKFold.
    scaffolds, scaffold_ids = _scaffold.compute_scaffolds(surviving_smiles)

    metadata: dict[str, Any] = {
        "n_descriptors": n_descriptors,
        "n_maccs": n_maccs,
        "n_morgan": n_morgan,
        "morgan_radius": morgan_radius if morgan else None,
        "morgan_n_bits": morgan_n_bits if morgan else None,
        "drop_invariant": bool(drop_invariant),
        "n_input_smiles": int(n_total),
        "n_kept_smiles": int(len(surviving_smiles)),
        "n_invalid_smiles": int(n_total - len(surviving_smiles)),
        "unique_scaffolds": sorted(set(scaffolds)),
    }

    if verbose:
        print(
            f"  [smiles_to_features] Total features: {X.shape[1]} "
            f"({n_descriptors} descriptors + {n_maccs} MACCS"
            + (f" + {n_morgan} Morgan)" if morgan else ")")
        )

    return FeatureBundle(
        X=X,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        smiles=surviving_smiles,
        y=surviving_y,
        scaffolds=scaffolds,
        scaffold_ids=scaffold_ids,
        metadata=metadata,
    )


# ── helpers ────────────────────────────────────────────────────────


def _filter_invariant(block: np.ndarray) -> np.ndarray:
    """Return the indices of columns with non-zero variance.

    Falls back to "keep everything" when the block has fewer than 2
    rows (variance is undefined / 0 for a singleton block).
    """
    if block.shape[0] < 2:
        return np.arange(block.shape[1])
    variances = block.var(axis=0)
    return np.where(variances > 0)[0]
