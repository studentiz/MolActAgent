"""MACCS and Morgan fingerprint extractors plus the curated MACCS
key dictionary.

The MACCS key descriptions (167 keys, indices 0-166) live as a
module-level constant so they remain accessible for
``feature_descriptions`` even when the user is dropping the
fingerprints into MolActAgent without going through the chemistry
batch loader. The dictionary is a verbatim port of
``MACCS_KEY_DESCRIPTIONS`` from ``scripts/prepare_features.py`` —
each entry is a short, plain-English structural-pattern note
written for LLM-readable prompts.

RDKit is imported lazily inside the extractor functions so that
``import molactagent.pp.fingerprints`` works without RDKit if all
the user wants is the description dictionary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── MACCS 167-key description dictionary (verbatim port) ───────────
#
# Index 0 is the padding bit (always 0); indices 1-166 map to MDL /
# Symyx public structural keys. We include 0 so callers indexing
# directly by RDKit's bitvector position (which also includes the
# padding slot) get coherent output.

MACCS_KEY_DESCRIPTIONS: dict[int, str] = {
    0: "Padding bit (always 0)",
    1: "Contains isotope atom",
    2: "Contains element with atomic number > 103",
    3: "Contains Ge, As, Se, Sn, Sb, Te, Pb, Bi, or Po",
    4: "Contains actinide element",
    5: "Contains Sc, Ti, Y, Zr, or Hf",
    6: "Contains lanthanide element",
    7: "Contains V, Cr, Mn, Nb, Mo, Tc, Re, or W",
    8: "Heteroatom in 4+ membered ring (QAAA@1)",
    9: "Contains Fe, Co, Ni, Ru, Rh, Pd, Os, Ir, or Pt",
    10: "Contains Cu, Zn, Ag, Cd, Au, or Hg",
    11: "ON(C)C fragment – nitrogen bonded to oxygen and two carbons",
    12: "S-S bond (disulfide)",
    13: "OC(O)O fragment – carbonate pattern",
    14: "Heteroatom in 3-membered ring (QAA@1)",
    15: "C≡C – carbon-carbon triple bond (alkyne)",
    16: "Contains alkaline-earth metal (Be, Mg, Ca, Sr, Ba, Ra)",
    17: "S=A – sulfur double bond to any atom",
    18: "Contains alkali metal (Li, Na, K, Rb, Cs, Fr)",
    19: "N-A-N – two nitrogens separated by one atom",
    20: "N-A-A – nitrogen connected through two-atom chain",
    21: "Triple bond attached to non-terminal atom",
    22: "O-A-A-O – two oxygens separated by two atoms",
    23: "S=A-A – sulfonyl / sulfoxide chain",
    24: "Halogen…A…A…A…halogen (two halogens separated by 3 atoms)",
    25: "Heteroatom in ring (QAAA@1, variant 2)",
    26: "Two adjacent heteroatoms (Q-Q)",
    27: "Tetravalent sulfur S(A)(A)(A)A",
    28: "O-A-A-A-O – two oxygens separated by three atoms",
    29: "All-carbon ring present",
    30: "Heteroatom in ring (variant 3)",
    31: "Oxygen linked through chain to ring (OAAAO)",
    32: "A-CH₂-O – methylene adjacent to oxygen",
    33: "Nitrogen atom in ring",
    34: "3-membered ring",
    35: "4-membered ring",
    36: "N-A-A-A-N – two nitrogens separated by three atoms",
    37: "A-CH₂-CH₂-A – ethylene bridge",
    38: "Adjacent heteroatoms (Q-Q, variant 2)",
    39: "A-CH₂-A – methylene bridge",
    40: "Branched heteroatom Q-A(A)-A",
    41: "SH – thiol group",
    42: "CH₃-A-CH₃ – geminal dimethyl",
    43: "O-A-A-A-O (variant 2)",
    44: "5-membered ring",
    45: "NH₂ – primary amine",
    46: "Aromatic nitrogen (N in aromatic system)",
    47: "OC(=O)N – carbamate / urethane",
    48: "CH₃-CH₃ – ethane fragment",
    49: "Non-ring C=C – alkene (not in ring)",
    50: "A-C≡N – attached nitrile",
    51: "A-CH₂-N – methylene next to nitrogen",
    52: "Q-SO – heteroatom bonded to sulfinyl / sulfonyl",
    53: "S connected to four atoms via single bonds",
    54: "N-C=O – amide bond",
    55: "O-C(C)-C – ether with branched carbon",
    56: "N-C(O)-O – carbamate nitrogen side",
    57: "N-C(C)-N – amidine / guanidine fragment",
    58: "O-C=O – ester or carboxyl group",
    59: "Oxygen atom in ring",
    60: "6-membered ring",
    61: "Aromatic ring present",
    62: "N≡C – nitrile or isonitrile",
    63: "N-C(=O)-N – urea fragment",
    64: "S-C(=O)-N – thioamide",
    65: "Heteroatom in 4-membered ring (variant 2)",
    66: "Multiple QH groups (≥2 OH / NH / SH)",
    67: "Multiple methyl groups (≥2 CH₃)",
    68: "Nitrogen in both ring and chain positions",
    69: "N-C(=O)-C – amide with adjacent carbon",
    70: "O-C(=O)-C – ester fragment",
    71: "Q-O bond – heteroatom bonded to oxygen",
    72: "Q-CH₃ – methyl on heteroatom",
    73: "N-CH₂-N – methylene between two nitrogens",
    74: "N-C(=O)-N – urea (variant 2)",
    75: "N-C-C(=O)-N – peptide-like linkage",
    76: "Q-O-CH₃ – methoxy on heteroatom",
    77: "≥2 chlorine atoms",
    78: "Q-A-A-A-Q – heteroatoms separated by 3 atoms",
    79: "CF₃ – trifluoromethyl group",
    80: "N-A-A-O – nitrogen and oxygen separated by two atoms",
    81: "N=C-C=N or N-C=C-N – conjugated N…N system",
    82: "N-S=O – sulfonamide fragment",
    83: "N-C(N)-N – triaminomethane pattern",
    84: "Fluorine atom present",
    85: "Aromatic OH / NH pair (QH on aromatic ring, ≥2)",
    86: "Miscellaneous / other pattern",
    87: "C=C-N – vinyl amine or enamine",
    88: "Bromine atom present",
    89: "S-A-N – sulfur-any-nitrogen chain",
    90: "O=C-N – reverse amide pattern",
    91: "Chain terminating in Q inside ring (AAAQ)",
    92: "Q-F – heteroatom bonded to fluorine",
    93: "C=C-O – vinyl ether / enol",
    94: "Chlorine atom present",
    95: "C-Br bond (bromine on carbon)",
    96: "N-N-O – hydrazine / nitrosamine fragment",
    97: "QH-CH₂-A – hydroxyl or amino next to methylene",
    98: "C=O-A-A-CH₂-A – carbonyl chain to methylene",
    99: "Branched heteroatom A-Q(A)-A",
    100: "≥2 chlorine atoms (variant 2)",
    101: "Q-O-C=O – heteroatom ester",
    102: "≥2 five-membered rings",
    103: "≥2 all-carbon rings",
    104: "Fluorine on aromatic ring",
    105: "Nitrogen in ring bonded to Q",
    106: "Q-A-N – heteroatom separated from nitrogen by one atom",
    107: "Nitrogen in aromatic ring",
    108: "Hydroxyl group (OH)",
    109: "Sulfur atom present",
    110: "O=C-O – carboxyl or ester (variant 2)",
    111: "C(=O)-CH₃ – acetyl / methyl ketone",
    112: "Q-CH₂-Q – methylene bridging two heteroatoms",
    113: "S=O – sulfinyl or sulfonyl group",
    114: "N(C)(C) – tertiary nitrogen with two carbon branches",
    115: "Heteroatom bonded to F (variant 2)",
    116: "8+ membered ring",
    117: "Q-C(=O)-N – heteroatom next to amide",
    118: "Long O-linked chain (O-A-A-A-O variant 3)",
    119: "Q=C(C)-C – heteroatom double-bonded carbon branch",
    120: "Q-C-N – heteroatom–carbon–nitrogen",
    121: "Q-N-Q – nitrogen bridging two heteroatoms",
    122: "N-O bond – hydroxylamine / N-oxide",
    123: "Aromatic NH₂ – primary amine on aromatic ring",
    124: "O-CH₂-CH₂-O – ethylene glycol fragment",
    125: "O=C-C=O – diketone / 1,2-dicarbonyl",
    126: "Q-A-A-Q – two heteroatoms separated by two atoms",
    127: "≥2 carbon rings (variant 2)",
    128: "CH₃-CH₂-A – ethyl group attached to any atom",
    129: "≥2 Q-Q pairs (adjacent heteroatom pairs)",
    130: "Aromatic QH…A…A…QH – two aromatic QH groups close",
    131: "Q-A-A-A-O – heteroatom to oxygen via three atoms",
    132: "Ester O-C=O (variant 2)",
    133: "A-N(A)-A – secondary / tertiary nitrogen",
    134: "Long methylene-bridged chain A-CH₂-A-A-A-CH₂-A",
    135: "Q-C(A)(A)-A – heteroatom on branched carbon",
    136: "≥3 rings in molecule",
    137: "Q-A(Q)-Q – multiply branched heteroatom center",
    138: "7-membered ring",
    139: "N-C(=O)-C – amide fragment (variant 2)",
    140: "C-O bond inside ring – cyclic ether / lactone",
    141: "Q-CHO – aldehyde attached to heteroatom",
    142: "8-membered ring (variant 2)",
    143: "N-Q bond – nitrogen bonded to heteroatom",
    144: "≥2 fused rings (bicyclic+)",
    145: "Methyl group on ring (ring-CH₃)",
    146: "Q-N bond – heteroatom bonded to nitrogen",
    147: "Branched nitrogen A-N(A)-A (variant 2)",
    148: "O-CH₃ – methoxy group",
    149: "Nitrogen in ring (general, any ring size)",
    150: "Oxygen in ring (general, any ring size)",
    151: "≥2 rings in molecule",
    152: "Q-Q bond (variant 3)",
    153: "N-A-A-A-O (variant 2)",
    154: "Q-A-A-A-Q (variant 2)",
    155: "Non-aromatic, non-ring C=C (isolated alkene)",
    156: "C=C inside ring (non-aromatic)",
    157: "≥2 heterocyclic rings",
    158: "Heteroatom present (any N, O, S, halogen, etc.)",
    159: "Q-C-O – heteroatom–carbon–oxygen",
    160: "Q-Q bond (variant 4)",
    161: "A-CH₂-A chain (variant 2)",
    162: "Q-CH₂-A – methylene adjacent to heteroatom chain",
    163: "≥2 aromatic rings",
    164: "At least one ring",
    165: "Aromatic ring (general presence)",
    166: "≥2 rings (general)",
}


def describe_maccs_key(idx: int) -> str:
    """Return a plain-English description for MACCS bit ``idx`` (0-166).

    Raises ``KeyError`` for out-of-range indices.
    """
    if idx not in MACCS_KEY_DESCRIPTIONS:
        raise KeyError(
            f"MACCS key {idx} out of range; valid indices are 0..166."
        )
    return MACCS_KEY_DESCRIPTIONS[idx]


# ── extractors (RDKit-backed) ─────────────────────────────────────


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs  # noqa: F401
        from rdkit.Chem import AllChem, MACCSkeys  # noqa: F401
        return Chem, DataStructs, AllChem, MACCSkeys
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "RDKit is required for fingerprint computation. Install with "
            "`pip install molactagent[chem]` (or "
            "`conda install -c conda-forge rdkit`)."
        ) from e


def maccs(mol: Any) -> np.ndarray:
    """Return the 167-bit MACCS key vector for an RDKit Mol.

    Indices follow the RDKit convention: position 0 is the padding
    bit, positions 1-166 are the MDL public keys. Output dtype is
    ``np.int8`` so the bitvector stays compact for downstream
    ``np.hstack`` / variance-filter pipelines.
    """
    _, _, _, MACCSkeys = _require_rdkit()
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.int8)


def morgan(mol: Any, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a Morgan (ECFP-like) circular fingerprint as a float vector.

    ``radius=2`` and ``n_bits=2048`` match the defaults used by
    every cached benchmark in ``data/`` so feature matrices remain
    interchangeable. Output dtype is ``np.float64`` (not ``int``)
    so the vector slots into the same downstream variance-filter +
    stack pipeline as the descriptor block.
    """
    _, DataStructs, AllChem, _ = _require_rdkit()
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def maccs_matrix(mols: list[Any]) -> np.ndarray:
    """Stack MACCS vectors into an ``(n, 167)`` float matrix.

    Cast to ``float64`` so the output composes cleanly with the
    descriptor / Morgan blocks during ``np.hstack``.
    """
    return np.vstack([maccs(m) for m in mols]).astype(np.float64)


def morgan_matrix(
    mols: list[Any], *, radius: int = 2, n_bits: int = 2048,
) -> np.ndarray:
    """Stack Morgan vectors into an ``(n, n_bits)`` matrix."""
    return np.vstack([morgan(m, radius=radius, n_bits=n_bits) for m in mols])
