"""
Convert SMILES → molecular descriptor + MACCS fingerprint feature matrix.

Outputs saved to  data/processed/  :
    features.npz   – X (n_samples, n_features) and y (n_samples,)
    metadata.json  – feature_names, feature_descriptions, background
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── molecular descriptors (interpretable for LLM reasoning) ─────────

DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "LabuteASA": Descriptors.LabuteASA,
    "BertzCT": Descriptors.BertzCT,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "Chi0v": Descriptors.Chi0v,
    "Chi1v": Descriptors.Chi1v,
    "MaxEStateIndex": Descriptors.MaxEStateIndex,
    "MinEStateIndex": Descriptors.MinEStateIndex,
    "qed": Descriptors.qed,
    "NHOHCount": Descriptors.NHOHCount,
    "NOCount": Descriptors.NOCount,
}

DESCRIPTOR_DESCRIPTIONS = {
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

# ── MACCS 166 key descriptions (MDL / Symyx public keys) ────────────
# Key 0 is always 0 (padding). Keys 1–166 map to structural patterns.

MACCS_KEY_DESCRIPTIONS = {
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

# ── EGFR research background ────────────────────────────────────────

BACKGROUND = (
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

# ────────────────────────────────────────────────────────────────────


def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def compute_descriptors(mol) -> dict[str, float]:
    vals = {}
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            v = func(mol)
            vals[name] = float(v) if v is not None else np.nan
        except Exception:
            vals[name] = np.nan
    return vals


def compute_maccs(mol) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.int8)


def compute_morgan(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan (circular / ECFP-like) bit fingerprint as float vector."""
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def resolve_label_column(df: pd.DataFrame) -> str:
    for col in ("class_label", "class_lable"):
        if col in df.columns:
            return col
    raise ValueError(
        "Expected a label column named 'class_label' or 'class_lable' (typo in source CSV)."
    )


def build_dataset_from_dataframe(
    df: pd.DataFrame,
    background: str,
    verbose: bool = True,
    add_morgan: bool = False,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build X, y, and metadata dict from an evidence table (SMILES + active/inactive).

    Default: RDKit descriptors + MACCS (non-zero variance bits), same as
    ``data/processed/features.npz``. Optionally append Morgan fingerprint bits
    (also variance-filtered).
    """
    if "smiles" not in df.columns:
        raise ValueError("DataFrame must contain a 'smiles' column")

    label_col = resolve_label_column(df)
    # Multi-class MolActAgent keeps the user's domain class names as-is
    # (e.g. ``"active"`` / ``"inactive"``) — no internal Pos/Neg remap.
    # We only enforce that every row has a non-null label string.
    nan_mask = df[label_col].isna()
    if nan_mask.any():
        bad_rows = df.loc[nan_mask].index.tolist()
        raise ValueError(
            f"Missing values in {label_col} on rows: {bad_rows}"
        )
    y_raw = df[label_col].astype(str)

    desc_rows = []
    maccs_rows = []
    morgan_rows: list[np.ndarray] = []
    valid_idx = []

    for i, smi in enumerate(df["smiles"]):
        try:
            mol = smiles_to_mol(smi)
        except ValueError as e:
            if verbose:
                print(f"  WARNING: skipping row {i} – {e}")
            continue
        desc_rows.append(compute_descriptors(mol))
        maccs_rows.append(compute_maccs(mol))
        if add_morgan:
            morgan_rows.append(compute_morgan(mol, morgan_radius, morgan_n_bits))
        valid_idx.append(i)

    if not valid_idx:
        raise ValueError("No valid SMILES rows after parsing")

    desc_names = list(DESCRIPTOR_FUNCS.keys())
    X_desc = np.array([[row[n] for n in desc_names] for row in desc_rows], dtype=np.float64)
    X_maccs = np.vstack(maccs_rows).astype(np.float64)
    y = y_raw.iloc[valid_idx].values

    maccs_var = X_maccs.var(axis=0)
    keep_maccs = np.where(maccs_var > 0)[0]
    X_maccs = X_maccs[:, keep_maccs]
    maccs_names = [f"MACCS_{k}" for k in keep_maccs]
    if verbose:
        print(f"  MACCS keys retained (non-zero variance): {len(keep_maccs)} / 167")

    blocks: list[np.ndarray] = [X_desc, X_maccs]
    feature_names = desc_names + maccs_names
    n_morgan_kept = 0
    if add_morgan:
        X_morg = np.vstack(morgan_rows).astype(np.float64)
        morg_var = X_morg.var(axis=0)
        keep_morg = np.where(morg_var > 0)[0]
        X_morg = X_morg[:, keep_morg]
        morgan_names = [f"Morgan_r{morgan_radius}_{k}" for k in keep_morg]
        n_morgan_kept = len(keep_morg)
        blocks.append(X_morg)
        feature_names = feature_names + morgan_names
        if verbose:
            print(
                f"  Morgan bits retained (non-zero variance): {n_morgan_kept} / {morgan_n_bits} "
                f"(radius={morgan_radius})"
            )

    X = np.hstack(blocks)
    if verbose:
        extra = f" + {n_morgan_kept} Morgan" if add_morgan else ""
        print(
            f"  Total features: {X.shape[1]}  "
            f"({len(desc_names)} descriptors + {len(maccs_names)} MACCS{extra})"
        )

    feature_descriptions: dict[str, str] = {}
    for name in desc_names:
        feature_descriptions[name] = DESCRIPTOR_DESCRIPTIONS[name]
    for k in keep_maccs:
        feature_descriptions[f"MACCS_{k}"] = MACCS_KEY_DESCRIPTIONS.get(
            k, f"MACCS structural key {k}"
        )
    if add_morgan:
        for nm in morgan_names:
            feature_descriptions[nm] = (
                f"Morgan circular fingerprint bit (radius={morgan_radius}, "
                f"ECFP-like; bit index in full {morgan_n_bits}-dim vector)"
            )

    smiles_list = [df["smiles"].iloc[i] for i in valid_idx]
    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(core))
        except Exception:
            scaffolds.append("")

    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    scaffold_ids = [scaffold_to_id[s] for s in scaffolds]

    if verbose:
        print(f"  Pos: {(y == 'Pos').sum()},  Neg: {(y == 'Neg').sum()}")
        print(f"  Unique Murcko scaffolds: {len(unique_scaffolds)}")

    meta = {
        "feature_names": feature_names,
        "feature_descriptions": feature_descriptions,
        "background": background,
        "smiles": smiles_list,
        "original_labels": [df[label_col].iloc[i] for i in valid_idx],
        "label_column": label_col,
        "n_descriptors": len(desc_names),
        "n_maccs": len(maccs_names),
        "add_morgan": add_morgan,
        "morgan_radius": morgan_radius if add_morgan else None,
        "morgan_n_bits": morgan_n_bits if add_morgan else None,
        "n_morgan": n_morgan_kept if add_morgan else 0,
        "scaffolds": scaffolds,
        "scaffold_ids": scaffold_ids,
        "unique_scaffolds": unique_scaffolds,
    }
    return X, y, meta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input CSV (default: data/tmp_test_data.csv)")
    parser.add_argument("--background", type=str, default=None,
                        help="Research background text (default: EGFR)")
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).resolve().parent.parent
    csv_path = pathlib.Path(args.input) if args.input else project_root / "data" / "tmp_test_data.csv"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = args.background if args.background else BACKGROUND

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} compounds from {csv_path}")

    X, y, meta = build_dataset_from_dataframe(df, bg, verbose=True)
    feature_names = meta["feature_names"]
    feature_descriptions = meta["feature_descriptions"]
    smiles_list = meta["smiles"]
    scaffolds = meta["scaffolds"]
    scaffold_ids = meta["scaffold_ids"]
    unique_scaffolds = meta["unique_scaffolds"]

    if len(unique_scaffolds) <= 30:
        for sid, smi in enumerate(unique_scaffolds):
            members = [i for i, x in enumerate(scaffold_ids) if x == sid]
            labels_in = [y[i] for i in members]
            print(f"    scaffold {sid} ({smi[:50]}): n={len(members)} "
                  f"Pos={labels_in.count('Pos')} Neg={labels_in.count('Neg')}")

    np.savez(out_dir / "features.npz", X=X, y=y)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "feature_descriptions": feature_descriptions,
                "background": bg,
                "smiles": smiles_list,
                "original_labels": meta["original_labels"],
                "n_descriptors": meta["n_descriptors"],
                "n_maccs": meta["n_maccs"],
                "scaffolds": scaffolds,
                "scaffold_ids": scaffold_ids,
                "unique_scaffolds": unique_scaffolds,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved to {out_dir}:")
    print(f"  features.npz  ({X.shape[0]} × {X.shape[1]})")
    print(f"  metadata.json (with scaffold info)")


if __name__ == "__main__":
    main()
