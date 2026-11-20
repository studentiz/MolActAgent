"""Molecule records shared by retrieval and evidence extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .pp.descriptors import compute

CLASSES = ("active", "inactive")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass
class MoleculeRecord:
    row: int
    canonical: str
    label: str
    mol: Chem.Mol
    scaffold: str
    descriptors: dict[str, float]
    morgan: Any
    maccs_bits: set[int]
    constitution: str
    stereo_complete: bool
    pchembl: float | None = None


def make_record(smiles, row, identifier, label="", pchembl=None):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Missing SMILES at {identifier!r}.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError(f"Invalid SMILES at {identifier!r}.")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    # RDKit's non-isomeric SMILES also drops isotope labels.
    without_stereo = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(without_stereo)
    stereo = Chem.FindPotentialStereo(molecule)
    return MoleculeRecord(
        row=row,
        canonical=Chem.MolToSmiles(molecule),
        label=label,
        mol=molecule,
        scaffold=Chem.MolToSmiles(scaffold),
        descriptors=compute(molecule),
        morgan=_MORGAN.GetFingerprint(molecule),
        maccs_bits=set(MACCSkeys.GenMACCSKeys(molecule).GetOnBits()),
        constitution=Chem.MolToSmiles(without_stereo),
        stereo_complete=bool(stereo) and all(
            info.specified == Chem.StereoSpecified.Specified for info in stereo
        ),
        pchembl=pchembl,
    )


def specified_stereoisomers(left, right):
    """Whether two different canonical structures differ in assigned stereo."""
    return (
        left.stereo_complete and right.stereo_complete
        and left.constitution == right.constitution
        and left.canonical != right.canonical
    )


def tanimoto(left, right):
    return float(DataStructs.TanimotoSimilarity(left, right))


def format_value(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.1f}" if abs(value) >= 1000 else f"{value:.{digits}f}"
    return str(value)


def parse_model_json(raw):
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        first_line, separator, body = text.partition("\n")
        if separator and first_line.lower() in {"```", "```json"} and body.endswith("```"):
            text = body[:-3].strip()
    decoder = json.JSONDecoder(parse_constant=lambda _: None)
    try:
        value = decoder.decode(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
