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
    pchembl: float | None = None


def make_record(smiles, row, identifier, label="", pchembl=None):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Missing SMILES at {identifier!r}.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError(f"Invalid SMILES at {identifier!r}.")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return MoleculeRecord(
        row=row,
        canonical=Chem.MolToSmiles(molecule),
        label=label,
        mol=molecule,
        scaffold=Chem.MolToSmiles(scaffold),
        descriptors=compute(molecule),
        morgan=_MORGAN.GetFingerprint(molecule),
        maccs_bits=set(MACCSkeys.GenMACCSKeys(molecule).GetOnBits()),
        pchembl=pchembl,
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
