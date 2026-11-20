#!/usr/bin/env python3
"""Generate aligned, full-structure RDKit SVGs and verified focus crops."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads((ROOT / "data" / "evidence.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "public" / "molecules"
DRAW_WIDTH = 1400
DRAW_HEIGHT = 600


def build_molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {smiles}")
    return molecule


def aligned_molecules(compounds: list[dict]) -> dict[str, Chem.Mol]:
    template = build_molecule(next(item["smiles"] for item in compounds if item["id"] == "R2"))
    AllChem.Compute2DCoords(template, canonOrient=True, clearConfs=True)
    molecules: dict[str, Chem.Mol] = {}

    for item in compounds:
        molecule = build_molecule(item["smiles"])
        if item["id"] == "R2":
            molecules[item["id"]] = template
            continue
        if item["id"] == "R3":
            AllChem.Compute2DCoords(molecule, canonOrient=True, clearConfs=True)
            molecules[item["id"]] = molecule
            continue

        mcs = rdFMCS.FindMCS(
            [template, molecule],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            timeout=5,
        )
        reference_pattern = Chem.MolFromSmarts(mcs.smartsString)
        AllChem.GenerateDepictionMatching2DStructure(
            molecule,
            template,
            refPatt=reference_pattern,
            acceptFailure=False,
        )
        molecules[item["id"]] = molecule

    return molecules


def target_atom_indices(molecule: Chem.Mol, smarts: str | None) -> list[int]:
    if not smarts:
        return []
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"Invalid highlight SMARTS: {smarts}")
    matches = molecule.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        raise ValueError(f"Highlight SMARTS {smarts!r} did not match {Chem.MolToSmiles(molecule)}")
    return sorted({index for match in matches for index in match})


def draw_structure(
    molecule: Chem.Mol,
    atom_indices: list[int],
    highlight_color: tuple[float, float, float],
) -> tuple[str, dict[int, tuple[float, float]]]:
    drawer = rdMolDraw2D.MolDraw2DSVG(DRAW_WIDTH, DRAW_HEIGHT)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.useBWAtomPalette()
    options.padding = 0.022
    options.bondLineWidth = 3.0
    options.multipleBondOffset = 0.15
    options.minFontSize = 28
    options.maxFontSize = 34
    options.fixedFontSize = 32
    options.highlightRadius = 0.42
    options.additionalAtomLabelPadding = 0.04
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=atom_indices,
        highlightAtomColors={index: highlight_color for index in atom_indices},
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    coordinates = {
        index: (drawer.GetDrawCoords(index).x, drawer.GetDrawCoords(index).y)
        for index in atom_indices
    }
    return drawer.GetDrawingText(), coordinates


def focus_bounds(coordinates: dict[int, tuple[float, float]]) -> tuple[float, float, float, float]:
    if not coordinates:
        return (0.0, 0.0, float(DRAW_WIDTH), float(DRAW_HEIGHT))
    points = list(coordinates.values())
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    # Keep the terminal ring, its attachment bond, and the highlighted atom
    # together. A tighter crop makes Cl/OH/F labels readable at release sizes.
    width = max(300.0, max_x - min_x + 210.0)
    height = max(280.0, max_y - min_y + 180.0)
    left = min(max(0.0, center_x - width / 2), DRAW_WIDTH - width)
    top = min(max(0.0, center_y - height / 2), DRAW_HEIGHT - height)
    return (left, top, width, height)


def with_view_box(svg: str, bounds: tuple[float, float, float, float]) -> str:
    left, top, width, height = bounds
    output_width = 800
    output_height = round(output_width * height / width)
    svg = re.sub(r"width=['\"]1400px['\"]", f"width='{output_width}px'", svg, count=1)
    svg = re.sub(r"height=['\"]600px['\"]", f"height='{output_height}px'", svg, count=1)
    svg = re.sub(
        r"viewBox=['\"]0 0 1400 600['\"]",
        f"viewBox='{left:.2f} {top:.2f} {width:.2f} {height:.2f}'",
        svg,
        count=1,
    )
    svg = svg.replace("font-family:sans-serif", "font-family:'DM Sans',sans-serif")
    return svg


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    compounds = [
        *EVIDENCE["references"],
        *EVIDENCE["queries"],
    ]
    molecules = aligned_molecules(compounds)
    manifest = {
        "drawWidth": DRAW_WIDTH,
        "drawHeight": DRAW_HEIGHT,
        "rdkitVersion": Chem.rdBase.rdkitVersion,
        "compounds": {},
    }

    for compound in compounds:
        molecule = molecules[compound["id"]]
        target_indices = target_atom_indices(molecule, compound.get("highlightSmarts"))
        is_active = compound["label"] == "active"
        highlight_color = (0.86, 0.94, 0.88) if is_active else (0.97, 0.90, 0.84)
        full_svg, coordinates = draw_structure(molecule, target_indices, highlight_color)
        focus = focus_bounds(coordinates)
        focus_svg = with_view_box(full_svg, focus)

        (OUTPUT / f"{compound['id']}.svg").write_text(full_svg, encoding="utf-8")
        (OUTPUT / f"{compound['id']}-focus.svg").write_text(focus_svg, encoding="utf-8")
        manifest["compounds"][compound["id"]] = {
            "canonicalSmiles": Chem.MolToSmiles(molecule, canonical=True),
            "targetAtomIndices": target_indices,
            "targetAtomSymbols": [molecule.GetAtomWithIdx(index).GetSymbol() for index in target_indices],
            "targetDrawCoordinates": {
                str(index): [round(value, 3) for value in point]
                for index, point in coordinates.items()
            },
            "focusViewBox": [round(value, 3) for value in focus],
        }

    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(compounds)} full structures and focus crops to {OUTPUT}")


if __name__ == "__main__":
    main()
