#!/usr/bin/env python3
"""Cross-check the film's checked-in evidence against the release fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT.parent
FIXTURE = RELEASE_ROOT / "examples" / "cell-assay"
EVIDENCE = json.loads((ROOT / "data" / "evidence.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "public" / "molecules" / "manifest.json").read_text(encoding="utf-8"))
Q2_JUDGMENT = (
    "On the shared scaffold, a single para halogen on the terminal phenoxy ring "
    "distinguishes the active analogs from the inactive para-hydroxy and "
    "para-alkyl analogs."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> None:
    references = {row["id"]: row for row in read_csv(FIXTURE / "reference.csv")}
    queries = {row["id"]: row for row in read_csv(FIXTURE / "query.csv")}
    outcomes = {row["id"]: row for row in read_csv(FIXTURE / "outcomes.csv")}

    for item in EVIDENCE["references"]:
        row = references[item["id"]]
        assert_equal(item["smiles"], row["smiles"], f"{item['id']} SMILES")
        assert_equal(item["label"], row["label"], f"{item['id']} label")
        assert_equal(item["measurement"], row["measurement"], f"{item['id']} measurement")

    for item in EVIDENCE["queries"]:
        row = queries[item["id"]]
        assert_equal(item["smiles"], row["smiles"], f"{item['id']} SMILES")

    for item in EVIDENCE["outcomes"]:
        row = outcomes[item["id"]]
        assert_equal(item["measurement"], row["measurement"], f"{item['id']} outcome")
        assert_equal(item["label"], row["label"], f"{item['id']} outcome label")

    for compound_id in ("R2", "R5", "Q2"):
        compound = MANIFEST["compounds"][compound_id]
        expected_symbol = {"R2": ["Cl"], "R5": ["O"], "Q2": ["F"]}[compound_id]
        assert_equal(compound["targetAtomSymbols"], expected_symbol, f"{compound_id} highlighted atom")
        left, top, width, height = compound["focusViewBox"]
        for index, (x, y) in compound["targetDrawCoordinates"].items():
            if not (left + 45 <= x <= left + width - 45):
                raise AssertionError(f"{compound_id} atom {index} is too close to the horizontal crop edge")
            if not (top + 15 <= y <= top + height - 60):
                raise AssertionError(f"{compound_id} atom {index} is too close to the vertical crop edge")
        focus_svg = (ROOT / "public" / "molecules" / f"{compound_id}-focus.svg").read_text(encoding="utf-8")
        match = re.search(r"viewBox='([^']+)'", focus_svg)
        if match is None:
            raise AssertionError(f"{compound_id} focus SVG has no viewBox")
        rendered_view_box = [float(value) for value in match.group(1).split()]
        expected_view_box = compound["focusViewBox"]
        if any(abs(actual - expected) > 0.011 for actual, expected in zip(rendered_view_box, expected_view_box)):
            raise AssertionError(f"{compound_id} focus SVG does not match its manifest viewBox")

    prediction_path = Path(os.environ.get("SOURCE_PREDICTIONS", FIXTURE / "predictions.json"))
    assert_equal(sha256(prediction_path), EVIDENCE["assay"]["predictionRecordSha256"], "prediction hash")
    record = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = {item["identifier"]: item for item in record["predictions"]}
    for item in EVIDENCE["queries"]:
        assert_equal(item["label"], predictions[item["id"]]["label"], f"{item['id']} prediction")
    q2 = predictions["Q2"]
    assert_equal(q2["judgments"][0]["evidence"]["class_contrast"], Q2_JUDGMENT, "Q2 recorded judgment")
    reference_classes = {item["reference"]: item["label"] for item in q2["references"]}
    assert_equal(reference_classes["R2"], "active", "Q2 R2 comparison")
    assert_equal(reference_classes["R5"], "inactive", "Q2 R5 comparison")
    assert_equal(reference_classes["R7"], "active", "Q2 R7 comparison")

    q4 = predictions["Q4"]
    q4_reference_classes = {item["reference"]: item["label"] for item in q4["references"]}
    assert_equal(q4["smiles"], queries["Q4"]["smiles"], "Q4 prediction SMILES")
    assert_equal(q4_reference_classes["R4"], "active", "Q4 R4 comparison")
    assert_equal(references["R4"]["measurement"], "156 nM", "Q4 R4 comparator measurement")
    assert_equal(references["R4"]["label"], "active", "Q4 R4 comparator class")
    assert_equal(q4["evidence_references"][0]["reference"], "R4", "Q4 nearest comparator")
    assert_equal(q4["evidence_references"][0]["smiles"], references["R4"]["smiles"], "Q4 R4 comparator SMILES")

    print("Evidence verified: 7 references, 4 query records, 4 held-out outcomes.")
    print("Q2: para-F predicted active; held-out outcome 66 nM, active.")
    print(f"Prediction record verified: {prediction_path}")
    print("Comparator checks: R2 para-Cl 180 nM active; R5 para-OH >3000 nM inactive.")
    print("Q4 comparator verified: R4 para-Cl + meta-Me, 156 nM active.")
    print("Focus crops verified: RDKit target atoms and Cl/OH/F glyph margins are within view.")


if __name__ == "__main__":
    main()
