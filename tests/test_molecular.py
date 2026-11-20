"""The SMILES classifier keeps decisions and response replay auditable."""

import json

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from molactagent._molecular_recording import ReplayClient, ReplayError
from molactagent._molecular_report import render_molecular_report
from molactagent.molecular import MolecularClassifier


@pytest.fixture
def molecules():
    reference = pd.DataFrame(
        {
            "id": ["exact", "active-ref", "inactive-ref"],
            "smiles": ["Oc1ccccc1", "Cc1ccc(O)cc1", "Cc1ccccc1"],
            "label": ["active", "active", "inactive"],
        }
    )
    return reference, pd.DataFrame({"id": ["query"], "smiles": ["c1ccc(O)cc1"]})


class Replies:
    model_name = "test-model"

    def __init__(self, choices):
        self.choices = iter(choices)
        self.last_call = None

    def chat(self, **kwargs):
        choice, truncated = next(self.choices)
        self.last_call = {"truncated": truncated, "response_model": self.model_name}
        response = {"choice": choice, "confidence": 0.75} if isinstance(choice, str) else choice
        return json.dumps(response)


def test_molecular_votes_can_disagree_with_nearest_reference(molecules):
    reference, query = molecules
    classifier = MolecularClassifier(Replies([("inactive", False)]), repeats=1)
    result = classifier.fit(reference, id_column="id").predict(query, id_column="id")[0]
    assert [item["reference"] for item in result.references] == ["active-ref", "inactive-ref"]
    assert result.references[0]["similarity"] > result.references[1]["similarity"]
    assert result.label == "inactive"


def test_truncated_response_retries_and_replays(molecules, tmp_path):
    reference, query = molecules
    classifier = MolecularClassifier(
        Replies([("inactive", True), ("active", False)]), repeats=1,
    ).fit(reference, id_column="id")
    result = classifier.predict(query, id_column="id", record=True)[0]
    assert result.label == "active"
    assert result.calls[0]["truncated"]
    assert len(result.calls) == 2
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(classifier.recording), encoding="utf-8")
    replay = MolecularClassifier(ReplayClient(path), repeats=1).fit(reference, id_column="id")
    assert replay.predict(query, id_column="id") == [result]
    changed = MolecularClassifier(
        ReplayClient(path), repeats=1, endpoint="Different endpoint",
    ).fit(reference, id_column="id")
    with pytest.raises(ReplayError, match="Recorded request differs"):
        changed.predict(query, id_column="id")


def test_unusable_responses_yield_abstention(molecules):
    reference, query = molecules
    classifier = MolecularClassifier(
        Replies([("Uncertain", False), ("Uncertain", False)]), repeats=1,
    ).fit(reference)
    result = classifier.predict(query)[0]
    assert result.label == "Inconclusive"
    assert result.confidence == 0.0
    assert result.judgments[0]["parsed"] is False


@pytest.mark.parametrize(
    ("confidence", "label", "description"),
    [(0.8, "active", "Model scores resolved the tied vote."),
     (0.75, "Inconclusive", "left the decision unresolved.")],
)
def test_report_exposes_both_interpretations_when_votes_disagree(molecules, confidence, label, description):
    reference, query = molecules
    replies = Replies([
        ({"choice": "active", "confidence": confidence, "rationale": "Supports <active>."}, False),
        ({"choice": "inactive", "confidence": 0.75, "rationale": "Supports <inactive>."}, False),
    ])
    result = MolecularClassifier(replies, repeats=2, workers=1).fit(reference).predict(query)[0]
    assert result.label == label
    report = render_molecular_report(
        [result], endpoint="Example assay",
        run={"mode": "replay", "model": "test-model", "temperature": 0.0, "max_tokens": 16384},
    )
    reading, _ = report.split('<div class="records">', 1)
    assert description in reading
    assert "Supports &lt;active&gt;." in reading and "Supports &lt;inactive&gt;." in reading


def test_focused_report_shows_references_from_the_deciding_comparison():
    reference = pd.DataFrame({
        "id": ["panel-active", "focused-active", "halogenated-inactive", "other-inactive"],
        "smiles": ["Oc1ccc(Cl)cc1", "Oc1cccc(Cl)c1", "Clc1ccccc1", "CCO"],
        "label": ["active", "active", "inactive", "inactive"],
    })
    query = pd.DataFrame({"smiles": ["Oc1ccccc1"]})
    replies = Replies([
        ("active", False),
        ({"choice": "inactive", "confidence": 0.75, "decisive_reference": "1",
          "halogen_change_interpretation": "This judgment considers the chlorine substitution."}, False),
    ])
    result = MolecularClassifier(replies, profile="focused", repeats=1, neighbors=1).fit(
        reference, id_column="id",
    ).predict(query)[0]
    assert result.selected_view == "halogen_deletion" and result.label == "inactive"
    assert [row["reference"] for row in result.references] == ["panel-active", "halogenated-inactive"]
    assert [row["reference"] for row in result.judgments[1]["references"]] == [
        "panel-active", "focused-active", "halogenated-inactive",
    ]
    report = render_molecular_report(
        [result], endpoint="Example assay",
        run={"mode": "replay", "model": "test-model", "temperature": 0.0, "max_tokens": 16384},
    )
    reading, _ = report.split('<div class="records">', 1)
    assert "focused-active" in reading and "panel-active" not in reading
    assert "This judgment considers the chlorine substitution." in reading
    assert "A judgment from the reference panel favored active" in reading
