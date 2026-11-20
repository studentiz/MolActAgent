"""The SMILES classifier keeps decisions and response replay auditable."""

import json

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from molactagent._molecular_recording import ReplayClient, ReplayError
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
        return json.dumps({"choice": choice, "confidence": 0.75})


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
