"""SMILES decisions and replay must retain the evidence the model used."""

import json

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from molactagent._molecular_evidence import ReferenceJudge
from molactagent._molecular_recording import ReplayClient, ReplayError
from molactagent._molecular_report import _matched_ends, render_molecular_report
from molactagent._molecules import make_record, specified_stereoisomers
from molactagent.molecular import MolecularClassifier


@pytest.fixture
def molecules():
    reference = pd.DataFrame({
        "id": ["exact", "active-ref", "inactive-ref"],
        "smiles": ["Oc1ccccc1", "Cc1ccc(O)cc1", "Cc1ccccc1"],
        "label": ["active", "active", "inactive"],
    })
    return reference, pd.DataFrame({"id": ["query"], "smiles": ["c1ccc(O)cc1"]})


def evidence_reply(label, *, confidence=0.75, references=(1, 2), rationale="Only one analog shares the linkage."):
    return {"choice": label, "confidence": confidence, "reference_ids": list(references),
            "rationale": rationale, "counter_evidence": "The remaining change is untested.",
            "counter_reference_ids": []}


class Replies:
    model_name = "test-model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.last_call = None
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        response, truncated = next(self.responses)
        self.last_call = {"truncated": truncated, "response_model": self.model_name}
        return json.dumps(response)


def report_for(result):
    return render_molecular_report(
        [result], endpoint="Example assay",
        run={"mode": "replay", "model": "test-model", "temperature": 0.0, "max_tokens": 16384},
    )


def test_molecular_votes_can_disagree_with_nearest_reference(molecules):
    reference, query = molecules
    classifier = MolecularClassifier(Replies([(evidence_reply("inactive"), False)]), repeats=1)
    result = classifier.fit(reference, id_column="id").predict(query, id_column="id")[0]
    assert [item["reference"] for item in result.references] == ["active-ref", "inactive-ref"]
    assert result.references[0]["similarity"] > result.references[1]["similarity"]
    assert result.label == "inactive"


def test_written_references_use_lab_ids_from_the_comparison(molecules):
    reference, query = molecules
    rationale = "Compare refs 1 and 2. Ref 0 is absent. The cutoff is 1000 nM."
    client = Replies([(evidence_reply("active", rationale=rationale), False)])
    result, = MolecularClassifier(client, repeats=1).fit(reference, id_column="id").predict(query)
    html = report_for(result)
    assert ('Compare <a href="#compound-1-ref-1">active-ref</a> and '
            '<a href="#compound-1-ref-2">inactive-ref</a>. Ref 0 is absent. The cutoff is 1000 nM.') in html
    assert 'href="#compound-1-ref-0"' not in html


def test_truncated_response_retries_and_replays(molecules, tmp_path):
    reference, query = molecules
    classifier = MolecularClassifier(
        Replies([(evidence_reply("inactive"), True), (evidence_reply("active"), False)]), repeats=1,
    ).fit(reference)
    result = classifier.predict(query, record=True)[0]
    assert result.label == "active"
    assert [call["status"] for call in result.calls] == ["truncated", "accepted"]

    path = tmp_path / "recording.json"
    path.write_text(json.dumps(classifier.recording), encoding="utf-8")
    replay = ReplayClient(path)
    offline = MolecularClassifier(replay, **replay.settings).fit(reference)
    assert offline.predict(query) == [result]
    assert offline.predict(query) == [result]
    with pytest.raises(ReplayError, match="setting differs for 'endpoint'"):
        MolecularClassifier(replay, repeats=1, endpoint="Different endpoint").fit(reference).predict(query)


@pytest.mark.parametrize("invalid", [
    {"choice": "active"},
    evidence_reply("active", confidence="NaN"),
    evidence_reply("active", confidence=float("inf")),
    evidence_reply("active", references=(0,)),  # This row matches the query and is excluded.
    evidence_reply("active", references=(True,)),
])
def test_incomplete_or_uncited_responses_do_not_vote(molecules, invalid):
    reference, query = molecules
    classifier = MolecularClassifier(Replies([(invalid, False)] * 2), repeats=1).fit(reference)
    result = classifier.predict(query)[0]
    assert result.label == "Inconclusive"
    assert [call["status"] for call in result.calls] == ["invalid_response"] * 2
    assert not result.judgments[0]["parsed"]


def test_model_uncertainty_is_retained_and_single_vote_lacks_coverage(molecules):
    reference, query = molecules
    client = Replies([(evidence_reply("active"), False), (evidence_reply("Uncertain"), False)] +
                     [({"choice": "active"}, False)] * 6)
    result = MolecularClassifier(client, workers=1).fit(reference).predict(query)[0]
    assert len(client.requests) == 8
    assert result.label == "Inconclusive"
    assert result.decision["counts"] == {"active": 1, "inactive": 0, "uncertain": 1, "failed": 3}
    assert result.decision["required"] == 3
    assert result.judgments[1]["label"] == "Uncertain" and result.judgments[1]["parsed"]
    assert "Why the model abstained" in report_for(result)


@pytest.mark.parametrize(
    ("confidence", "label", "description"),
    [(0.8, "active", "Their total model scores resolved the tie"),
     (0.75, "Inconclusive", "tied on both vote count and total model score")],
)
def test_report_shows_both_interpretations_when_votes_disagree(molecules, confidence, label, description):
    reference, query = molecules
    client = Replies([
        (evidence_reply("active", confidence=confidence, rationale="Supports <active>."), False),
        (evidence_reply("inactive", rationale="Supports <inactive>."), False),
    ])
    result = MolecularClassifier(client, repeats=2, workers=1).fit(reference).predict(query)[0]
    assert result.label == label
    reading, _ = report_for(result).split('<div class="records">', 1)
    assert description in reading
    assert "Supports &lt;active&gt;." in reading and "Supports &lt;inactive&gt;." in reading


@pytest.fixture
def halogen_references():
    return pd.DataFrame({
        "id": ["panel-active", "other-active", "halogenated-inactive", "other-inactive"],
        "smiles": ["Oc1ccc(Cl)cc1", "Oc1cccc(Cl)c1", "Clc1ccccc1", "CCO"],
        "label": ["active", "active", "inactive", "inactive"],
    })


def test_report_links_to_a_cited_reference_outside_the_ranked_panel(halogen_references):
    query = pd.DataFrame({"smiles": ["Oc1ccccc1"]})
    client = Replies([(evidence_reply("active", references=(1,),
                                     rationale="Compare ref 1 to the query."), False)])
    result = MolecularClassifier(client, repeats=1, neighbors=1).fit(
        halogen_references, id_column="id",
    ).predict(query)[0]
    assert "other-active" not in [row["reference"] for row in result.references]
    assert "other-active" in [row["reference"] for row in result.evidence_references]
    report = report_for(result)
    assert '<a href="#compound-1-ref-1">other-active</a>' in report
    assert '<tr id="compound-1-ref-1">' in report


def test_focused_report_names_the_comparison_that_supplied_the_call(halogen_references):
    query = pd.DataFrame({"smiles": ["Oc1ccccc1"]})
    client = Replies([
        (evidence_reply("active", references=(0, 2)), False),
        (evidence_reply("inactive", references=(1, 2),
                        rationale="This judgment considers the chlorine substitution."), False),
    ])
    result = MolecularClassifier(client, profile="focused", repeats=1, neighbors=1).fit(
        halogen_references, id_column="id",
    ).predict(query)[0]
    assert result.selected_view == "halogen_deletion" and result.label == "inactive"
    assert [row["reference"] for row in result.judgments[1]["references"]] == [
        "panel-active", "other-active", "halogenated-inactive",
    ]
    report = report_for(result)
    assert "The halogen comparison supplied the final call." in report
    assert "This judgment considers the chlorine substitution." in report
    assert "Another judgment favored active" in report


@pytest.mark.parametrize(("smiles", "opposite", "expected"), [
    ("CCO", "[13CH3]CO", False),
    ("C[C@H](O)CC", "CC(O)CC", False),
    ("C[C@H](O)CC", "C[C@@H](O)CC", True),
])
def test_stereo_comparisons_require_two_fully_specified_stereoisomers(smiles, opposite, expected):
    active = make_record(smiles, 0, "A", "active")
    inactive = make_record(opposite, 1, "I", "inactive")
    assert specified_stereoisomers(active, inactive) is expected
    pairs = ReferenceJudge._stereochemical_matched_pairs(active, [active, inactive], [0, 1])
    assert ("TEST_CONSTITUTION_MATCH" in pairs) is expected


def test_rejects_conflicting_canonical_labels_and_checks_only_declared_cutoffs(molecules):
    reference, _ = molecules
    conflicting = pd.concat([reference, pd.DataFrame({
        "id": ["same-molecule"], "smiles": ["c1ccc(O)cc1"], "label": ["inactive"],
    })])
    with pytest.raises(ValueError, match="exact.*same-molecule.*conflicting labels"):
        MolecularClassifier(None).fit(conflicting, id_column="id")
    reference["potency"] = [5.0, 7.1, 4.1]
    MolecularClassifier(None, pchembl_threshold=6.5).fit(reference, pchembl_column="potency")
    with pytest.raises(ValueError, match="exact.*conflicts"):
        MolecularClassifier(None, pchembl_threshold=6.5, labels_from_pchembl=True).fit(
            reference, id_column="id", pchembl_column="potency",
        )


def test_bond_marks_require_a_unique_site_in_the_same_structure():
    match = _matched_ends("Fc1ccccc1O", "Oc1ccccc1Cl")
    assert match and [site[2] for site in match] == ["C–F", "C–Cl"]
    assert _matched_ends("Fc1ccccc1O", "Clc1ccc(O)cc1") is None  # Position also changed.
    assert _matched_ends("Fc1ccccc1F", "Fc1ccccc1Cl") is None  # Ambiguous query end.
    assert _matched_ends("C[C@H](O)CCl", "C[C@@H](O)CF") is None


def test_repeated_explanations_and_reference_measurements(molecules):
    reference, query = molecules
    reply = evidence_reply("active", references=(1, 2))
    result = MolecularClassifier(Replies([(reply, False)] * 3), repeats=3, workers=1).fit(
        reference, id_column="id",
    ).predict(query)[0]
    html = render_molecular_report(
        [result], endpoint="Example assay",
        run={"mode": "live", "model": "fixture", "temperature": 0, "max_tokens": 16384},
        reference_notes={1: {"measurement": "<5 nM", "source_url": "https://example.org/A"},
                         2: {"measurement": ">50 nM", "source_url": "javascript:alert(1)"}},
    )
    assert html.count('<section class="argument">') == 1
    assert "Identical evidence in 3 judgments" in html
    assert "&lt;5 nM" in html and 'href="https://example.org/A"' in html
    assert "javascript:" not in html and "Individual LLM judgments (3)" in html


def test_repeated_measurements_use_one_panel_slot_and_keep_sources(molecules):
    ref, query = molecules
    ref = pd.concat([ref, ref.iloc[[1]].assign(id="repeat", measurement="12 nM")], ignore_index=True)
    model = Replies([(evidence_reply("active"), False)])
    result = MolecularClassifier(model, repeats=1).fit(ref, id_column="id").predict(query)[0]
    assert len(result.references) == 2
    replicates = result.references[0]["measurements"]
    assert [r["reference"] for r in replicates] == ["active-ref", "repeat"]
    assert replicates[1]["measurement"] == "12 nM"
    assert "2 measurements of this structure" in report_for(result)
    assert "The active and inactive panels share a chemical series." not in str(model.requests)


def test_mixed_transport_failure_is_reported_as_a_run_status(molecules):
    from molactagent._molecular_recording import ModelRequestError

    ref, query = molecules

    class Timeouts(Replies):
        def chat(self, **kwargs):
            if ":r0:" in kwargs["tag"]:
                return super().chat(**kwargs)
            raise ModelRequestError("APITimeoutError")

    model = Timeouts([(evidence_reply("active"), False)])
    result = MolecularClassifier(model, workers=1).fit(ref).predict(query)[0]
    assert result.decision["counts"] == dict(active=1, inactive=0, uncertain=0, failed=4)
    assert "1 favored active, 0 favored inactive, 0 abstained, 4 timed out" in report_for(result)


def test_measurement_text_is_display_only_and_numeric_potency_is_sent(molecules):
    ref, query = molecules

    def request(text, potency):
        measured = ref.assign(measurement=text, pchembl=[6.0, potency, 4.0])
        model = Replies([(evidence_reply("active"), False)])
        MolecularClassifier(model, repeats=1).fit(measured, pchembl_column="pchembl").predict(query)
        return model.requests[0]["messages"]

    assert request("12 nM", 7.1) == request("archived 12 nM", 7.1)
    assert request("12 nM", 7.1) != request("12 nM", 7.9)
