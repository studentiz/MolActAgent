"""Exercise the documented command with file-based inputs and recorded replies."""

import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from molactagent import _molecular_cli
from molactagent.cli import main
from tests.test_molecular import Replies, evidence_reply


@pytest.fixture
def run_files(tmp_path, monkeypatch):
    reference = tmp_path / "references.csv"
    reference.write_text("id,structure,label,potency\n0007,Oc1ccccc1,active,7.1\n7,COc1ccccc1,inactive,5.2\n")
    query = tmp_path / "queries.csv"
    query.write_text("id,structure\n0021,Oc1ccc(F)cc1\nNA,Oc1ccccc1\n")
    model = Replies([(evidence_reply("active", references=(0, 1)), False)])
    monkeypatch.setattr(_molecular_cli, "build_llm_client", lambda config: model)
    inputs = ["molecular", "--reference", str(reference), "--test", str(query)]
    options = ["--id-column", "id", "--smiles-column", "structure", "--pchembl-column", "potency",
               "--pchembl-threshold", "6.5", "--labels-from-pchembl", "--repeats", "1",
               "--series", "Demo", "--endpoint", "Synthetic labels.", "--plain-json"]
    live = ["--model", model.model_name, "--api-key", "fixture", "--base-url", "https://example.invalid/v1"]
    return inputs, options, live, model, query


def test_cli_replays_nondefault_settings_and_preserves_ids(run_files, tmp_path):
    inputs, options, live, model, query = run_files
    output = tmp_path / "live"
    recording = output / "responses.json"
    assert main([*inputs, *options, *live, "--record", str(recording), "-o", str(output)]) == 4
    data = json.loads((output / "predictions.json").read_text())
    assert [row["identifier"] for row in data["predictions"]] == ["0021", "NA"]
    assert [row["label"] for row in data["predictions"]] == ["active", "Inconclusive"]
    assert [row["reference"] for row in data["predictions"][0]["references"]] == ["0007", "7"]
    assert "cutoff · 6.5" in (output / "report.html").read_text()
    assert len(model.requests) == 1
    assert model.requests[0]["max_tokens"] == 16384
    assert "response_format" not in model.requests[0]
    assert "api_key" not in recording.read_text()

    replay = tmp_path / "replay"
    offline = [*inputs, "--replay", str(recording)]
    assert main([*offline, "-o", str(replay)]) == 4
    result = json.loads((replay / "predictions.json").read_text())
    assert data["predictions"] == result["predictions"]
    assert main([*offline, "--neighbors", "3"]) == 2
    query.write_text(query.read_text().replace("0021", "0022"))
    assert main(offline) == 2
    assert len(model.requests) == 1


def test_output_preflight_blocks_expense_and_preserves_existing_file(run_files, tmp_path):
    inputs, options, live, model, _ = run_files
    recording = tmp_path / "recording.json"
    recording.write_text("previous recording")
    output = tmp_path / "new"
    assert main([*inputs, *options, *live, "--record", str(recording), "-o", str(output)]) == 2
    assert recording.read_text() == "previous recording"
    assert not list(output.iterdir())
    assert not model.requests


def test_invalid_input_removes_reserved_outputs_before_any_calls(run_files, tmp_path):
    inputs, options, live, model, query = run_files
    query.write_text("id,structure\n0000,Oc1ccc(F)cc1\nBAD,not-smiles\n")
    output = tmp_path / "new"
    assert main([*inputs, *options, *live, "-o", str(output)]) == 2
    assert not list(output.iterdir())
    assert not model.requests


def test_output_failure_leaves_neither_json_nor_html(run_files, tmp_path, monkeypatch):
    inputs, options, live, model, _ = run_files
    output = tmp_path / "new"
    original_replace = Path.replace

    def fail_second_write(path, target):
        if target.name == "report.html":
            raise OSError("Write failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_write)
    assert main([*inputs, *options, *live, "-o", str(output)]) == 2
    assert not list(output.iterdir())
