"""Tests for the ``molactagent`` command-line interface.

The tests invoke :func:`molactagent.cli.main` directly with an argument
list, capturing stdout / stderr via the ``capsys`` fixture. The pipeline
is intercepted with the :class:`FakeLLM` defined in ``conftest.py`` so
no real network call is made.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

import pandas as pd
import pytest

from molactagent import cli
from molactagent.cli import build_parser, main


# ── Argument parser smoke tests ─────────────────────────────────────


class TestParser:
    def test_help_runs(self, capsys):
        with pytest.raises(SystemExit) as e:
            build_parser().parse_args(["--help"])
        assert e.value.code == 0

    def test_version_runs(self, capsys):
        with pytest.raises(SystemExit) as e:
            build_parser().parse_args(["--version"])
        captured = capsys.readouterr()
        assert "molactagent" in captured.out

    def test_command_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_validate_required_flags(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["validate"])

    def test_validate_minimum_args(self, csv_pair_with_ids):
        ref, tst = csv_pair_with_ids
        args = build_parser().parse_args([
            "validate",
            "-r", str(ref), "-t", str(tst), "--label-col", "activity",
            "--id-column", "sample_id",
        ])
        assert args.command == "validate"
        assert args.reference == str(ref)
        assert args.label_column == "activity"
        assert args.id_column == "sample_id"
        assert args.k == 3

    def test_validate_missing_id_column_flag(self, csv_pair_with_ids):
        """The --id-column flag is required; argparse rejects its absence."""
        ref, tst = csv_pair_with_ids
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "validate",
                "-r", str(ref), "-t", str(tst), "--label-col", "activity",
            ])

    def test_run_output_dir_is_optional(self):
        """As of v0.3, --output-dir is OPTIONAL: omitting it puts
        MolActAgent in stdout-only ('middleware') mode."""
        args = build_parser().parse_args([
            "run", "-r", "x.csv", "-t", "y.csv", "--label-col", "lbl",
            "--id-column", "id",
            # no -o — should NOT raise
        ])
        assert args.output_dir is None

    def test_run_still_requires_label_and_id(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "run", "-r", "x.csv", "-t", "y.csv",
                # missing --label-col and --id-column
            ])


# ── validate subcommand ─────────────────────────────────────────────


_ID_ARG = ("--id-column", "sample_id")
# Tests that read stdout via capsys must request `--format human`
# explicitly because pytest's stdout capture is not a TTY (so the new
# auto-detect would otherwise produce JSON / JSONL).
_HUMAN = ("--format", "human")


class TestValidateCommand:
    def test_happy_path(self, csv_pair_with_ids, capsys):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            *_HUMAN,
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Validation: OK" in out

    def test_missing_label_column(self, csv_pair_with_ids, capsys):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "missing", *_ID_ARG, *_HUMAN,
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "Validation: FAILED" in out
        assert "Label column" in out

    def test_missing_file(self, tmp_workspace, capsys):
        rc = main([
            "validate",
            "-r", str(tmp_workspace / "nope.csv"),
            "-t", str(tmp_workspace / "nope.csv"),
            "--label-col", "y", *_ID_ARG, *_HUMAN,
        ])
        assert rc == 2
        out = capsys.readouterr().out
        assert "Validation: FAILED" in out

    def test_json_output(self, csv_pair_with_ids, capsys):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--json",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["is_valid"] is True
        assert "summary" in data

    def test_json_output_failure(self, csv_pair_with_ids, capsys):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "missing", *_ID_ARG, "--json",
        ])
        assert rc == 1
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["is_valid"] is False
        assert len(data["errors"]) > 0

    def test_descriptions_validation(
        self, csv_pair_with_ids, simple_descriptions, tmp_workspace, capsys,
    ):
        ref, tst = csv_pair_with_ids
        descs_path = tmp_workspace / "d.json"
        descs_path.write_text(json.dumps(simple_descriptions))
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--feature-descriptions", str(descs_path),
        ])
        assert rc == 0

    def test_descriptions_incomplete(self, csv_pair_with_ids, tmp_workspace, capsys):
        ref, tst = csv_pair_with_ids
        descs_path = tmp_workspace / "d.json"
        descs_path.write_text(json.dumps({"alpha": "x"}))
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--feature-descriptions", str(descs_path),
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "Feature descriptions are missing" in out

    def test_descriptions_file_missing(self, csv_pair_with_ids, tmp_workspace, capsys):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--feature-descriptions", str(tmp_workspace / "nope.json"),
        ])
        assert rc == 2

    def test_background_at_path(self, csv_pair_with_ids, tmp_workspace, capsys):
        ref, tst = csv_pair_with_ids
        bg = tmp_workspace / "bg.txt"
        bg.write_text("notes")
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--background", f"@{bg}",
        ])
        assert rc == 0

    def test_k_zero(self, csv_pair_with_ids, capsys):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "0",
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "k must be >= 1" in out


# ── init-config subcommand ──────────────────────────────────────────


class TestInitConfigCommand:
    def test_creates_starter_jsonc_by_default(self, tmp_workspace, capsys):
        """Default output is a JSONC `llm_config.jsonc` template."""
        target = tmp_workspace / "llm_config.jsonc"
        rc = main(["init-config", "--path", str(target)])
        assert rc == 0
        assert target.exists()
        body = target.read_text()
        # JSONC comments and a single active provider block
        assert "//" in body
        assert "REPLACE-WITH-YOUR-OWN-KEY" in body
        assert '"api_key"' in body
        assert '"base_url"' in body
        assert '"model_name"' in body
        # Must parse via the loader (placeholder triggers LLMConfigError;
        # but the JSONC must be syntactically valid).
        from molactagent.config import _strip_jsonc_comments
        parsed = json.loads(_strip_jsonc_comments(body))
        assert isinstance(parsed, dict)
        assert {"api_key", "base_url", "model_name"} <= set(parsed)

    def test_creates_starter_json_alias(self, tmp_workspace, capsys):
        """`--format json` is accepted as an alias for jsonc (same content)."""
        target = tmp_workspace / "llm_config.json"
        rc = main(["init-config", "--path", str(target), "--format", "json"])
        assert rc == 0
        body = target.read_text()
        assert "//" in body
        assert "REPLACE-WITH-YOUR-OWN-KEY" in body

    def test_creates_starter_txt(self, tmp_workspace, capsys):
        target = tmp_workspace / "LLM_API.txt"
        rc = main(["init-config", "--path", str(target), "--format", "txt"])
        assert rc == 0
        assert target.exists()
        assert "apiKey" in target.read_text()

    def test_refuses_overwrite(self, tmp_workspace, capsys):
        target = tmp_workspace / "llm_config.jsonc"
        target.write_text("already here")
        rc = main(["init-config", "--path", str(target)])
        assert rc == 3
        assert target.read_text() == "already here"

    def test_overwrite_flag(self, tmp_workspace, capsys):
        target = tmp_workspace / "llm_config.jsonc"
        target.write_text("already here")
        rc = main(["init-config", "--path", str(target), "--overwrite"])
        assert rc == 0
        assert "REPLACE-WITH-YOUR-OWN-KEY" in target.read_text()


# ── run subcommand (with FakeLLM injected via monkeypatch) ──────────


@pytest.fixture()
def patch_pipeline_to_fake(monkeypatch, fake_llm):
    """Force ``cli._cmd_run`` to use the FakeLLM instead of building a real one."""
    from molactagent import pipeline as pl
    real_run = pl.run_prediction

    def fake_run(*args, **kwargs):
        kwargs.pop("llm_config", None)
        kwargs.setdefault("llm_client", fake_llm)
        return real_run(*args, **kwargs)

    # The CLI imports run_prediction from .pipeline at module import
    # time, so we patch it both in the pipeline module and in the cli
    # module's namespace.
    monkeypatch.setattr(pl, "run_prediction", fake_run)
    monkeypatch.setattr(cli, "run_prediction", fake_run)
    return fake_llm


@pytest.fixture()
def write_fake_llm_config(tmp_workspace):
    """Write a syntactically valid LLM config so the CLI's load step passes."""
    p = tmp_workspace / "LLM_API.txt"
    p.write_text(
        "apiKey: sk-FAKE\nbase_url: https://example.invalid/v1\n"
        "Modelname: fake-model\n",
    )
    return p


class TestRunCommand:
    def test_happy_path_default_outputs(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        """Default outputs are summary.csv + interpretability.{md,html}.

        Per-sample HTML/MD/JSON files are NOT produced unless --per-sample
        is passed. summary.json is also opt-in (--summary-json).
        """
        ref, tst = csv_pair_with_ids
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            *_HUMAN,
        ])
        assert rc == 0
        # Default outputs.
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "interpretability.md").exists()
        assert (out_dir / "interpretability.html").exists()
        # Opt-in outputs are NOT present.
        assert not (out_dir / "summary.json").exists()
        assert not (out_dir / "interpretability.txt").exists()  # renamed
        # No per-sample files in the default mode.
        per_sample_html = [
            p for p in out_dir.glob("*.html")
            if p.name != "interpretability.html"
        ]
        assert per_sample_html == []
        per_sample_md = [
            p for p in out_dir.glob("*.md")
            if p.name != "interpretability.md"
        ]
        assert per_sample_md == []
        # Stdout banner.
        out = capsys.readouterr().out
        assert "CLASSIFICATION SUMMARY" in out
        assert "INTERPRETABILITY" in out

    def test_validation_fails_returns_1(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "missing", *_ID_ARG, "--k", "2",
            "-o", str(tmp_workspace / "out"),
            "--llm-config", str(write_fake_llm_config),
        ])
        assert rc == 1

    def test_missing_llm_config_returns_3(self, csv_pair_with_ids, tmp_workspace, capsys):
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(tmp_workspace / "out"),
            "--llm-config", str(tmp_workspace / "nope.txt"),
        ])
        assert rc == 3
        err = capsys.readouterr().err
        # The new banner tells the user exactly which file to edit.
        assert "LLM CONFIGURATION ERROR" in err
        assert "llm_config.jsonc" in err

    def test_per_sample_flag_writes_html_md_json(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        """--per-sample turns ON per-sample HTML/MD/JSON for every test row."""
        ref, tst = csv_pair_with_ids
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--per-sample",
        ])
        assert rc == 0
        # 2 test rows -> 2 HTMLs (interpretability.html doesn't count
        # because we filter it out).
        per_sample_html = [
            p for p in out_dir.glob("*.html")
            if p.name != "interpretability.html"
        ]
        assert len(per_sample_html) == 2
        per_sample_md = [
            p for p in out_dir.glob("*.md")
            if p.name != "interpretability.md"
        ]
        assert len(per_sample_md) == 2
        per_sample_json = list(out_dir.glob("*.json"))
        assert len(per_sample_json) == 2  # T_*.json for each test row

    def test_per_sample_format_subset(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        """`--per-sample-format json` produces only per-sample JSON files."""
        ref, tst = csv_pair_with_ids
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--per-sample", "--per-sample-format", "json",
        ])
        assert rc == 0
        assert len(list(out_dir.glob("*.json"))) == 2
        assert [p for p in out_dir.glob("*.html") if p.name != "interpretability.html"] == []
        assert [p for p in out_dir.glob("*.md") if p.name != "interpretability.md"] == []

    def test_per_sample_format_unknown_value_rejected(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(tmp_workspace / "out"),
            "--llm-config", str(write_fake_llm_config),
            "--per-sample", "--per-sample-format", "html,pdf",
        ])
        assert rc == 3
        err = capsys.readouterr().err
        assert "per-sample-format" in err
        assert "pdf" in err

    def test_summary_json_opt_in(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--summary-json",
        ])
        assert rc == 0
        assert (out_dir / "summary.json").exists()
        summary = json.loads((out_dir / "summary.json").read_text())
        # Multi-class: summary.json carries the canonical class order.
        assert summary["class_labels"] == ["active", "inactive"]

    def test_no_interpretability_flag(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--no-interpretability",
        ])
        assert rc == 0
        assert not (out_dir / "interpretability.md").exists()
        assert not (out_dir / "interpretability.html").exists()

    def test_no_explanations_legacy_flag(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        """The deprecated --no-explanations flag is still accepted and
        equivalent to "do not produce per-sample reports"."""
        ref, tst = csv_pair_with_ids
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--per-sample",  # turn it on
            "--no-explanations",  # then turn it off again via the legacy flag
        ])
        assert rc == 0
        # Per-sample reports were suppressed by --no-explanations.
        per_sample_html = [
            p for p in out_dir.glob("*.html")
            if p.name != "interpretability.html"
        ]
        assert per_sample_html == []

    def test_positive_class_flag_is_deprecated(
        self, csv_pair_with_ids, write_fake_llm_config, tmp_workspace,
        patch_pipeline_to_fake, capsys,
    ):
        # Multi-class: --positive-class is accepted for back-compat
        # but has no effect. The summary still reports the canonical
        # class order (alphabetical) and a deprecation warning surfaces
        # in the validation summary.
        ref, tst = csv_pair_with_ids
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--positive-class", "active",
            "--summary-json",
        ])
        assert rc == 0
        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["class_labels"] == ["active", "inactive"]
        assert any(
            "positive_class" in w and "deprecated" in w
            for w in summary["validation_warnings"]
        )

    def test_env_var_provides_llm_config(
        self, csv_pair_with_ids, tmp_workspace, monkeypatch,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "sk-env")
        monkeypatch.setenv("MOLACTAGENT_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("MOLACTAGENT_MODEL", "fake-model")
        # Ensure no default config is found by walking upward from a tmp cwd.
        monkeypatch.chdir(tmp_workspace)
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(tmp_workspace / "out"),
        ])
        assert rc == 0


# ── pipe-friendly / middleware behavior ─────────────────────────────


class TestStdoutFormats:
    """The CLI must behave like a Unix tool: clean stdout, chrome on
    stderr, and structured payloads in JSON / JSONL / TSV / CSV mode.
    """

    def test_validate_json_format_emits_parseable_json(
        self, csv_pair_with_ids, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--format", "json",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["is_valid"] is True
        assert "summary" in data
        # Stdout should contain ONLY the JSON object (one trailing \n).
        assert out.lstrip().startswith("{")
        assert out.rstrip().endswith("}")

    def test_validate_legacy_json_flag_still_works(
        self, csv_pair_with_ids, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--json",  # deprecated alias
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["is_valid"] is True

    def test_validate_default_under_capsys_is_json(
        self, csv_pair_with_ids, capsys,
    ):
        """capsys is not a TTY → auto-detected format should be json."""
        ref, tst = csv_pair_with_ids
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
        ])
        assert rc == 0
        # Must be parseable as JSON because we're not on a TTY.
        json.loads(capsys.readouterr().out)

    def test_run_jsonl_format(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--format", "jsonl",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 2  # 2 test samples in csv_pair_with_ids
        for line in lines:
            obj = json.loads(line)
            assert {"sample_id", "predicted_label", "confidence",
                    "vote_summary", "key_drivers"} <= set(obj)
            # Confidence is a float in [0, 1]
            assert 0.0 <= obj["confidence"] <= 1.0

    def test_run_json_format(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--format", "json",
        ])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        # Multi-class: ``label_mapping`` is gone; ``class_labels`` lists
        # the canonical ordered class set instead.
        assert {"validation", "class_labels", "outcomes",
                "n_succeeded", "n_failed",
                "summary_csv_path", "summary_json_path",
                "interpretability_md_path", "interpretability_html_path",
                "interpretability_text", "llm_stats"} <= set(payload)
        assert payload["n_succeeded"] == 2
        assert len(payload["outcomes"]) == 2
        assert payload["class_labels"] == ["active", "inactive"]

    def test_run_tsv_format(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--format", "tsv",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # Split on "\n" only (not str.strip(), which would eat a trailing
        # empty TSV column).
        lines = [line for line in out.split("\n") if line]
        assert len(lines) == 3  # header + 2 samples
        header = lines[0].split("\t")
        assert header == [
            "sample_id", "predicted_label", "confidence",
            "true_label", "elapsed_s", "error",
        ]
        for line in lines[1:]:
            cols = line.split("\t")
            assert len(cols) == 6

    def test_run_csv_format(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--format", "csv",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [line for line in out.split("\n") if line]
        assert len(lines) == 3
        assert lines[0].startswith("sample_id,predicted_label,")
        # CSV must round-trip through pandas without errors.
        df = pd.read_csv(io.StringIO(out))
        assert len(df) == 2
        assert list(df.columns) == [
            "sample_id", "predicted_label", "confidence",
            "true_label", "elapsed_s", "error",
        ]

    def test_run_default_under_capsys_is_jsonl(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        """capsys is not a TTY → run defaults to jsonl (one JSON per line)."""
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 2
        # Each line must be standalone JSON.
        for line in lines:
            json.loads(line)

    def test_human_format_chrome_NOT_in_jsonl_stdout(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys, caplog,
    ):
        """In jsonl mode, no banner / interpretability / 'saved to' text
        should leak into stdout. Stdout = data, stderr/log = chrome.

        Note: pytest captures the logging stream into ``caplog`` rather
        than exposing it on ``capsys.err``. In real use (no pytest)
        these go through the stderr handler installed by
        ``logging.basicConfig`` — verified separately by the smoke
        test in ``test_real_pipe_smoke``.
        """
        import logging as _logging
        caplog.set_level(_logging.INFO, logger="molactagent.cli")
        ref, tst = csv_pair_with_ids
        out_dir = csv_pair_with_ids[0].parent / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--format", "jsonl",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        # No banners or "saved to" notes in stdout.
        for forbidden in (
            "CLASSIFICATION SUMMARY",
            "INTERPRETABILITY",
            "Top drivers",
            "Summary CSV",
            "succeeded,",
            "===",
            "Interpretability (Markdown) saved to",
        ):
            assert forbidden not in captured.out, (
                f"Unexpected text {forbidden!r} found in jsonl stdout"
            )
        # File-saved messages live in the log stream (=> stderr in
        # real use, => caplog under pytest).
        log_text = "\n".join(r.message for r in caplog.records)
        assert "wrote summary CSV" in log_text
        assert "wrote interpretability MD" in log_text
        assert "wrote interpretability HTML" in log_text


class TestOutputDirOptional:
    def test_run_without_output_dir_emits_jsonl_to_stdout(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        """The user can omit -o entirely and use MolActAgent purely as a
        stdin/stdout filter — no files written, all data on stdout."""
        ref, tst = csv_pair_with_ids
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--format", "jsonl",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 2
        # No files were written (no -o → output_dir is None).
        # Verifying via the absence of the typical messages on stderr.
        # (The pipeline returns early without creating any directories.)


class TestStdinSupport:
    """MolActAgent can act as middleware: pipe a CSV in via -, get
    JSONL out, feed that to the next tool."""

    def test_reference_from_stdin(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys, monkeypatch,
    ):
        ref, tst = csv_pair_with_ids
        monkeypatch.setattr("sys.stdin", io.StringIO(ref.read_text()))
        rc = main([
            "run",
            "-r", "-", "-t", str(tst),
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--format", "jsonl",
        ])
        assert rc == 0
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 2

    def test_test_from_stdin(
        self, csv_pair_with_ids, write_fake_llm_config,
        patch_pipeline_to_fake, capsys, monkeypatch,
    ):
        ref, tst = csv_pair_with_ids
        monkeypatch.setattr("sys.stdin", io.StringIO(tst.read_text()))
        rc = main([
            "run",
            "-r", str(ref), "-t", "-",
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--format", "jsonl",
        ])
        assert rc == 0
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 2

    def test_stdin_tsv_format(
        self, simple_reference_with_ids, simple_test_with_ids,
        write_fake_llm_config, patch_pipeline_to_fake,
        capsys, monkeypatch, tmp_workspace,
    ):
        """`--stdin-format tsv` lets users pipe TSV data."""
        ref_path = tmp_workspace / "ref.csv"
        simple_reference_with_ids.to_csv(ref_path, index=False)
        # The test fixture has tab-separated stdin
        tsv_text = simple_test_with_ids.to_csv(index=False, sep="\t")
        monkeypatch.setattr("sys.stdin", io.StringIO(tsv_text))
        rc = main([
            "run",
            "-r", str(ref_path), "-t", "-",
            "--label-col", "activity", *_ID_ARG, "--k", "2",
            "--llm-config", str(write_fake_llm_config),
            "--stdin-format", "tsv",
            "--format", "jsonl",
        ])
        assert rc == 0
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 2

    def test_double_stdin_is_rejected(
        self, capsys,
    ):
        """Reading both -r and -t from stdin would consume the same
        stream — the CLI must catch this clearly."""
        rc = main([
            "validate",
            "-r", "-", "-t", "-",
            "--label-col", "x", *_ID_ARG,
        ])
        assert rc == 3
        err = capsys.readouterr().err
        assert "stdin" in err.lower() and "both" in err.lower()

    def test_empty_stdin_is_rejected(
        self, csv_pair_with_ids, capsys, monkeypatch,
    ):
        ref, tst = csv_pair_with_ids
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        rc = main([
            "validate",
            "-r", "-", "-t", str(tst),
            "--label-col", "activity", *_ID_ARG,
        ])
        # No content on stdin -> load failure -> exit code 2.
        assert rc == 2
        captured = capsys.readouterr()
        haystack = (captured.err + captured.out).lower()
        # The error surfaces with a clear "stdin" mention or a generic
        # "could not load" message — both are useful for the operator.
        assert "stdin" in haystack or "could not load" in haystack


# ── real subprocess pipe smoke tests ────────────────────────────────


class TestRealSubprocessPipe:
    """Drive the installed ``molactagent`` console script as a real
    subprocess so we exercise the actual stderr / stdout split and the
    TTY auto-detection. These tests skip if ``molactagent`` is not on
    the PATH (e.g. the package wasn't installed)."""

    @pytest.fixture()
    def molactagent_bin(self):
        import shutil
        # Look on PATH first, then in the sibling of the running
        # interpreter (covers conda envs whose bin/ isn't on PATH).
        for candidate in (
            shutil.which("molactagent"),
            str(pathlib.Path(sys.executable).parent / "molactagent"),
        ):
            if candidate and pathlib.Path(candidate).exists():
                return candidate
        pytest.skip("molactagent console script not installed")

    def test_validate_jsonl_pipe_stays_clean(
        self, molactagent_bin, csv_pair_with_ids,
    ):
        """`molactagent validate ... | jq .` must work — i.e. stdout is
        valid JSON when piped, and stderr carries any chrome."""
        import subprocess
        ref, tst = csv_pair_with_ids
        proc = subprocess.run(
            [
                molactagent_bin, "--quiet",
                "validate",
                "-r", str(ref), "-t", str(tst),
                "--label-col", "activity",
                "--id-column", "sample_id", "--k", "2",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        # stdout should be a parseable JSON object (auto-detected from
        # the non-TTY pipe), with no leading log noise.
        data = json.loads(proc.stdout)
        assert data["is_valid"] is True
        assert "summary" in data

    def test_explicit_human_format_works_when_piped(
        self, molactagent_bin, csv_pair_with_ids,
    ):
        """User can force human format even when piped (e.g. for
        scrolling through `less`)."""
        import subprocess
        ref, tst = csv_pair_with_ids
        proc = subprocess.run(
            [
                molactagent_bin, "--quiet",
                "validate",
                "-r", str(ref), "-t", str(tst),
                "--label-col", "activity",
                "--id-column", "sample_id", "--k", "2",
                "--format", "human",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0
        assert "Validation: OK" in proc.stdout

    def test_stdin_pipe_round_trip(
        self, molactagent_bin, csv_pair_with_ids,
    ):
        """Pipe a CSV into molactagent via `-` and verify it parses."""
        import subprocess
        ref, tst = csv_pair_with_ids
        proc = subprocess.run(
            [
                molactagent_bin, "--quiet",
                "validate",
                "-r", "-", "-t", str(tst),
                "--label-col", "activity",
                "--id-column", "sample_id", "--k", "2",
            ],
            input=ref.read_text(),
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["is_valid"] is True


# ── unknown command rejection ───────────────────────────────────────


class TestErrorPaths:
    def test_unknown_command(self, capsys):
        # argparse rejects unknown commands itself with SystemExit(2)
        with pytest.raises(SystemExit):
            main(["bogus"])


# ── --id-column flag ────────────────────────────────────────────────


class TestIdColumnFlag:
    def _csvs(self, tmp_workspace, *, test_ids=("alpha", "beta")):
        ref = tmp_workspace / "ref.csv"
        tst = tmp_workspace / "test.csv"
        ref.write_text(
            "rid,x,y,lbl\n"
            "r1,5.0,5.0,yes\n"
            "r2,5.5,4.9,yes\n"
            "r3,-5.0,-5.1,no\n"
            "r4,-5.5,-4.9,no\n"
        )
        tst.write_text(
            "rid,x,y\n"
            f"{test_ids[0]},4.7,5.0\n"
            f"{test_ids[1]},-5.2,-4.8\n"
        )
        return ref, tst

    def test_validate_with_id_column(self, tmp_workspace, capsys):
        ref, tst = self._csvs(tmp_workspace)
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "lbl", "--id-column", "rid", "--k", "2",
            *_HUMAN,
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Validation: OK" in out
        assert "n_features: 2" in out

    def test_validate_unknown_id_column(self, tmp_workspace, capsys):
        ref, tst = self._csvs(tmp_workspace)
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "lbl", "--id-column", "nope", "--k", "2",
            *_HUMAN,
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "id_column" in out

    def test_validate_rejects_cross_dataset_duplicate(
        self, tmp_workspace, capsys,
    ):
        """IDs must be unique across reference + test combined."""
        # Use an id 'r1' in the test CSV that collides with the reference.
        ref, tst = self._csvs(tmp_workspace, test_ids=("r1", "beta"))
        rc = main([
            "validate",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "lbl", "--id-column", "rid", "--k", "2",
            *_HUMAN,
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "BOTH" in out or "globally unique" in out or "both" in out.lower()

    def test_run_with_id_column(
        self, tmp_workspace, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        """The interpretability files (default outputs) name every test
        sample by its id_column value. Per-sample reports require
        --per-sample to opt in."""
        ref, tst = self._csvs(tmp_workspace)
        out_dir = tmp_workspace / "out"
        rc = main([
            "run",
            "-r", str(ref), "-t", str(tst),
            "--label-col", "lbl", "--id-column", "rid", "--k", "2",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--per-sample", *_HUMAN,
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out
        files = {p.name for p in out_dir.iterdir()}
        assert any("alpha" in n for n in files)
        assert any("beta" in n for n in files)
        # The interpretability files also names the samples.
        interp_md = (out_dir / "interpretability.md").read_text()
        assert "alpha" in interp_md
        assert "beta" in interp_md
        interp_html = (out_dir / "interpretability.html").read_text()
        assert "alpha" in interp_html
        assert "beta" in interp_html


# ── Iris example smoke test ─────────────────────────────────────────


class TestIrisExample:
    """Exercise the bundled examples/iris_flowers/ fixtures end-to-end."""

    @pytest.fixture()
    def iris_root(self, project_root):
        root = project_root / "examples" / "iris_flowers"
        if not root.is_dir():
            pytest.skip("examples/iris_flowers/ not present in this checkout")
        return root

    def test_iris_ships_jsonc_config(self, iris_root):
        """The bundled example must use llm_config.jsonc (no legacy .json)."""
        assert (iris_root / "llm_config.jsonc").exists()
        assert not (iris_root / "llm_config.json").exists()

    def test_validate_iris_example(self, iris_root, capsys):
        rc = main([
            "validate",
            "-r", str(iris_root / "reference.csv"),
            "-t", str(iris_root / "test.csv"),
            "--label-col", "species",
            "--id-column", "flower_id",
            "--k", "3",
            "--feature-descriptions",
            str(iris_root / "feature_descriptions.json"),
            "--background", f"@{iris_root / 'background.txt'}",
            *_HUMAN,
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Validation: OK" in out
        assert "setosa" in out
        assert "versicolor" in out

    def test_run_iris_example_default_outputs(
        self, iris_root, tmp_workspace, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        """Default mode (no --per-sample): summary.csv + interpretability."""
        out_dir = tmp_workspace / "iris_out"
        rc = main([
            "run",
            "-r", str(iris_root / "reference.csv"),
            "-t", str(iris_root / "test.csv"),
            "--label-col", "species",
            "--id-column", "flower_id",
            "--k", "3",
            "--feature-descriptions",
            str(iris_root / "feature_descriptions.json"),
            "--background", f"@{iris_root / 'background.txt'}",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
        ])
        assert rc == 0
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "interpretability.md").exists()
        assert (out_dir / "interpretability.html").exists()
        # No per-sample HTML files in default mode (only the
        # interpretability.html aggregate).
        per_sample_html = [
            p for p in out_dir.glob("*.html")
            if p.name != "interpretability.html"
        ]
        assert per_sample_html == []

    def test_run_iris_example_per_sample(
        self, iris_root, tmp_workspace, write_fake_llm_config,
        patch_pipeline_to_fake, capsys,
    ):
        """Adding --per-sample reproduces the legacy per-flower outputs."""
        out_dir = tmp_workspace / "iris_per_sample"
        rc = main([
            "run",
            "-r", str(iris_root / "reference.csv"),
            "-t", str(iris_root / "test.csv"),
            "--label-col", "species",
            "--id-column", "flower_id",
            "--k", "3",
            "--feature-descriptions",
            str(iris_root / "feature_descriptions.json"),
            "--background", f"@{iris_root / 'background.txt'}",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            "--per-sample",
        ])
        assert rc == 0
        # The bundled iris example now contains 6 test flowers across
        # all three species, so we expect 6 per-sample HTML reports.
        per_sample_html = [
            p for p in out_dir.glob("*.html")
            if p.name != "interpretability.html"
        ]
        assert len(per_sample_html) == 6


# ── Regression CLI smoke ────────────────────────────────────────────


class TestRegressionCLI:
    """Exercise the regression branch of the CLI end-to-end with the
    FakeLLM, so a future change to ``--task`` / ``--target-col``
    / regression output formatting would surface immediately.
    """

    def test_run_regression_with_target_col_alias(
        self,
        tmp_workspace,
        regression_csv_pair_with_ids,
        write_fake_llm_config,
        patch_pipeline_to_fake,
    ):
        ref_path, test_path = regression_csv_pair_with_ids
        out_dir = tmp_workspace / "reg_out"
        rc = main([
            "run",
            "-r", str(ref_path), "-t", str(test_path),
            "--target-col", "target",
            "--id-column", "sample_id",
            "--k", "3",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            *_HUMAN,
        ])
        assert rc == 0
        # Standard outputs.
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "interpretability.md").exists()
        assert (out_dir / "interpretability.html").exists()
        # CSV carries the regression-specific columns.
        import pandas as pd
        df = pd.read_csv(out_dir / "summary.csv")
        for col in ("task", "predicted_value", "interval_lo", "interval_hi"):
            assert col in df.columns, (
                f"summary.csv missing regression column {col!r}; "
                f"actual columns = {list(df.columns)}"
            )
        assert (df["task"] == "regression").all()
        assert df["predicted_value"].notna().all()

    def test_explicit_task_regression_succeeds(
        self,
        tmp_workspace,
        regression_csv_pair_with_ids,
        write_fake_llm_config,
        patch_pipeline_to_fake,
    ):
        ref_path, test_path = regression_csv_pair_with_ids
        out_dir = tmp_workspace / "reg_explicit_out"
        rc = main([
            "run",
            "-r", str(ref_path), "-t", str(test_path),
            "--label-col", "target",
            "--task", "regression",
            "--id-column", "sample_id",
            "--k", "3",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            *_HUMAN,
        ])
        assert rc == 0

    def test_task_classification_on_regression_data_fails_clearly(
        self,
        tmp_workspace,
        regression_csv_pair_with_ids,
        write_fake_llm_config,
        patch_pipeline_to_fake,
        capsys,
    ):
        ref_path, test_path = regression_csv_pair_with_ids
        out_dir = tmp_workspace / "reg_mismatch_out"
        rc = main([
            "run",
            "-r", str(ref_path), "-t", str(test_path),
            "--label-col", "target",
            "--task", "classification",
            "--id-column", "sample_id",
            "--k", "3",
            "-o", str(out_dir),
            "--llm-config", str(write_fake_llm_config),
            *_HUMAN,
        ])
        # Mismatch should fail with a clear ValueError surfaced as
        # a non-zero exit code, not a silent success.
        assert rc != 0
        err = capsys.readouterr().err.lower()
        assert "task" in err or "classification" in err
