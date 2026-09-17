"""End-to-end pipeline tests using the FakeLLM (no network)."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from molactagent import (
    LLMConfig,
    PipelineOutput,
    SampleOutcome,
    run_classification,
)
from molactagent.config import LLMConfigError


class TestPipelineHappyPath:
    def test_simple_run_with_dataframes(
        self, simple_reference, simple_test, fake_llm,
    ):
        out = run_classification(
            reference=simple_reference,
            test=simple_test,
            label_column="activity",
            k=2,
            llm_client=fake_llm,
        )
        assert isinstance(out, PipelineOutput)
        assert out.validation.is_valid
        assert len(out.outcomes) == 2
        assert out.n_failed == 0
        # Both outcomes should produce the user's domain class names
        labels = {o.predicted_label for o in out.outcomes}
        assert labels.issubset({"active", "inactive", "Inconclusive"})

    def test_with_csv_paths_minimal_default(self, csv_pair, fake_llm, tmp_workspace):
        """Default outputs are summary.csv + interpretability.{md,html} only.

        Per-sample HTML/MD/JSON files are NOT produced unless the caller
        opts in with ``per_sample_reports=True`` — that change halves
        the LLM token cost for users who only need the verdicts.
        """
        ref_csv, test_csv = csv_pair
        out_dir = tmp_workspace / "out"
        out = run_classification(
            reference=ref_csv,
            test=test_csv,
            label_column="activity",
            k=2,
            llm_client=fake_llm,
            output_dir=out_dir,
        )
        assert out.validation.is_valid
        assert out_dir.exists()
        # No per-sample artefacts by default.
        for o in out.outcomes:
            assert o.raw_json_path is None
            assert o.explanation_md_path is None
            assert o.explanation_html_path is None
        # Default outputs.
        assert pathlib.Path(out.summary_csv_path).exists()
        assert out.summary_json_path is None  # opt-in
        assert pathlib.Path(out.interpretability_md_path).exists()
        assert pathlib.Path(out.interpretability_html_path).exists()

    def test_per_sample_reports_opt_in(self, csv_pair, fake_llm, tmp_workspace):
        """Setting ``per_sample_reports=True`` produces every per-sample
        artefact (HTML + MD + JSON) and triggers the LLM explanation."""
        ref_csv, test_csv = csv_pair
        out_dir = tmp_workspace / "out"
        out = run_classification(
            reference=ref_csv,
            test=test_csv,
            label_column="activity",
            k=2,
            llm_client=fake_llm,
            output_dir=out_dir,
            per_sample_reports=True,
            write_summary_json=True,
        )
        for o in out.outcomes:
            assert pathlib.Path(o.raw_json_path).exists()
            assert pathlib.Path(o.explanation_md_path).exists()
            assert pathlib.Path(o.explanation_html_path).exists()
        assert pathlib.Path(out.summary_csv_path).exists()
        assert pathlib.Path(out.summary_json_path).exists()
        summary = json.loads(pathlib.Path(out.summary_json_path).read_text())
        # Multi-class: summary.json carries the canonical class order
        # rather than a binary positive/negative mapping.
        assert summary["class_labels"] == ["active", "inactive"]

    def test_per_sample_format_subset(self, csv_pair, fake_llm, tmp_workspace):
        """``per_sample_formats`` lets the caller pick a subset (e.g. JSON
        only) — useful for downstream pipelines that don't need HTML."""
        ref_csv, test_csv = csv_pair
        out_dir = tmp_workspace / "out"
        out = run_classification(
            reference=ref_csv, test=test_csv,
            label_column="activity", k=2,
            llm_client=fake_llm, output_dir=out_dir,
            per_sample_reports=True,
            per_sample_formats=("json",),
        )
        for o in out.outcomes:
            assert pathlib.Path(o.raw_json_path).exists()
            assert o.explanation_md_path is None
            assert o.explanation_html_path is None
            # MD body skipped → no explanation LLM call was made
            assert o.explanation_md == ""

    def test_summary_csv_columns(self, csv_pair, fake_llm, tmp_workspace):
        ref_csv, test_csv = csv_pair
        out_dir = tmp_workspace / "out"
        out = run_classification(
            reference=ref_csv, test=test_csv,
            label_column="activity", k=2,
            llm_client=fake_llm, output_dir=out_dir,
        )
        df = pd.read_csv(out.summary_csv_path)
        for col in (
            "sample_id", "predicted_label", "confidence",
            "true_label", "elapsed_s", "html_report", "error",
        ):
            assert col in df.columns

    def test_no_interpretability_flag(self, csv_pair, fake_llm, tmp_workspace):
        """Users can opt out of interpretability files (no token cost,
        but they may want a quieter output dir)."""
        ref_csv, test_csv = csv_pair
        out_dir = tmp_workspace / "out"
        out = run_classification(
            reference=ref_csv, test=test_csv,
            label_column="activity", k=2,
            llm_client=fake_llm, output_dir=out_dir,
            write_interpretability=False,
        )
        assert out.interpretability_md_path is None
        assert out.interpretability_html_path is None
        assert out.interpretability_text == ""
        assert pathlib.Path(out.summary_csv_path).exists()

    def test_per_sample_format_unknown_value_rejected(
        self, csv_pair, fake_llm, tmp_workspace,
    ):
        ref_csv, test_csv = csv_pair
        with pytest.raises(ValueError, match="per_sample_formats"):
            run_classification(
                reference=ref_csv, test=test_csv,
                label_column="activity", k=2,
                llm_client=fake_llm, output_dir=tmp_workspace / "x",
                per_sample_reports=True,
                per_sample_formats=("html", "pdf"),
            )

    def test_legacy_generate_explanations_false(self, csv_pair, fake_llm, tmp_workspace):
        """The deprecated ``generate_explanations=False`` still suppresses
        per-sample explanations (now equivalent to per_sample_reports=False)."""
        ref_csv, test_csv = csv_pair
        out_dir = tmp_workspace / "out"
        out = run_classification(
            reference=ref_csv, test=test_csv,
            label_column="activity", k=2,
            llm_client=fake_llm, output_dir=out_dir,
            generate_explanations=False,
        )
        for o in out.outcomes:
            assert o.explanation_md == ""
            assert o.explanation_html_path is None
            assert o.explanation_md_path is None
            assert o.raw_json_path is None  # back-compat: maps to per_sample_reports=False

    def test_progress_callback_called(
        self, simple_reference, simple_test, fake_llm,
    ):
        events = []

        def on_progress(i, n, outcome):
            events.append((i, n, outcome.predicted_label))

        run_classification(
            reference=simple_reference,
            test=simple_test,
            label_column="activity",
            k=2,
            llm_client=fake_llm,
            progress_callback=on_progress,
        )
        assert [e[0] for e in events] == [1, 2]
        assert all(e[1] == 2 for e in events)


class TestPipelineValidationFailure:
    def test_missing_label_column_returns_no_outcomes(
        self, simple_reference, simple_test, fake_llm,
    ):
        out = run_classification(
            reference=simple_reference,
            test=simple_test,
            label_column="not-here",
            k=2,
            llm_client=fake_llm,
        )
        assert not out.validation.is_valid
        assert out.outcomes == []

    def test_raises_when_asked(self, simple_reference, simple_test, fake_llm):
        with pytest.raises(ValueError, match="validation failed"):
            run_classification(
                reference=simple_reference,
                test=simple_test,
                label_column="missing",
                k=2,
                llm_client=fake_llm,
                raise_on_validation_error=True,
            )

    def test_three_class_reference_now_supported(
        self, simple_reference, simple_test, fake_llm,
    ):
        # Multi-class: a third class is no longer a validation error.
        # The pipeline runs all 2 test samples successfully.
        ref = simple_reference.copy()
        ref.loc[0, "activity"] = "intermediate"
        out = run_classification(
            reference=ref, test=simple_test,
            label_column="activity", k=2, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert len(out.outcomes) == 2
        assert sorted(out.validation.summary["reference_classes"]) == [
            "active", "inactive", "intermediate",
        ]

    def test_descriptions_incomplete_blocks_run(
        self, simple_reference, simple_test, fake_llm,
    ):
        out = run_classification(
            reference=simple_reference, test=simple_test,
            label_column="activity", k=2,
            feature_descriptions={"alpha": "x"},
            llm_client=fake_llm,
        )
        assert not out.validation.is_valid


class TestPipelineErrorHandling:
    def test_per_sample_error_recorded(
        self, simple_reference, simple_test, fake_llm, monkeypatch,
    ):
        # Force the agent to raise on the first sample.
        original_predict_one = None

        def boom_then_normal(self, x_test, test_id=None):
            nonlocal original_predict_one
            if test_id == 0:
                raise RuntimeError("boom on sample 0")
            return original_predict_one(self, x_test, test_id=test_id)

        from molactagent.agent import MolActAgent
        original_predict_one = MolActAgent.predict_one
        monkeypatch.setattr(MolActAgent, "predict_one", boom_then_normal)

        out = run_classification(
            reference=simple_reference, test=simple_test,
            label_column="activity", k=2, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert out.outcomes[0].error == "boom on sample 0"
        assert out.outcomes[0].predicted_label == "Error"
        assert out.outcomes[1].error is None

    def test_output_dir_is_file(
        self, simple_reference, simple_test, fake_llm, tmp_workspace,
    ):
        bad = tmp_workspace / "imafile.txt"
        bad.write_text("not a directory")
        with pytest.raises(NotADirectoryError):
            run_classification(
                reference=simple_reference, test=simple_test,
                label_column="activity", k=2,
                llm_client=fake_llm, output_dir=bad,
            )

    def test_requires_llm(self, simple_reference, simple_test):
        with pytest.raises(ValueError, match="llm_config or llm_client"):
            run_classification(
                reference=simple_reference, test=simple_test,
                label_column="activity", k=2,
            )


class TestPipelineEdgeData:
    def test_single_test_sample(self, simple_reference, fake_llm):
        single = pd.DataFrame(
            {"alpha": [4.8], "beta": [5.0], "gamma": [4.7], "delta": [5.1]}
        )
        out = run_classification(
            reference=simple_reference, test=single,
            label_column="activity", k=2, llm_client=fake_llm,
        )
        assert len(out.outcomes) == 1
        assert out.outcomes[0].predicted_label in ("active", "inactive", "Inconclusive")

    def test_test_columns_reordered(self, simple_reference, fake_llm):
        scrambled = pd.DataFrame({
            "delta": [4.8, -5.0],
            "gamma": [4.7, -4.9],
            "beta": [5.0, -4.8],
            "alpha": [4.6, -5.2],
        })
        out = run_classification(
            reference=simple_reference, test=scrambled,
            label_column="activity", k=2, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert len(out.outcomes) == 2

    def test_k_larger_than_class_size_warns_then_runs(self, fake_llm):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(rng.normal(size=(4, 3)), columns=["a", "b", "c"])
        ref["lbl"] = ["yes", "yes", "no", "no"]
        tst = pd.DataFrame(rng.normal(size=(1, 3)), columns=["a", "b", "c"])
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=10, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert any("but k=10" in w for w in out.validation.warnings)
        assert len(out.outcomes) == 1

    def test_zero_one_labels(self, fake_llm):
        rng = np.random.default_rng(1)
        ref = pd.DataFrame(rng.normal(size=(6, 3)), columns=["a", "b", "c"])
        ref["activity"] = [1, 1, 1, 0, 0, 0]
        tst = pd.DataFrame(rng.normal(size=(2, 3)), columns=["a", "b", "c"])
        out = run_classification(
            reference=ref, test=tst, label_column="activity",
            k=2, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        # Multi-class: integer labels are stringified and kept verbatim;
        # both "0" and "1" appear in class_labels in canonical order.
        assert out.validation.summary["class_labels"] == ["0", "1"]

    def test_missing_features_filled(self, simple_reference, fake_llm):
        tst = pd.DataFrame({
            "alpha": [4.8, np.nan],
            "beta": [5.0, -5.1],
            "gamma": [4.9, -4.8],
            "delta": [5.0, -5.0],
        })
        out = run_classification(
            reference=simple_reference, test=tst,
            label_column="activity", k=2, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert len(out.outcomes) == 2

    def test_extra_test_columns_ignored(self, simple_reference, fake_llm):
        tst = pd.DataFrame({
            "alpha": [4.8, -5.1],
            "beta": [5.0, -5.0],
            "gamma": [4.9, -4.7],
            "delta": [5.0, -5.0],
            "noise": ["junk", "more junk"],
        })
        out = run_classification(
            reference=simple_reference, test=tst,
            label_column="activity", k=2, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert any("extra column" in w for w in out.validation.warnings)


class TestPipelineConfig:
    def test_no_descriptions_works(self, simple_reference, simple_test, fake_llm):
        out = run_classification(
            reference=simple_reference, test=simple_test,
            label_column="activity", k=2,
            feature_descriptions=None,
            llm_client=fake_llm,
        )
        assert out.validation.is_valid

    def test_complete_descriptions_passes(
        self, simple_reference, simple_test, simple_descriptions, fake_llm,
    ):
        out = run_classification(
            reference=simple_reference, test=simple_test,
            label_column="activity", k=2,
            feature_descriptions=simple_descriptions,
            llm_client=fake_llm,
        )
        assert out.validation.is_valid

    def test_descriptions_from_json_file(
        self, simple_reference, simple_test, simple_descriptions,
        fake_llm, tmp_workspace,
    ):
        path = tmp_workspace / "descs.json"
        path.write_text(json.dumps(simple_descriptions))
        out = run_classification(
            reference=simple_reference, test=simple_test,
            label_column="activity", k=2,
            feature_descriptions=path,
            llm_client=fake_llm,
        )
        assert out.validation.is_valid

    def test_background_string(self, simple_reference, simple_test, fake_llm):
        out = run_classification(
            reference=simple_reference, test=simple_test,
            label_column="activity", k=2,
            background="A study on cell-line activity for compound screening.",
            llm_client=fake_llm,
        )
        assert out.validation.is_valid

    def test_background_at_path(
        self, simple_reference, simple_test, fake_llm, tmp_workspace,
    ):
        bg = tmp_workspace / "bg.txt"
        bg.write_text("Background loaded from a file.\n")
        out = run_classification(
            reference=simple_reference, test=simple_test,
            label_column="activity", k=2,
            background=f"@{bg}",
            llm_client=fake_llm,
        )
        assert out.validation.is_valid


class TestSampleOutcome:
    def test_to_summary_row(self):
        o = SampleOutcome(
            test_index=0, sample_id="s0",
            predicted_label="active", confidence=0.9,
            true_label="active", elapsed_s=1.234,
        )
        row = o.to_summary_row()
        assert row["sample_id"] == "s0"
        assert row["predicted_label"] == "active"
        assert row["confidence"] == 0.9
        assert row["true_label"] == "active"
        assert row["elapsed_s"] == 1.23


# ── Backwards compatibility with the legacy MolActAgent interface ────


class TestBackwardCompat:
    def test_molactagent_direct_usage_still_works(self, fake_llm):
        """The pre-existing ``MolActAgent(...)`` ergonomics are unchanged."""
        rng = np.random.default_rng(0)
        n = 6
        X = np.vstack([rng.normal(loc=5, size=(n, 3)),
                       rng.normal(loc=-5, size=(n, 3))])
        y = np.array(["Pos"] * n + ["Neg"] * n)
        from molactagent import MolActAgent
        agent = MolActAgent(
            llm_client=fake_llm,
            k=2,
            feature_names=["a", "b", "c"],
            background="legacy api",
        )
        agent.fit(X, y)
        out = agent.predict_one(np.array([4.5, 5.0, 5.1]))
        assert out["final_label"] in ("Pos", "Neg", "Unknown")


# ── Tiny dataset (1 ref per class) end-to-end ───────────────────────


class TestMinimalDataset:
    def test_one_per_class(self, fake_llm):
        ref = pd.DataFrame({
            "a": [5.0, -5.0],
            "b": [5.0, -5.0],
            "c": [5.0, -5.0],
            "lbl": ["yes", "no"],
        })
        tst = pd.DataFrame({"a": [4.5], "b": [5.0], "c": [4.8]})
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=1, llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert len(out.outcomes) == 1
        # With only 1 sample per class, the agent's calibration formula
        # heavily discounts sample_size_score (= ref_min/10 = 0.1).
        cb = out.outcomes[0].raw_result["confidence_basis"]
        assert cb["sample_size_score"] <= 0.2

    def test_explanation_renders_with_one_per_class(self, fake_llm, tmp_workspace):
        ref = pd.DataFrame({
            "a": [5.0, -5.0],
            "b": [5.0, -5.0],
            "lbl": ["yes", "no"],
        })
        tst = pd.DataFrame({"a": [4.5], "b": [5.0]})
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=1, llm_client=fake_llm,
            output_dir=tmp_workspace / "tiny",
            per_sample_reports=True,
        )
        assert out.validation.is_valid
        html = pathlib.Path(out.outcomes[0].explanation_html_path).read_text()
        assert "MolActAgent Report" in html or "<html" in html


# ── LLM preflight (ping) behaviour ──────────────────────────────────


class _PingingLLM:
    """Fake LLM whose ``ping`` returns whatever we tell it to."""

    def __init__(self, ping_result: dict):
        self._ping = ping_result
        self.calls: list = []

    def ping(self, *, timeout: float = 30.0) -> dict:  # noqa: D401
        return dict(self._ping)

    def chat(self, *args, **kwargs):  # pragma: no cover - never called on failure
        self.calls.append((args, kwargs))
        return '{"label": "Pos", "confidence": 0.5}'

    @property
    def stats(self) -> dict:
        return {
            "calls": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "total_tokens": 0, "total_time_s": 0.0, "avg_time_s": 0.0,
        }

    @property
    def call_log(self) -> list[dict]:
        return []

    @property
    def throttle_count(self) -> int:
        return 0


class TestLLMPreflight:
    def _kwargs(self, simple_reference, simple_test):
        return dict(
            reference=simple_reference,
            test=simple_test,
            label_column="activity",
            k=2,
        )

    def test_auth_failure_becomes_configerror(
        self, simple_reference, simple_test,
    ):
        bad = _PingingLLM({
            "ok": False, "error_kind": "auth",
            "error": "AuthenticationError: 401", "elapsed_s": 0.1,
        })
        with pytest.raises(LLMConfigError, match="authentication failed"):
            run_classification(
                **self._kwargs(simple_reference, simple_test),
                llm_client=bad,
            )

    def test_model_failure_becomes_configerror(
        self, simple_reference, simple_test,
    ):
        bad = _PingingLLM({
            "ok": False, "error_kind": "model",
            "error": "NotFoundError: model 'x' not found", "elapsed_s": 0.1,
        })
        cfg = LLMConfig(
            api_key="k", base_url="u", model_name="nonexistent-model",
        )
        with pytest.raises(LLMConfigError) as e:
            run_classification(
                **self._kwargs(simple_reference, simple_test),
                llm_client=bad, llm_config=cfg,
            )
        assert "nonexistent-model" in str(e.value)

    def test_network_failure_becomes_configerror(
        self, simple_reference, simple_test,
    ):
        bad = _PingingLLM({
            "ok": False, "error_kind": "network",
            "error": "ConnectionError: name resolution", "elapsed_s": 0.1,
        })
        with pytest.raises(LLMConfigError, match="unreachable"):
            run_classification(
                **self._kwargs(simple_reference, simple_test),
                llm_client=bad,
            )

    def test_unknown_failure_becomes_configerror(
        self, simple_reference, simple_test,
    ):
        bad = _PingingLLM({
            "ok": False, "error_kind": "unknown",
            "error": "Something weird", "elapsed_s": 0.1,
        })
        with pytest.raises(LLMConfigError, match="preflight failed"):
            run_classification(
                **self._kwargs(simple_reference, simple_test),
                llm_client=bad,
            )

    def test_verify_llm_false_skips_ping(
        self, simple_reference, simple_test, fake_llm,
    ):
        """``verify_llm=False`` is the opt-out used by our test harness."""
        bad = _PingingLLM({
            "ok": False, "error_kind": "auth", "error": "401",
            "elapsed_s": 0.0,
        })
        # Failing ping would normally raise — but with verify_llm=False the
        # pipeline proceeds straight to chat() (which we divert back to the
        # good FakeLLM via monkey-patching the agent below). To keep this
        # test self-contained, we just pass the good fake_llm; the
        # important bit is that NO exception fires.
        out = run_classification(
            **self._kwargs(simple_reference, simple_test),
            llm_client=fake_llm, verify_llm=False,
        )
        assert out.validation.is_valid

    def test_ping_less_client_is_allowed(
        self, simple_reference, simple_test, fake_llm,
    ):
        """Clients that don't expose ``ping`` (e.g. legacy stubs) must skip
        preflight silently."""

        class NoPing:
            def chat(self, *a, **kw):
                return fake_llm.chat(*a, **kw)

            stats = fake_llm.stats
            call_log = fake_llm.call_log
            throttle_count = 0

        out = run_classification(
            **self._kwargs(simple_reference, simple_test),
            llm_client=NoPing(),
        )
        assert out.validation.is_valid


# ── --id-column / id_column ─────────────────────────────────────────


class TestIdColumn:
    def _data(self):
        ref = pd.DataFrame({
            "rid":  ["r1", "r2", "r3", "r4"],
            "x":    [5.0, 5.5, -5.0, -5.5],
            "y":    [5.1, 4.9, -5.1, -4.9],
            "lbl":  ["yes", "yes", "no", "no"],
        })
        tst = pd.DataFrame({
            "rid": ["alpha", "beta"],
            "x":   [4.8, -5.2],
            "y":   [5.0, -5.0],
        })
        return ref, tst

    def test_id_column_used_for_sample_id(self, fake_llm, tmp_workspace):
        ref, tst = self._data()
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=2, llm_client=fake_llm,
            id_column="rid",
            output_dir=tmp_workspace / "id_out",
            per_sample_reports=True,
        )
        assert out.validation.is_valid
        ids = [o.sample_id for o in out.outcomes]
        assert ids == ["alpha", "beta"]
        for o in out.outcomes:
            assert o.sample_id in pathlib.Path(o.raw_json_path).name

    def test_unknown_id_column_is_validation_error(self, fake_llm):
        """Non-existent id_column surfaces as a validation failure."""
        ref, tst = self._data()
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=2, llm_client=fake_llm, id_column="not_a_col",
        )
        assert not out.validation.is_valid
        assert out.outcomes == []
        assert any("id_column" in e for e in out.validation.errors)

    def test_duplicate_id_is_validation_error(self, fake_llm):
        """In-set duplicates block the run before any LLM call."""
        ref, tst = self._data()
        tst["rid"] = ["dup", "dup"]
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=2, llm_client=fake_llm, id_column="rid",
        )
        assert not out.validation.is_valid
        assert any("duplicate" in e for e in out.validation.errors)

    def test_cross_dataset_duplicate_blocks_run(self, fake_llm):
        """IDs must be unique across reference + test combined."""
        ref, tst = self._data()
        tst["rid"] = ["r1", "beta"]  # 'r1' collides with the reference
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=2, llm_client=fake_llm, id_column="rid",
        )
        assert not out.validation.is_valid
        assert any(
            "BOTH" in e or "both" in e.lower() or "globally unique" in e
            for e in out.validation.errors
        )

    def test_id_column_missing_in_reference_is_error(self, fake_llm):
        """Both datasets must carry the id_column now."""
        ref, tst = self._data()
        ref = ref.drop(columns=["rid"])
        out = run_classification(
            reference=ref, test=tst, label_column="lbl",
            k=2, llm_client=fake_llm, id_column="rid",
        )
        assert not out.validation.is_valid
        assert any(
            "id_column" in e and ("missing" in e.lower() or "not" in e.lower())
            for e in out.validation.errors
        )
