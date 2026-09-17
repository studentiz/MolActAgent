"""Tests for the multi-class extension of MolActAgent.

These tests exercise the new code paths introduced when the binary
contrastive triplet pipeline was generalised into the multi-class
adaptive M-way pipeline. Binary (M=2) coverage already lives in the
other test modules; this file focuses on the M >= 3 cases and the
glue points that change when the candidate-class set is larger than
two.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from molactagent import MolActAgent
from molactagent.neighbors import NeighborFinder
from molactagent.pipeline import run_classification
from molactagent.report import (
    render_html_report,
    render_interpretability_html,
    render_interpretability_md,
)


# ── End-to-end: 3-class run via run_classification ──────────────────


class TestMulticlassPipelineEndToEnd:
    def test_multiclass_3way_prediction(
        self, multiclass_reference, multiclass_test, fake_llm,
    ):
        out = run_classification(
            reference=multiclass_reference,
            test=multiclass_test,
            label_column="bucket",
            k=3,
            llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert out.validation.summary["class_labels"] == ["high", "low", "mid"]
        assert len(out.outcomes) == 3

        # Row 0 was sampled from the "low" cluster; row 1 from "high".
        # The deterministic baseline (and FakeLLM agreement) should
        # produce those exact verdicts.
        labels = [o.predicted_label for o in out.outcomes]
        assert labels[0] == "low"
        assert labels[1] == "high"
        # Row 2 is at the centroid; the FakeLLM agrees with whatever
        # baseline emerges from the alignment tally, so we only assert
        # that the agent picked one of the candidate classes (or
        # Inconclusive) rather than a hallucinated value.
        assert labels[2] in {"low", "mid", "high", "Inconclusive"}

    def test_summary_csv_uses_real_class_names(
        self, multiclass_reference, multiclass_test, fake_llm, tmp_workspace,
    ):
        out_dir = tmp_workspace / "out"
        out = run_classification(
            reference=multiclass_reference,
            test=multiclass_test,
            label_column="bucket",
            k=3,
            llm_client=fake_llm,
            output_dir=out_dir,
            write_summary_csv=True,
            write_summary_json=True,
        )
        # Multi-class: predicted_label should be a real class name from
        # the reference set, never the legacy "Pos" / "Neg" sentinels.
        df = pd.read_csv(out.summary_csv_path)
        assert set(df["predicted_label"]).issubset(
            {"low", "mid", "high", "Inconclusive"}
        )
        summary = json.loads(pathlib.Path(out.summary_json_path).read_text())
        assert summary["class_labels"] == ["high", "low", "mid"]


# ── Adaptive top-M selection ────────────────────────────────────────


class TestAdaptiveTopM:
    def _agent(self, multiclass_reference, fake_llm):
        agent = MolActAgent(
            llm_client=fake_llm, k=3, max_workers=1,
        )
        feature_cols = [c for c in multiclass_reference.columns if c != "bucket"]
        agent.feature_names = feature_cols
        agent.fit(
            multiclass_reference[feature_cols].to_numpy(),
            multiclass_reference["bucket"].to_numpy(),
        )
        return agent

    def test_top_m_selects_two_when_clearly_in_one_cluster(
        self, multiclass_reference, fake_llm,
    ):
        agent = self._agent(multiclass_reference, fake_llm)
        # A test sample sitting deep inside the "low" cluster (mu = -5).
        x_test = np.array([-5.0, -5.0, -5.0, -5.0])
        x_scaled = agent._preprocessor.transform(x_test.reshape(1, -1))[0]
        nbr = agent._neighbor_finder.find_local_references(
            agent._X_ref_scaled, agent._y_ref, x_scaled,
        )
        candidates = agent._neighbor_finder.select_top_m_classes(nbr)
        # Closest cluster is "low"; runner-up is "mid"; "high" is far
        # enough that the 1.5x threshold excludes it.
        assert candidates == ["low", "mid"]

    def test_top_m_opens_to_three_when_test_is_central(
        self, multiclass_reference, fake_llm,
    ):
        # Construct a custom finder with a very permissive threshold so
        # the centroid sample reaches all three classes regardless of
        # scaling noise.
        nf = NeighborFinder(k=3, m_threshold=1000.0, max_m=4)
        # Re-fit a fresh agent so we can swap the finder out.
        agent = self._agent(multiclass_reference, fake_llm)
        agent._neighbor_finder = nf
        x_test = np.array([0.0, 0.0, 0.0, 0.0])
        x_scaled = agent._preprocessor.transform(x_test.reshape(1, -1))[0]
        nbr = nf.find_local_references(
            agent._X_ref_scaled, agent._y_ref, x_scaled,
        )
        candidates = nf.select_top_m_classes(nbr)
        assert sorted(candidates) == ["high", "low", "mid"]

    def test_top_m_capped_at_max_m(self, multiclass_reference, fake_llm):
        nf = NeighborFinder(k=3, m_threshold=100.0, max_m=2)
        agent = self._agent(multiclass_reference, fake_llm)
        agent._neighbor_finder = nf
        x_test = np.array([0.0, 0.0, 0.0, 0.0])
        x_scaled = agent._preprocessor.transform(x_test.reshape(1, -1))[0]
        nbr = nf.find_local_references(
            agent._X_ref_scaled, agent._y_ref, x_scaled,
        )
        candidates = nf.select_top_m_classes(nbr)
        # max_m=2 caps the candidate set even though every class is
        # within the (extremely permissive) distance threshold.
        assert len(candidates) == 2


# ── Baseline-class enforcement ──────────────────────────────────────


class TestBaselineClamping:
    def test_baseline_class_from_counts_argmax(self):
        from molactagent.agent import MolActAgent as MA
        # Unique winner -> winner is baseline.
        assert MA._baseline_class_from_counts({"a": 5, "b": 2, "c": 1}) == "a"
        # Tie at the top -> Unknown so the LLM has freedom to break the
        # deadlock with a domain argument.
        assert MA._baseline_class_from_counts({"a": 3, "b": 3}) == "Unknown"
        # All zero -> Unknown.
        assert MA._baseline_class_from_counts({"a": 0, "b": 0, "c": 0}) == "Unknown"
        # Empty -> Unknown.
        assert MA._baseline_class_from_counts({}) == "Unknown"

    def test_finalize_clamps_dissenting_label(self):
        from molactagent.agent import MolActAgent as MA
        parsed = {
            "label": "B",
            "confidence": 0.9,
            "key_drivers": [
                {"feature": "f0", "favored_class": "B", "strength": 0.5,
                 "test_value": 0, "favored_value": 0, "why": ""},
                {"feature": "f1", "favored_class": "B", "strength": 0.5,
                 "test_value": 0, "favored_value": 0, "why": ""},
            ],
            "summary": "...",
        }
        # Baseline says A wins. The finalizer is the last guard and
        # clamps any disagreeing label back to the baseline regardless
        # of the LLM's stated confidence.
        record = MA._finalize_tuple(
            parsed,
            ref_indices={"A": 0, "B": 1},
            candidate_classes=["A", "B"],
            disc_names=["f0", "f1"],
            n_aligned={"A": 5, "B": 2},
            n_neutral=0,
            baseline_class="A",
        )
        assert record["label"] == "A"
        assert record["baseline_class"] == "A"


# ── Aggregation: three-way Inconclusive ────────────────────────────


class TestThreeWayInconclusive:
    def test_aggregation_returns_inconclusive_on_three_way_split(self):
        from molactagent.agent import MolActAgent as MA
        # Hand-build a MolActAgent shell so we can call the aggregator
        # without going through the LLM-driven pipeline.
        agent = MA.__new__(MA)
        agent.feature_descriptions = None

        candidate_classes = ["A", "B", "C"]
        # Three tuples, one vote per class, all at equal confidence.
        tuple_results = [
            {"label": "A", "confidence": 0.6, "key_drivers": []},
            {"label": "B", "confidence": 0.6, "key_drivers": []},
            {"label": "C", "confidence": 0.6, "key_drivers": []},
        ]
        result = agent._aggregate_results(
            tuple_results,
            test_id=0,
            candidate_classes=candidate_classes,
            n_degenerate=0,
            class_reference_counts={"A": 4, "B": 4, "C": 4},
        )
        # No class is meaningfully ahead, so the verdict is Inconclusive.
        assert result["final_label"] == "Inconclusive"
        assert result["confidence"] >= 0.0
        assert result["vote_summary"]["per_class"]["A"]["votes"] == 1
        assert result["vote_summary"]["per_class"]["B"]["votes"] == 1
        assert result["vote_summary"]["per_class"]["C"]["votes"] == 1


# ── Report rendering: N+1 robustness segments ──────────────────────


class TestReportRendering:
    def _three_class_result(self):
        return {
            "final_label": "low",
            "confidence": 0.83,
            "test_index": 0,
            "candidate_classes": ["high", "low", "mid"],
            "confidence_basis": {
                "basis_text": "3 of 3 tuple comparisons were decisive.",
                "rationale": "weighted margin 0.50",
                "margin_score": 0.75,
                "sample_size_score": 0.4,
                "coverage_score": 1.0,
            },
            "reference_info": {
                "total_reference_samples": 12,
                "class_reference_counts": {"low": 4, "mid": 4, "high": 4},
                "class_neighbor_counts": {"low": 3, "mid": 3},
                "candidate_classes": ["low", "mid"],
                "n_comparisons": 9,
                "n_degenerate_excluded": 0,
            },
            "test_feature_values": {"alpha": -5.0, "beta": -5.0},
            "tuple_results": [],
            "driver_attribution": [
                {
                    "feature": "alpha", "description": "first",
                    "n_cited": 3,
                    "class_strength": {"low": 2.4, "mid": 0.0},
                    "weighted_score": 2.4,
                    "abs_strength": 2.4,
                    "dominant_class": "low",
                    "rationales": ["alpha is decisive"],
                },
            ],
            "key_features_per_class": {"low": ["alpha"], "mid": [], "high": []},
            "vote_summary": {
                "total": 9,
                "per_class": {
                    "low": {"votes": 7, "weighted": 5.6},
                    "mid": {"votes": 2, "weighted": 1.2},
                    "high": {"votes": 0, "weighted": 0.0},
                },
                "num_unknown": 0,
                "n_degenerate": 0,
            },
            "supporting_evidence": "alpha cited in 3 tuples toward low",
            "opposing_evidence": "",
            "uncertainty": "low",
            "reason": "vote tally: low=7, mid=2, high=0 (weighted 5.6 vs 1.2 vs 0.0)",
        }

    def test_html_robustness_card_has_n_plus_one_segments(self):
        result = self._three_class_result()
        html = render_html_report(
            result,
            explanation="## Executive Summary\n\nDemo.",
            class_labels=["high", "low", "mid"],
        )
        # 3 class segments + 1 Inconclusive = 4 coloured strips.
        # The CSS rule itself also references ``rb-seg`` once, so we
        # match the actual element marker (``class="rb-seg"``).
        assert html.count('class="rb-seg"') == 4
        # Each class label appears in the legend, and so does
        # "Inconclusive".
        assert ">low: " in html
        assert ">mid: " in html
        assert ">high: " in html
        assert ">Inconclusive: " in html
        # The verdict badge uses the canonical class name (no
        # "Pos"/"Neg" leaking through).
        assert "low</span>" in html

    def test_interpretability_md_has_one_column_per_class(self):
        # Build a tiny SampleOutcome stand-in so we can exercise the
        # aggregate renderer without spinning up the whole pipeline.
        class _Outcome:
            def __init__(self, sid, raw):
                self.sample_id = sid
                self.predicted_label = raw["final_label"]
                self.confidence = raw["confidence"]
                self.true_label = None
                self.error = None
                self.raw_result = raw
                self.explanation_html_path = None

        result = self._three_class_result()
        md = render_interpretability_md(
            [_Outcome("T_0", result)],
            class_labels=["high", "low", "mid"],
        )
        # Header row mentions every class plus Inconclusive.
        assert "high votes" in md
        assert "low votes" in md
        assert "mid votes" in md
        assert "Inconclusive" in md
        assert "high: 0" in md
        assert "low: 7" in md
        assert "mid: 2" in md

    def test_interpretability_html_table_columns_match_class_count(self):
        class _Outcome:
            def __init__(self, sid, raw):
                self.sample_id = sid
                self.predicted_label = raw["final_label"]
                self.confidence = raw["confidence"]
                self.true_label = None
                self.error = None
                self.raw_result = raw
                self.explanation_html_path = None

        result = self._three_class_result()
        html = render_interpretability_html(
            [_Outcome("T_0", result)],
            class_labels=["high", "low", "mid"],
        )
        # One <th> per class header in the summary table.
        assert "high votes (weighted)" in html
        assert "low votes (weighted)" in html
        assert "mid votes (weighted)" in html
        assert "Inconclusive" in html
        # Per-sample section uses the class names too.
        assert "<strong>low:</strong>" in html
        assert "<strong>mid:</strong>" in html


# ── A binary smoke test that reaches through the full pipeline ─────


class TestBinaryStillWorks:
    def test_binary_still_routes_through_unified_path(
        self, simple_reference, simple_test, fake_llm,
    ):
        out = run_classification(
            reference=simple_reference,
            test=simple_test,
            label_column="activity",
            k=2,
            llm_client=fake_llm,
        )
        assert out.validation.is_valid
        assert out.validation.summary["class_labels"] == ["active", "inactive"]
        # The first test row was sampled from the active cluster.
        assert out.outcomes[0].predicted_label == "active"
        assert out.outcomes[1].predicted_label == "inactive"
        # vote_summary is now per-class; both classes appear.
        vs = out.outcomes[0].raw_result["vote_summary"]
        assert "active" in vs["per_class"]
        assert "inactive" in vs["per_class"]


@pytest.fixture()
def patch_pipeline_to_fake(monkeypatch, fake_llm):
    """Compatibility shim — some tests in this module don't need it,
    but it's a no-op when the pipeline is already given an explicit
    ``llm_client``."""
    yield fake_llm
