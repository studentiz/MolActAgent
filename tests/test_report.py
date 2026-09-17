"""Structural assertions on the regression-aware report renderers
in :mod:`molactagent.report`.

These tests do not check pixel-level styling — they verify that the
HTML and Markdown contain the cards and fields a downstream
consumer depends on (predicted value, interval, drivers with
direction, tuple agreement, refused-to-predict signal, etc.).
"""

from __future__ import annotations

import pytest

from molactagent.pipeline import SampleOutcome
from molactagent.report import (
    render_html_report,
    render_interpretability_html,
    render_interpretability_md,
    render_interpretability_text,
)


# ── Per-sample regression HTML ──────────────────────────────────────


def _regression_result(
    *,
    predicted_value: float | None = 4250.0,
    interval: tuple[float, float] | None = (4000.0, 4500.0),
    confidence: float = 0.82,
    extrapolation_flagged: bool = False,
    extra_tuple_results: list[dict] | None = None,
) -> dict:
    """Build a minimal regression result dict shaped like the
    aggregator output for use in renderer tests."""
    return {
        "task": "regression",
        "test_index": 0,
        "predicted_value": predicted_value,
        "interval": interval,
        "confidence": confidence,
        "confidence_basis": {
            "agreement_score": 0.85,
            "avg_tuple_confidence": 0.80,
            "rationale": "test rationale",
            "basis_text": "synthetic basis",
        },
        "reason": "synthetic regression result",
        "tuple_results": extra_tuple_results or [],
        "driver_attribution": [
            {
                "feature": "alpha",
                "description": "Bill length",
                "n_cited": 4,
                "increases_strength": 2.5,
                "decreases_strength": 0.0,
                "mixed_strength": 0.0,
                "weighted_score": 2.5,
                "abs_strength": 2.5,
                "dominant_direction": "increases",
                "rationales": ["alpha drives the prediction up"],
            },
            {
                "feature": "beta",
                "description": "Bill depth",
                "n_cited": 4,
                "increases_strength": 0.2,
                "decreases_strength": 1.6,
                "mixed_strength": 0.0,
                "weighted_score": -1.4,
                "abs_strength": 1.8,
                "dominant_direction": "decreases",
                "rationales": ["beta pulls the prediction down"],
            },
        ],
        "extrapolation_flagged": extrapolation_flagged,
        "predictions": [4200.0, 4300.0, 4250.0, 4250.0],
        "reference_info": {
            "total_reference_samples": 23,
            "n_comparisons": 4,
            "target_range": [3300.0, 5750.0],
            "candidate_indices": [1, 5, 7, 12, 15],
            "n_degenerate_excluded": 0,
            "k_per_tuple": 5,
        },
        "test_feature_values": {"alpha": 45.1, "beta": 17.2, "gamma": 195.0},
    }


class TestRenderRegressionHtmlReport:
    def test_dispatches_on_task_field(self):
        result = _regression_result()
        html = render_html_report(result, explanation="")
        # The classification renderer emits 'final_label'-derived
        # text such as "Inconclusive"; the regression renderer should
        # emit a numeric badge instead.
        assert "Inconclusive" not in html
        assert "4250" in html

    def test_predicted_value_and_interval_visible(self):
        html = render_html_report(_regression_result(), explanation="")
        assert "4250" in html
        assert "4000" in html
        assert "4500" in html
        assert "82" in html  # confidence percent

    def test_drivers_card_shows_direction_arrows(self):
        html = render_html_report(_regression_result(), explanation="")
        assert "alpha" in html
        assert "beta" in html
        # New regression-only direction cues:
        assert "increases" in html.lower()
        assert "decreases" in html.lower()

    def test_robustness_card_renders_dots_along_target_axis(self):
        html = render_html_report(_regression_result(), explanation="")
        # One <div class="rb-dot" ...> per finite tuple prediction
        # (4 of them) plus one with the final-prediction modifier.
        assert html.count('class="rb-dot"') == 4
        assert html.count('class="rb-dot rb-dot-final"') == 1
        # Axis labels show the observed range endpoints.
        assert "3300" in html and "5750" in html

    def test_refused_prediction_renders_refused_badge(self):
        result = _regression_result(
            predicted_value=None,
            interval=(None, None),
            confidence=0.0,
        )
        html = render_html_report(result, explanation="")
        assert "REFUSED" in html

    def test_explanation_body_propagates(self):
        html = render_html_report(
            _regression_result(),
            explanation="## Executive Summary\n\nDemo body.",
        )
        assert "Executive Summary" in html
        assert "Demo body" in html


# ── Interpretability renderers (regression branch) ──────────────────


def _regression_outcome(idx: int, sample_id: str, true_value: str) -> SampleOutcome:
    raw = _regression_result()
    raw["test_index"] = idx
    return SampleOutcome(
        test_index=idx,
        sample_id=sample_id,
        predicted_label="4250",
        confidence=0.82,
        true_label=true_value,
        raw_result=raw,
        explanation_md="",
        explanation_html_path=None,
        explanation_md_path=None,
        raw_json_path=None,
        elapsed_s=1.0,
        task="regression",
        predicted_value=4250.0,
        interval=(4000.0, 4500.0),
    )


@pytest.fixture
def regression_outcomes() -> list[SampleOutcome]:
    return [
        _regression_outcome(0, "T0", "4400"),
        _regression_outcome(1, "T1", "3200"),
        _regression_outcome(2, "T2", "5950"),
    ]


class TestRegressionInterpretabilityRenderers:
    def test_text_banner_lists_predicted_and_true_per_sample(
        self, regression_outcomes,
    ):
        text = render_interpretability_text(regression_outcomes)
        assert "REGRESSION" in text.upper()
        for o in regression_outcomes:
            assert o.sample_id in text
            assert o.true_label in text
        # Predicted values come through as formatted numbers.
        assert "4250" in text

    def test_md_renders_predictions_table_with_abs_error(
        self, regression_outcomes,
    ):
        md = render_interpretability_md(regression_outcomes)
        assert "Predictions at a glance" in md
        # Markdown table header for the predictions glance row.
        assert "Predicted" in md
        assert "Interval" in md
        assert "Abs error" in md
        # Sample IDs reach the MD output.
        for o in regression_outcomes:
            assert o.sample_id in md

    def test_html_dispatches_to_regression_branch(
        self, regression_outcomes,
    ):
        html = render_interpretability_html(regression_outcomes)
        assert "regression" in html.lower()
        # Predictions-at-a-glance summary table is rendered.
        assert "Predictions at a glance" in html
        # No classification-only "Vote Consistency" card surfaces.
        assert "Vote Consistency" not in html
