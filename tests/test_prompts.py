"""Unit tests for :class:`molactagent.prompts.PromptBuilder`.

Covers the helpers added in the contextual-similarity / regression
refactor — ``format_reference_neighbourhood_block``,
``build_tuple_prompt`` (with texture), and
``build_regression_tuple_prompt`` — at the structural level so that
later prompt edits don't silently drop the witness baseline,
texture, or schema instructions the agent's validator depends on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from molactagent.prompts import PromptBuilder


# ── format_reference_neighbourhood_block ────────────────────────────


class TestFormatReferenceNeighbourhoodBlock:
    @pytest.fixture
    def builder(self):
        return PromptBuilder(
            feature_names=["alpha", "beta", "gamma"],
            background="Test background",
        )

    def test_renders_reference_and_neighbour_rows(self, builder):
        block = builder.format_reference_neighbourhood_block(
            ref_label_id="active_R7",
            ref_value="active",
            ref_features=np.array([1.0, 2.0, 3.0]),
            neighbour_ids=["R3", "R12", "R20"],
            neighbour_values=["active", "active", "inactive"],
            neighbour_features=[
                np.array([1.1, 2.1, 3.1]),
                np.array([0.9, 1.9, 2.9]),
                np.array([1.5, 2.5, 3.5]),
            ],
        )
        assert "active_R7" in block
        assert "R3" in block and "R12" in block and "R20" in block
        assert "alpha" in block and "beta" in block and "gamma" in block
        # The 'Identifier' / 'Value' header columns are present.
        assert "Identifier" in block
        assert "Value" in block

    def test_handles_long_identifiers_without_corruption(self, builder):
        long_id = "very_long_class_R12345"
        block = builder.format_reference_neighbourhood_block(
            ref_label_id=long_id,
            ref_value="some_value",
            ref_features=np.array([0.0, 0.0, 0.0]),
            neighbour_ids=["x", long_id + "_long_neighbour"],
            neighbour_values=["a", "b"],
            neighbour_features=[
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            ],
        )
        # Each row's contents survive the format step intact.
        assert long_id in block
        assert long_id + "_long_neighbour" in block

    def test_empty_neighbour_list_renders_just_the_reference(self, builder):
        block = builder.format_reference_neighbourhood_block(
            ref_label_id="solo_R0",
            ref_value="x",
            ref_features=np.array([1.0, 2.0, 3.0]),
            neighbour_ids=[],
            neighbour_values=[],
            neighbour_features=[],
        )
        assert "solo_R0" in block
        assert "alpha" in block

    def test_uses_feature_names_override(self):
        builder = PromptBuilder(feature_names=["a", "b", "c"])
        block = builder.format_reference_neighbourhood_block(
            ref_label_id="R0",
            ref_value="lbl",
            ref_features=np.array([1.0, 2.0, 3.0]),
            neighbour_ids=["R1"],
            neighbour_values=["lbl"],
            neighbour_features=[np.array([1.1, 2.1, 3.1])],
            feature_names=["custom_a", "custom_b", "custom_c"],
        )
        assert "custom_a" in block
        assert "custom_b" in block
        assert "custom_c" in block

# ── build_tuple_prompt with texture ─────────────────────────────────


class TestBuildTuplePromptWithTexture:
    @pytest.fixture
    def builder(self):
        return PromptBuilder(
            feature_names=["alpha", "beta", "gamma"],
            background="Discriminate active vs inactive.",
        )

    def test_texture_block_appears_when_provided(self, builder):
        msgs = builder.build_tuple_prompt(
            references=[
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0, 6.0]),
            ],
            class_names=["active", "inactive"],
            test_sample=np.array([2.0, 3.0, 4.0]),
            n_aligned={"active": 2, "inactive": 1},
            n_neutral=0,
            baseline_class="active",
            reference_neighbourhoods=[
                {
                    "ref_label_id": "active_R0",
                    "ref_value": "active",
                    "ref_features": np.array([1.0, 2.0, 3.0]),
                    "neighbour_ids": ["R5", "R6"],
                    "neighbour_values": ["active", "active"],
                    "neighbour_features": [
                        np.array([1.05, 2.05, 3.05]),
                        np.array([0.95, 1.95, 2.95]),
                    ],
                },
            ],
        )
        user = msgs[1]["content"]
        assert "PER-REFERENCE LOCAL CONTEXT" in user
        assert "active_R0" in user
        assert "R5" in user

    def test_no_texture_block_when_omitted(self, builder):
        msgs = builder.build_tuple_prompt(
            references=[
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0, 6.0]),
            ],
            class_names=["active", "inactive"],
            test_sample=np.array([2.0, 3.0, 4.0]),
            n_aligned={"active": 2, "inactive": 1},
            n_neutral=0,
            baseline_class="active",
        )
        user = msgs[1]["content"]
        assert "PER-REFERENCE LOCAL CONTEXT" not in user

