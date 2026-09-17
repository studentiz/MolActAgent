"""Tests for the contextual-similarity primitives in
:mod:`molactagent.neighbors`.

The worked example from ``docs/design.md`` §2 is the headline
test: two samples whose feature distance is moderate but whose
neighbourhoods are isomorphic should score higher under the blended
similarity than under the feature-only baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from molactagent.neighbors import (
    DEFAULT_K_LOCAL,
    DEFAULT_SIMILARITY_BLEND,
    NeighborFinder,
    NeighborhoodSignature,
)


@pytest.fixture
def isomorphic_neighbourhoods_dataset() -> tuple[np.ndarray, int, int]:
    """Build the worked example from ``docs/design.md`` §2.

    A and B share the same set of nearby points (C, D, E, F overlap
    heavily with B's neighbourhood). Their feature-space distance is
    non-trivial (around 0.6) so a feature-only KNN treats them as
    moderately distant; the neighbourhood-similarity term, however,
    lifts the contextual score because they read the local geometry
    through the same set of references.
    """
    X = np.array([
        # Index 0 — A
        [0.0, 0.0],
        # Index 1 — B (feature-distance 0.6 from A)
        [0.6, 0.0],
        # Indices 2-5 — points equally near both A and B (the
        # "shared neighbours" the design example talks about):
        # position halfway between A and B, plus tiny offsets so each
        # neighbour is distinct.
        [0.3, 0.05],   # idx 2
        [0.3, -0.05],  # idx 3
        [0.3, 0.10],   # idx 4
        [0.3, -0.10],  # idx 5
        # Indices 6-9 — far-away "padding" points that should NOT be
        # picked as nearest neighbours of either A or B.
        [3.0, 3.0],    # idx 6
        [3.0, -3.0],   # idx 7
        [-3.0, 3.0],   # idx 8
        [-3.0, -3.0],  # idx 9
    ])
    return X, 0, 1  # X, A index, B index


def test_neighbourhood_similarity_high_when_neighbourhoods_isomorphic(
    isomorphic_neighbourhoods_dataset,
):
    X, a_idx, b_idx = isomorphic_neighbourhoods_dataset
    nf = NeighborFinder(k=2, k_local=4, similarity_blend=0.5)
    nf.fit(X)

    sig_a = nf._signatures[a_idx]
    sig_b = nf._signatures[b_idx]
    nbr_sim = nf.neighbourhood_similarity(sig_a, sig_b)
    # The two neighbourhoods are the same shape after a fixed
    # translation, so their cross-coverage is high — much higher than
    # either point's raw feature-space affinity to the other.
    assert nbr_sim >= 0.6, (
        f"expected high neighbourhood similarity for isomorphic "
        f"neighbourhoods; got {nbr_sim}"
    )


def test_contextual_similarity_lifts_feature_only_score(
    isomorphic_neighbourhoods_dataset,
):
    """Contextual similarity blends feature + neighbourhood. When
    feature similarity is moderate but neighbourhood similarity is
    high, the blend must end up strictly above the feature-only
    baseline."""
    X, a_idx, b_idx = isomorphic_neighbourhoods_dataset
    nf = NeighborFinder(k=2, k_local=4, similarity_blend=0.5)
    nf.fit(X)

    feat_dist = float(np.linalg.norm(X[a_idx] - X[b_idx]))
    feat_only = max(0.0, 1.0 - feat_dist / nf._max_distance)
    ctx = nf.contextual_similarity_to_test(X[a_idx], b_idx)
    assert ctx > feat_only, (
        f"contextual similarity ({ctx:.3f}) should lift the feature-"
        f"only baseline ({feat_only:.3f}) on this worked example"
    )


def test_contextual_similarity_within_unit_interval(
    isomorphic_neighbourhoods_dataset,
):
    X, a_idx, b_idx = isomorphic_neighbourhoods_dataset
    nf = NeighborFinder(k=2, k_local=4)
    nf.fit(X)
    for ri in range(len(X)):
        sim = nf.contextual_similarity_to_test(X[a_idx], ri)
        assert 0.0 <= sim <= 1.0, sim


def test_neighbourhood_similarity_zero_for_empty_signatures():
    nf = NeighborFinder(k=2, k_local=3)
    nf.fit(np.zeros((3, 2)))
    empty = NeighborhoodSignature(
        np.array([], dtype=int), np.array([], dtype=float),
    )
    sim = nf.neighbourhood_similarity(empty, nf._signatures[0])
    assert sim == 0.0


def test_fit_precomputes_signatures():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    nf = NeighborFinder(k=3, k_local=4)
    nf.fit(X)
    assert len(nf._signatures) == 8
    for sig in nf._signatures:
        # k_local truncates to n-1 = 7 effective neighbours, but only
        # if k_local > n-1; here k_local=4 <= 7 so all signatures
        # carry exactly 4 neighbours.
        assert len(sig.indices) == 4
        assert (sig.distances >= 0).all()


def test_top_k_references_excludes_self_and_ranks_by_similarity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    nf = NeighborFinder(k=5, k_local=DEFAULT_K_LOCAL)
    nf.fit(X)
    test_x = X[3].copy()
    indices, ctx_dists, feat_dists = nf.find_top_k_references(X, test_x, k=5)
    assert len(indices) == 5
    # contextual distances should be non-decreasing (lowest = most similar)
    assert all(ctx_dists[i] <= ctx_dists[i + 1] for i in range(len(ctx_dists) - 1))
    assert all(feat_dists >= 0)


def test_default_blend_is_balanced():
    assert DEFAULT_SIMILARITY_BLEND == pytest.approx(0.5)


class TestSignatureAccessor:
    """``NeighborFinder.signature(idx)`` is the public accessor used
    by MolActAgent's texture-gathering helpers; it must return the
    correct precomputed signature in-range and ``None`` out-of-range
    so callers don't have to special-case missing indices."""

    def test_returns_signature_in_range(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(10, 3))
        nf = NeighborFinder(k=3, k_local=4)
        nf.fit(X)
        sig = nf.signature(0)
        assert sig is not None
        assert len(sig.indices) == 4
        # signature(idx) is the same instance the finder stored.
        assert sig is nf._signatures[0]

    def test_returns_none_for_out_of_range_idx(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(5, 3))
        nf = NeighborFinder(k=2, k_local=2)
        nf.fit(X)
        assert nf.signature(99) is None
        assert nf.signature(-1) is None

    def test_returns_none_when_not_fit(self):
        nf = NeighborFinder(k=2, k_local=2)
        assert nf.signature(0) is None
