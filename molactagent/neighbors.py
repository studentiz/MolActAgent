"""Per-class nearest-reference search with contextual similarity.

Contextual similarity (per [docs/design.md](../docs/design.md) §2)
blends two ingredients:

- **Feature similarity** — the conventional inverse-distance score
  between two samples in feature space.
- **Neighbourhood similarity** — a measure of how isomorphic the two
  samples' k_local nearest-neighbour sets are.

The blend is symmetric and defaults to 50/50 (tunable). Both
components are normalised to [0, 1] so the blend itself stays in
[0, 1]; the corresponding "contextual distance" used internally is
just ``1 - sim``.

Public API
----------
:class:`NeighborhoodSignature`
    Indices and distances of a sample's k_local nearest neighbours.
:class:`NeighborResult`
    Per-class indices, contextual distances (used for ranking and
    adaptive top-M), and feature distances (used only for
    degeneracy detection so the existing ``np.allclose`` filter at
    the tuple level keeps working).
:class:`NeighborFinder`
    Wraps the per-class search plus the adaptive top-M selector.
    Must be ``fit()`` on the (scaled) reference matrix before
    ``find_local_references`` is called; ``find_local_references``
    will lazy-fit on its first call if the caller forgets, to keep
    legacy entry points working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


DEFAULT_K_LOCAL = 3
DEFAULT_SIMILARITY_BLEND = 0.5


@dataclass
class NeighborhoodSignature:
    """A sample's local context: indices of its ``k_local`` nearest
    reference neighbours and the corresponding distances.

    For reference samples the signature is precomputed at
    ``NeighborFinder.fit`` time; for the test sample it is computed
    on the fly inside ``find_local_references`` /
    ``contextual_similarity_to_test``.
    """

    indices: np.ndarray
    distances: np.ndarray

    @property
    def is_empty(self) -> bool:
        return len(self.indices) == 0


class NeighborResult:
    """Container for the per-class neighbour search.

    ``class_distances`` holds the *contextual* distance (1 - sim) for
    each picked neighbour, used both for ranking and for
    :meth:`NeighborFinder.select_top_m_classes`.
    ``class_feature_distances`` is the raw feature-space distance,
    kept around so the agent's tuple-level degeneracy filter (which
    detects test-vs-reference numerical identity) keeps working
    independently of the contextual blend.
    """

    __slots__ = (
        "class_indices",
        "class_distances",
        "class_feature_distances",
        "class_degenerate",
    )

    def __init__(
        self,
        class_indices: dict[str, np.ndarray],
        class_distances: dict[str, np.ndarray],
        class_feature_distances: dict[str, np.ndarray] | None = None,
        class_degenerate: dict[str, np.ndarray] | None = None,
    ):
        self.class_indices = class_indices
        self.class_distances = class_distances
        if class_feature_distances is None:
            class_feature_distances = {
                cls: np.asarray(d, dtype=float).copy()
                for cls, d in class_distances.items()
            }
        self.class_feature_distances = class_feature_distances
        if class_degenerate is None:
            class_degenerate = {
                cls: np.zeros(len(idx), dtype=bool)
                for cls, idx in class_indices.items()
            }
        self.class_degenerate = class_degenerate

    @property
    def classes(self) -> list[str]:
        """Sorted list of class labels with at least one neighbour."""
        return sorted(self.class_indices.keys())

    def min_distance(self, cls: str) -> float:
        """Smallest contextual distance for ``cls``; ``inf`` if absent."""
        d = self.class_distances.get(cls)
        if d is None or len(d) == 0:
            return float("inf")
        return float(np.min(d))


class NeighborFinder:
    """Finds k references per class ranked by contextual similarity.

    Workflow:

    1. ``fit(X_ref)`` once at agent fit time. Precomputes each
       reference's ``k_local`` nearest-neighbour signature and the
       max pairwise distance used to normalise similarity.
    2. ``find_local_references(X_ref, y_ref, x_test)`` per test
       sample. Returns the picked references per class with their
       contextual distances. ``X_ref`` and ``y_ref`` are also
       accepted at this point for backward compatibility with
       call sites that don't go through ``fit``; if no fit has
       happened yet, the call lazy-fits on the supplied ``X_ref``.
    3. ``select_top_m_classes(result)`` picks the candidate classes
       that are 'in contention' for the test sample (adaptive top-M).
    4. ``find_top_k_references(x_test, k)`` for the regression path,
       which doesn't filter by class.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        diversity_ratio: float = 0.1,
        degenerate_eps: float = 1e-6,
        m_threshold: float = 1.5,
        max_m: int = 4,
        k_local: int = DEFAULT_K_LOCAL,
        similarity_blend: float = DEFAULT_SIMILARITY_BLEND,
    ):
        self.k = k
        self.metric = metric
        self.diversity_ratio = diversity_ratio
        self.degenerate_eps = degenerate_eps
        self.m_threshold = m_threshold
        self.max_m = max_m
        self.k_local = k_local
        self.similarity_blend = similarity_blend

        self._signatures: list[NeighborhoodSignature] = []
        self._X_ref: np.ndarray | None = None
        self._max_distance: float = 1.0

    # ── fit-time precomputation ─────────────────────────────────────

    def signature(self, idx: int) -> NeighborhoodSignature | None:
        """Return the precomputed neighbourhood signature for the
        reference at ``idx``, or ``None`` when ``idx`` is out of
        range or the finder has not been fit yet.

        Public accessor so callers (e.g. ``MolActAgent``) don't have
        to reach into the private ``_signatures`` list.
        """
        if not self._signatures or idx < 0 or idx >= len(self._signatures):
            return None
        return self._signatures[idx]

    def fit(self, X_ref: np.ndarray) -> "NeighborFinder":
        """Precompute neighbourhood signatures for every reference.

        Stores per-reference ``k_local`` nearest indices/distances
        and the max pairwise distance ``self._max_distance`` used to
        normalise feature similarity. Cost: one O(N^2) distance
        matrix; trivial for N up to a few thousand.
        """
        self._X_ref = np.asarray(X_ref, dtype=float)
        n = len(self._X_ref)
        if n == 0:
            self._signatures = []
            self._max_distance = 1.0
            return self

        full = pairwise_distances(self._X_ref, self._X_ref, metric=self.metric)
        masked = full.copy()
        np.fill_diagonal(masked, np.inf)

        finite = np.isfinite(masked)
        if finite.any():
            self._max_distance = max(float(masked[finite].max()), 1e-9)
        else:
            self._max_distance = 1.0

        k_eff = min(self.k_local, max(0, n - 1))
        sigs: list[NeighborhoodSignature] = []
        for i in range(n):
            if k_eff == 0:
                sigs.append(NeighborhoodSignature(
                    np.array([], dtype=int),
                    np.array([], dtype=float),
                ))
                continue
            order = np.argsort(masked[i])[:k_eff]
            sigs.append(NeighborhoodSignature(order, masked[i][order]))
        self._signatures = sigs
        return self

    # ── similarity primitives (the design's witness layer) ─────────

    def neighbourhood_similarity(
        self,
        sig_a: NeighborhoodSignature,
        sig_b: NeighborhoodSignature,
    ) -> float:
        """Symmetric mean min-distance between two neighbourhoods,
        normalised to [0, 1] (1 = perfectly congruent neighbourhoods).

        Picks the simplest formulation called for by
        [docs/design.md](../docs/design.md) §2: average minimum
        cross-coverage in both directions, then normalised. The exact
        formula is an implementation choice; this one captures "how
        well does B's neighbourhood substitute for A's" without the
        extra cost of a bipartite matching.

        Why this specific formula:

        - **Symmetric** by construction (we average a→b and b→a) so
          ``sim(A, B) == sim(B, A)``. The design demands symmetry
          (§6.3) — a non-symmetric formula would make the
          contextual blend depend on which sample is "the test
          sample", which would be philosophically wrong.
        - **Mean-of-min** is forgiving of size differences. If A's
          neighbourhood has 3 points and B's has 5, every point in
          A still gets to vote for "is there a near-twin in B?"
          and the mean is taken over A's count.
        - **Normalised by max_distance** so the output lives in
          ``[0, 1]`` and composes cleanly with feature similarity
          in the blend.
        - **Linear in matrix sizes**, not quadratic like a true
          bipartite assignment. The cost is one ``k_local x
          k_local`` distance matrix per pair, which stays small
          for the default ``k_local=3``.

        Alternatives considered and rejected (per §6.2): bipartite
        matching (Hungarian algorithm, O(k³)); EMD (linear-program
        flavour, even more expensive); single-direction mean min
        (asymmetric — fails the symmetry requirement).
        """
        if sig_a.is_empty or sig_b.is_empty:
            return 0.0
        if self._X_ref is None:
            return 0.0
        a_pts = self._X_ref[sig_a.indices]
        b_pts = self._X_ref[sig_b.indices]
        d = pairwise_distances(a_pts, b_pts, metric=self.metric)
        # Mean min cross-coverage in each direction, then average.
        a_to_b = float(d.min(axis=1).mean())
        b_to_a = float(d.min(axis=0).mean())
        avg = 0.5 * (a_to_b + b_to_a)
        return max(0.0, 1.0 - avg / self._max_distance)

    def _test_signature(self, x_test: np.ndarray) -> NeighborhoodSignature:
        """Compute the test sample's neighbourhood signature on the fly."""
        if self._X_ref is None or len(self._X_ref) == 0:
            return NeighborhoodSignature(
                np.array([], dtype=int), np.array([], dtype=float),
            )
        d = pairwise_distances(
            x_test.reshape(1, -1), self._X_ref, metric=self.metric,
        )[0]
        n = len(self._X_ref)
        k_eff = min(self.k_local, n)
        order = np.argsort(d)[:k_eff]
        return NeighborhoodSignature(order, d[order])

    def contextual_similarity_to_test(
        self,
        x_test: np.ndarray,
        ref_idx: int,
        test_sig: NeighborhoodSignature | None = None,
    ) -> float:
        """Combined feature + neighbourhood similarity from a test
        point to a reference. Returns a value in [0, 1] where 1
        means "as similar as feature space lets us be"."""
        if (
            self._X_ref is None
            or ref_idx < 0
            or ref_idx >= len(self._X_ref)
        ):
            return 0.0
        feat_dist = float(pairwise_distances(
            x_test.reshape(1, -1),
            self._X_ref[ref_idx:ref_idx + 1],
            metric=self.metric,
        )[0, 0])
        feat_sim = max(0.0, 1.0 - feat_dist / self._max_distance)

        if test_sig is None:
            test_sig = self._test_signature(x_test)
        if ref_idx < len(self._signatures):
            ref_sig = self._signatures[ref_idx]
        else:
            ref_sig = NeighborhoodSignature(
                np.array([], dtype=int), np.array([], dtype=float),
            )
        nbr_sim = self.neighbourhood_similarity(test_sig, ref_sig)

        b = self.similarity_blend
        return b * feat_sim + (1.0 - b) * nbr_sim

    # ── per-class ranking (classification path) ─────────────────────

    def find_local_references(
        self,
        X_ref_scaled: np.ndarray,
        y_ref: np.ndarray,
        x_test_scaled: np.ndarray,
    ) -> NeighborResult:
        """Find ``k`` references per class, ranked by contextual
        similarity to the test sample.

        Lazy-fits if the caller hasn't called :meth:`fit` yet, so
        legacy entry points (e.g. test fixtures that swap a fresh
        finder onto an already-fitted agent) keep working.
        """
        if self._X_ref is None:
            self.fit(X_ref_scaled)
        # Defensive: if the caller passed a different reference
        # matrix than what we were fit on (shape mismatch or
        # different values), refit on the fresh matrix so distances
        # stay consistent. Cheap because fit is one-shot.
        elif (
            X_ref_scaled.shape != self._X_ref.shape
            or not np.array_equal(X_ref_scaled, self._X_ref)
        ):
            self.fit(X_ref_scaled)

        test_sig = self._test_signature(x_test_scaled)

        feat_dists_to_test = pairwise_distances(
            x_test_scaled.reshape(1, -1), self._X_ref, metric=self.metric,
        )[0]

        class_indices: dict[str, np.ndarray] = {}
        class_distances: dict[str, np.ndarray] = {}
        class_feature_distances: dict[str, np.ndarray] = {}
        class_degenerate: dict[str, np.ndarray] = {}

        unique_classes = sorted(np.unique(y_ref).tolist())
        for cls in unique_classes:
            mask = np.where(y_ref == cls)[0]
            if len(mask) == 0:
                continue
            sims = np.array([
                self.contextual_similarity_to_test(
                    x_test_scaled, ri, test_sig=test_sig,
                )
                for ri in mask
            ])
            ctx_dists = 1.0 - sims  # lower = more similar
            order = np.argsort(-sims)  # highest similarity first
            k_eff = min(self.k, len(mask))
            picked_local = order[:k_eff]
            picked_global = mask[picked_local]
            feat_d = feat_dists_to_test[picked_global]
            degen = feat_d < self.degenerate_eps

            class_indices[cls] = picked_global
            class_distances[cls] = ctx_dists[picked_local]
            class_feature_distances[cls] = feat_d
            class_degenerate[cls] = degen

        n_degen_total = sum(int(d.sum()) for d in class_degenerate.values())
        if n_degen_total:
            per_cls = ", ".join(
                f"{cls}={int(d.sum())}"
                for cls, d in class_degenerate.items()
                if int(d.sum()) > 0
            )
            logger.warning(
                f"Degenerate neighbours detected (feature distance < "
                f"{self.degenerate_eps}): {per_cls}"
            )

        return NeighborResult(
            class_indices=class_indices,
            class_distances=class_distances,
            class_feature_distances=class_feature_distances,
            class_degenerate=class_degenerate,
        )

    # ── adaptive top-M selection ────────────────────────────────────

    def select_top_m_classes(
        self,
        result: NeighborResult,
        candidate_classes: Iterable[str] | None = None,
    ) -> list[str]:
        """Return the candidate classes that are "in contention".

        A class belongs to the candidate set when its closest
        contextual distance is within ``m_threshold`` x the closest
        class's closest contextual distance. The set is capped at
        ``max_m`` and always contains at least the two closest
        classes (so a degenerate "only one class is in contention"
        outcome still gives the agent something to compare against).

        Empirically (per design notes): default ``m_threshold=1.5``
        keeps M=2 in roughly 80 percent of cases, opens to M=3 in
        roughly 19 percent, and to M=4 in the rare remaining ~1
        percent. M never falls to 1.
        """
        if candidate_classes is None:
            candidate_classes = result.classes
        scored = sorted(
            ((cls, result.min_distance(cls)) for cls in candidate_classes),
            key=lambda t: t[1],
        )
        scored = [(c, d) for c, d in scored if d != float("inf")]
        if len(scored) <= 2:
            # ≤2 classes total → no selection to make; everything's
            # in contention by definition.
            return [c for c, _ in scored]

        closest_d = scored[0][1]
        # The floor uses ``max(closest_d, degenerate_eps)`` to
        # protect against the degenerate case where the closest
        # class is at distance ~0 (e.g. the test sample sits
        # exactly on top of one reference). Without the eps,
        # ``floor = 0 * 1.5 = 0`` and every other class would be
        # excluded — even ones that are objectively close. The
        # eps gives the multiplier a non-zero base to scale.
        #
        # Why m_threshold = 1.5 (the default): empirically the
        # 1.0x-2.0x range gives a good split between
        # "comfortably resolved" (M=2) and "near-boundary" (M=3+)
        # cases. 1.5 keeps M=2 in roughly 80% of test samples on
        # the project's benchmarks (BSLT n=16, antiTB n≈30,
        # penguins n=24); raising it pushes more samples into M=3
        # and bloats the prompt; lowering it makes M=3 too rare
        # to catch genuinely-borderline cases.
        floor = max(closest_d, self.degenerate_eps) * self.m_threshold
        chosen = [c for c, d in scored if d <= floor]
        if len(chosen) < 2 and len(scored) >= 2:
            # Floor was too tight (e.g. one class is dramatically
            # closer than the others) — fall back to the two
            # closest so a contrastive comparison is always
            # possible. The system never falls back to M=1.
            chosen = [scored[0][0], scored[1][0]]
        if len(chosen) > self.max_m:
            # Cap at ``max_m`` (default 4) — beyond that the
            # prompt grows large enough to hurt LLM reasoning
            # stability without any benefit.
            chosen = chosen[: self.max_m]
        return chosen

    # ── flat top-k (regression path) ────────────────────────────────

    def find_top_k_references(
        self,
        X_ref_scaled: np.ndarray,
        x_test_scaled: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find top-k references by contextual similarity, ignoring
        class. Used by the regression path.

        Returns ``(indices, ctx_distances, feature_distances)`` for
        the picked references, ordered most-similar first. Lazy-fits
        if needed.
        """
        if self._X_ref is None:
            self.fit(X_ref_scaled)
        elif (
            X_ref_scaled.shape != self._X_ref.shape
            or not np.array_equal(X_ref_scaled, self._X_ref)
        ):
            self.fit(X_ref_scaled)

        n = len(self._X_ref)
        if n == 0 or k <= 0:
            return (
                np.array([], dtype=int),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )

        test_sig = self._test_signature(x_test_scaled)
        feat_dists = pairwise_distances(
            x_test_scaled.reshape(1, -1), self._X_ref, metric=self.metric,
        )[0]
        sims = np.array([
            self.contextual_similarity_to_test(
                x_test_scaled, ri, test_sig=test_sig,
            )
            for ri in range(n)
        ])
        ctx_dists = 1.0 - sims
        order = np.argsort(-sims)[:min(k, n)]
        return order, ctx_dists[order], feat_dists[order]
