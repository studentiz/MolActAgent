"""Scaffold-aware splitting for compound datasets.

Murcko scaffold splitting is the QSAR convention for assessing
generalisation: hold out entire scaffolds rather than individual
molecules so the test set genuinely has structures the training set
has not seen. Without it, random splits can leak structural cousins
between train and test and inflate validation metrics.

This module gives two helpers:

- :func:`compute_scaffolds` — Murcko scaffold SMILES per molecule.
  Mirrors the scaffold-extraction logic embedded in
  ``scripts/prepare_features.build_dataset_from_dataframe``.
- :func:`scaffold_split` — produce scaffold-disjoint train/test
  splits. The split is deterministic given a ``random_state`` —
  scaffolds are sorted by size descending so the largest groups
  are placed first (greedy filling), then a seeded shuffle decides
  which group goes to train vs. test.

The helpers also return per-row group indices so callers can plug
them into ``GroupKFold`` for nested CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: F401
        return Chem, MurckoScaffold
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "RDKit is required for scaffold splitting. Install with "
            "`pip install molactagent[chem]` (or "
            "`conda install -c conda-forge rdkit`)."
        ) from e


def compute_scaffolds(smiles: list[str]) -> tuple[list[str], list[int]]:
    """Compute Murcko scaffold SMILES for each input molecule.

    Returns ``(scaffolds, scaffold_ids)`` where ``scaffolds[i]`` is
    the canonical SMILES of molecule ``i``'s Murcko scaffold and
    ``scaffold_ids[i]`` is a small integer assigned per unique
    scaffold (useful as a ``GroupKFold`` group label).

    Molecules whose scaffold cannot be extracted (rare; usually
    fragments or invalid structures) get the empty string ``""``
    and are grouped together — they form their own pseudo-cluster
    so the splitter does not place them across train/test.
    """
    Chem, MurckoScaffold = _require_rdkit()
    scaffolds: list[str] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append("")
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(core))
        except Exception:
            scaffolds.append("")
    unique = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique)}
    return scaffolds, [scaffold_to_id[s] for s in scaffolds]


@dataclass
class ScaffoldSplit:
    """Output of :func:`scaffold_split`.

    Attributes
    ----------
    train_idx : np.ndarray
        Row indices of the training samples in the original input
        order.
    test_idx : np.ndarray
        Row indices of the test samples.
    train_scaffolds : list[str]
        Murcko-scaffold SMILES of the training molecules.
    test_scaffolds : list[str]
        Murcko-scaffold SMILES of the test molecules.
    scaffold_ids : np.ndarray
        Per-row scaffold group id; usable as ``groups=`` for
        ``sklearn.model_selection.GroupKFold``.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_scaffolds: list[str]
    test_scaffolds: list[str]
    scaffold_ids: np.ndarray


def scaffold_split(
    smiles: list[str],
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> ScaffoldSplit:
    """Split molecules into scaffold-disjoint train / test sets.

    Algorithm:

    1. Compute Murcko scaffolds for every input.
    2. Group molecules by scaffold; sort groups by descending size
       (largest first — gives the most predictable behaviour on
       small datasets).
    3. Within groups of the same size, apply a seeded shuffle so
       different ``random_state`` values produce different splits.
    4. Greedily assign each group to test until the cumulative
       test fraction crosses ``test_size``; assign the rest to
       train.

    The resulting split is **scaffold-disjoint**: no scaffold
    appears in both sides. For LOOCV-style protocols use the
    ``scaffold_ids`` field with ``GroupKFold(len(unique_scaffolds))``
    instead.

    Notes
    -----
    The actual test-set size may exceed ``test_size`` when the
    largest scaffold group is bigger than ``round(test_size * n)``.
    This is unavoidable: scaffold-disjoint splitting cannot break
    a group across train/test, so a single dominant scaffold can
    push the test fraction up. The function still guarantees at
    least one scaffold goes to test (so a degenerate "everything
    in train" outcome cannot happen).
    """
    n = len(smiles)
    if n == 0:
        return ScaffoldSplit(
            train_idx=np.array([], dtype=int),
            test_idx=np.array([], dtype=int),
            train_scaffolds=[],
            test_scaffolds=[],
            scaffold_ids=np.array([], dtype=int),
        )
    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"test_size must be in (0, 1); got {test_size}."
        )

    scaffolds, scaffold_ids = compute_scaffolds(list(smiles))
    scaffold_ids_arr = np.asarray(scaffold_ids, dtype=int)

    # Group by scaffold id.
    groups: dict[int, list[int]] = {}
    for row, gid in enumerate(scaffold_ids):
        groups.setdefault(gid, []).append(row)

    # Sort groups by size descending. Within a size tier, shuffle by
    # the supplied random_state so callers get different splits.
    rng = np.random.default_rng(random_state)
    by_size = sorted(groups.values(), key=lambda rows: -len(rows))
    # Group together same-size buckets and shuffle each bucket.
    grouped: list[list[list[int]]] = []
    if by_size:
        current_size = len(by_size[0])
        bucket: list[list[int]] = []
        for rows in by_size:
            if len(rows) != current_size:
                grouped.append(bucket)
                bucket = []
                current_size = len(rows)
            bucket.append(rows)
        grouped.append(bucket)
    for bucket in grouped:
        rng.shuffle(bucket)
    by_size = [rows for bucket in grouped for rows in bucket]

    target_test = int(round(test_size * n))
    test_rows: list[int] = []
    train_rows: list[int] = []
    for rows in by_size:
        # Greedy: if we still have room in the test set, the next
        # scaffold group goes to test; otherwise to train.
        if len(test_rows) + len(rows) <= target_test or not test_rows:
            test_rows.extend(rows)
        else:
            train_rows.extend(rows)

    # Anything we never assigned (only happens when target_test = 0)
    # falls into train.
    assigned = set(test_rows + train_rows)
    for i in range(n):
        if i not in assigned:
            train_rows.append(i)

    train_rows.sort()
    test_rows.sort()
    train_idx = np.array(train_rows, dtype=int)
    test_idx = np.array(test_rows, dtype=int)

    return ScaffoldSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        train_scaffolds=[scaffolds[i] for i in train_idx],
        test_scaffolds=[scaffolds[i] for i in test_idx],
        scaffold_ids=scaffold_ids_arr,
    )
