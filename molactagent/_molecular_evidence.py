"""Evidence extraction and prompts for molecular activity classification."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from ._molecules import CLASSES, MoleculeRecord, format_value as _fmt, tanimoto
from ._molecular_recording import ReplayError
from ._molecules import parse_model_json as safe_parse_json

FUNCTIONAL_PATTERNS = {
    "hydroxyl": "[OX2H]",
    "aryl_hydroxyl": "[c][OX2H]",
    "aliphatic_hydroxyl": "[CX4][OX2H]",
    "methoxy": "[OX2][CH3]",
    "ether": "[OD2]([#6])[#6]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=O)",
    "urea": "[NX3][CX3](=O)[NX3]",
    "nitrile": "[CX2]#N",
    "azide": "[N-]=[N+]=N",
    "halogen": "[F,Cl,Br,I]",
    "thioether": "[SX2]([#6])[#6]",
    "amine_methyl": "[NX3;!$(N-C=O)][CH3]",
    "tertiary_amine": "[NX3;!$(N-C=O);H0]",
    "aryl_exocyclic_alkene": "[c][CX3]=[CX3]",
    "aryl_methylene_carbocycle": "[c][CH2][C;R]",
}


@dataclass
class PairChoice:
    pair_id: int
    active_ref: int
    inactive_ref: int
    choice: str
    confidence: float
    parsed: dict[str, Any]
    raw_response: str
    call: dict[str, Any]
    parsed_ok: bool
    error: str | None = None
    active_side: str = "A"


@dataclass
class FinalChoice:
    label: str
    confidence: float
    parsed: dict[str, Any]
    raw_response: str
    call: dict[str, Any]
    parsed_ok: bool
    error: str | None = None


class ReferenceJudge:
    def __init__(self, llm, context, temperature, max_tokens, response_format):
        self.llm = llm
        self.context = context
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_response_format = response_format
        self.sidechain_min_fragments = 4

    @staticmethod
    def select_pairs(
        records: list[MoleculeRecord],
        query_idx: int,
        train_idx: list[int],
        n_pairs: int,
        strategy: str = "legacy",
    ) -> list[tuple[int, int, float]]:
        query = records[query_idx]
        active = [i for i in train_idx if records[i].label == "active"]
        inactive = [i for i in train_idx if records[i].label == "inactive"]
        q_bits = set(records[query_idx].morgan.GetOnBits())
        candidates = []
        for a in active:
            sim_a = tanimoto(query.morgan, records[a].morgan)
            a_bits = set(records[a].morgan.GetOnBits())
            for i in inactive:
                sim_i = tanimoto(query.morgan, records[i].morgan)
                pair_sim = tanimoto(records[a].morgan, records[i].morgan)
                if strategy == "legacy":
                    score = 0.6 * pair_sim + 0.4 * 0.5 * (sim_a + sim_i)
                else:
                    i_bits = set(records[i].morgan.GetOnBits())
                    contrast = a_bits ^ i_bits
                    query_difference = q_bits ^ a_bits | q_bits ^ i_bits
                    shared_query_difference = (
                        (q_bits ^ a_bits) & (q_bits ^ i_bits)
                    ) - contrast
                    shared_fraction = (
                        len(shared_query_difference) / len(query_difference)
                        if query_difference
                        else 0.0
                    )
                    aligned = sum(
                        (
                            1
                            for bit in contrast
                            if (bit in q_bits) == (bit in a_bits)
                            and (bit in q_bits) != (bit in i_bits)
                        )
                    )
                    alignment = aligned / len(contrast) if contrast else 0.0
                    score = (
                        0.4 * 0.5 * (sim_a + sim_i)
                        + 0.2 * pair_sim
                        + 0.35 * (1.0 - shared_fraction)
                        + 0.05 * alignment
                    )
                candidates.append(
                    {"a": a, "i": i, "score": float(score), "pair_sim": float(pair_sim)}
                )
        candidates.sort(
            key=lambda item: (-item["score"], -item["pair_sim"], item["a"], item["i"])
        )
        selected = []
        while candidates and len(selected) < n_pairs:
            best_idx = 0
            best_adjusted = -np.inf
            for idx, candidate in enumerate(candidates):
                active_reuse = sum((item["a"] == candidate["a"] for item in selected))
                inactive_reuse = sum((item["i"] == candidate["i"] for item in selected))
                adjusted = candidate["score"] - 0.035 * (active_reuse + inactive_reuse)
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_idx = idx
            chosen = candidates.pop(best_idx)
            selected.append(chosen)
        return [(item["a"], item["i"], float(item["score"])) for item in selected]

    @staticmethod
    def select_panel(
        records: list[MoleculeRecord],
        query_idx: int,
        train_idx: list[int],
        n_per_class: int = 4,
    ) -> tuple[list[int], list[int]]:
        query = records[query_idx]
        active = sorted(
            (i for i in train_idx if records[i].label == "active"),
            key=lambda i: (-tanimoto(query.morgan, records[i].morgan), i),
        )
        inactive = sorted(
            (i for i in train_idx if records[i].label == "inactive"),
            key=lambda i: (-tanimoto(query.morgan, records[i].morgan), i),
        )
        return (active[:n_per_class], inactive[:n_per_class])

    @staticmethod
    def _element_signature(ref: MoleculeRecord) -> str:
        counts = Counter((atom.GetSymbol() for atom in ref.mol.GetAtoms()))
        keys = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si")
        return ", ".join((f"{key}={counts[key]}" for key in keys if counts[key]))

    @staticmethod
    def _halogen_count(ref: MoleculeRecord) -> int:
        return sum(
            (atom.GetSymbol() in {"F", "Cl", "Br", "I"} for atom in ref.mol.GetAtoms())
        )

    @staticmethod
    def _functional_group_counts(ref: MoleculeRecord) -> dict[str, int]:
        counts = {}
        for name, smarts in FUNCTIONAL_PATTERNS.items():
            query = Chem.MolFromSmarts(smarts)
            count = len(ref.mol.GetSubstructMatches(query)) if query else 0
            if count:
                counts[name] = count
        return counts

    @staticmethod
    def _functional_signature(ref: MoleculeRecord) -> str:
        parts = [
            f"{name}={count}"
            for name, count in ReferenceJudge._functional_group_counts(ref).items()
        ]
        return ", ".join(parts) if parts else "none"

    @staticmethod
    def _bond_signature(ref: MoleculeRecord) -> str:
        counts = Counter(
            (
                "aromatic"
                if bond.GetIsAromatic()
                else {1.0: "single", 1.5: "aromatic", 2.0: "double", 3.0: "triple"}.get(
                    float(bond.GetBondTypeAsDouble()), "other"
                )
                for bond in ref.mol.GetBonds()
            )
        )
        return (
            ", ".join(
                (
                    f"{name}={counts[name]}"
                    for name in ("single", "double", "triple", "aromatic", "other")
                    if counts[name]
                )
            )
            or "none"
        )

    @staticmethod
    def _has_opposite_exact_reference(
        query: MoleculeRecord, records: list[MoleculeRecord], train_idx: list[int]
    ) -> bool:
        query_key = Chem.MolToSmiles(query.mol, isomericSmiles=False)
        query_isomer = Chem.MolToSmiles(query.mol, isomericSmiles=True)
        return any(
            (
                Chem.MolToSmiles(records[idx].mol, isomericSmiles=False) == query_key
                and Chem.MolToSmiles(records[idx].mol, isomericSmiles=True)
                != query_isomer
                for idx in train_idx
            )
        )

    @staticmethod
    def _ring_signature(ref: MoleculeRecord) -> str:
        ring_atoms = Counter(
            (atom.GetSymbol() for atom in ref.mol.GetAtoms() if atom.IsInRing())
        )
        ring_sizes = Counter((len(ring) for ring in ref.mol.GetRingInfo().AtomRings()))
        atom_text = ", ".join(
            (
                f"{key}={ring_atoms[key]}"
                for key in ("C", "N", "O", "S")
                if ring_atoms[key]
            )
        )
        size_text = ", ".join(
            (f"{size}-membered={count}" for size, count in sorted(ring_sizes.items()))
        )
        return f"ring_atoms[{atom_text or 'none'}]; sizes[{size_text or 'none'}]"

    @staticmethod
    def _sidechain_signature(ref: MoleculeRecord) -> str:
        """Show Murcko-core substituents so local R-group changes are explicit."""
        try:
            core = MurckoScaffold.GetScaffoldForMol(ref.mol)
            sidechains = Chem.ReplaceCore(ref.mol, core, labelByIndex=True)
            if sidechains is None:
                return "none"
            fragments = Chem.GetMolFrags(sidechains, asMols=True, sanitizeFrags=True)
        except Exception:
            return "unavailable"
        values = sorted(
            (Chem.MolToSmiles(fragment, isomericSmiles=True) for fragment in fragments)
        )
        return " | ".join(values) if values else "none"

    @staticmethod
    def _linker_type(ref: MoleculeRecord) -> str:
        """Classify the terminal substituent's attachment/linker chemistry."""
        fragments = ReferenceJudge._sidechain_signature(ref).split(" | ")
        best_text = ""
        best_score = -1
        for text in fragments:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                continue
            ring_atoms = sum((atom.IsInRing() for atom in mol.GetAtoms()))
            heavy = sum((atom.GetSymbol() != "*" for atom in mol.GetAtoms()))
            score = 100 * ring_atoms + heavy
            if score > best_score:
                best_score = score
                best_text = text
        if not best_text:
            return "none"
        if best_text.startswith("*O"):
            return "O-linker"
        if best_text.startswith("*C(=O)"):
            return "carbonyl"
        if best_text.startswith("*C=C") or best_text.startswith("*C#"):
            return "unsaturated-direct"
        if best_text.startswith("*CC") or best_text.startswith("*CCC"):
            return "alkyl-linker"
        if best_text.startswith("*C"):
            return "direct-carbon"
        if best_text.startswith("*N"):
            return "N-linker"
        if best_text.startswith("*S"):
            return "S-linker"
        return "other"

    @classmethod
    def _terminal_similarity(cls, query: MoleculeRecord, ref: MoleculeRecord) -> float:
        """Compare Murcko sidechain fragments by substructure, not text."""
        query_fragments = cls._sidechain_signature(query).split(" | ")
        ref_fragments = cls._sidechain_signature(ref).split(" | ")
        best = 0.0
        for query_text in query_fragments:
            query_mol = Chem.MolFromSmiles(query_text)
            if query_mol is None:
                continue
            query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
            for ref_text in ref_fragments:
                ref_mol = Chem.MolFromSmiles(ref_text)
                if ref_mol is None:
                    continue
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                best = max(best, tanimoto(query_fp, ref_fp))
        if best == 0.0:
            return SequenceMatcher(
                None, cls._sidechain_signature(query), cls._sidechain_signature(ref)
            ).ratio()
        return best

    @classmethod
    def _sidechain_fragment_count(cls, ref: MoleculeRecord) -> int:
        signature = cls._sidechain_signature(ref)
        if signature in {"none", "unavailable"}:
            return 0
        return len(signature.split(" | "))

    @staticmethod
    def _stereo_signature(ref: MoleculeRecord) -> str:
        try:
            centers = Chem.FindMolChiralCenters(
                ref.mol, includeUnassigned=True, useLegacyImplementation=False
            )
        except TypeError:
            centers = Chem.FindMolChiralCenters(ref.mol, includeUnassigned=True)
        if not centers:
            return "none specified"
        return "; ".join((f"atom {idx}: {cip}" for idx, cip in centers))

    @staticmethod
    def _maccs_differences(
        query: MoleculeRecord, active: MoleculeRecord, inactive: MoleculeRecord
    ) -> str:
        q_bits = query.maccs_bits
        a_bits = active.maccs_bits
        i_bits = inactive.maccs_bits
        lines = []
        for key in sorted(a_bits ^ i_bits)[:10]:
            pattern = MACCSkeys.smartsPatts.get(key, ("?", 0))[0]
            lines.append(
                f"  bit {key} ({pattern}); test={int(key in q_bits)}, A={int(key in a_bits)}, B={int(key in i_bits)}"
            )
        shared_query_difference = ((q_bits ^ a_bits) & (q_bits ^ i_bits)) - (
            a_bits ^ i_bits
        )
        for key in sorted(shared_query_difference)[:5]:
            pattern = MACCSkeys.smartsPatts.get(key, ("?", 0))[0]
            lines.append(
                f"  test-specific bit {key} ({pattern}); test={int(key in q_bits)}, A={int(key in a_bits)}, B={int(key in i_bits)}"
            )
        return "\n".join(lines) if lines else "  none in MACCS keys"

    @staticmethod
    def _class_contrast_features(
        query: MoleculeRecord,
        records: list[MoleculeRecord],
        train_idx: list[int],
        limit: int = 10,
    ) -> str:
        active = [idx for idx in train_idx if records[idx].label == "active"]
        inactive = [idx for idx in train_idx if records[idx].label == "inactive"]
        if not active or not inactive:
            return "  no class contrast available"
        scored = []
        for key in range(1, 167):
            active_count = sum((key in records[idx].maccs_bits for idx in active))
            inactive_count = sum((key in records[idx].maccs_bits for idx in inactive))
            active_fraction = active_count / len(active)
            inactive_fraction = inactive_count / len(inactive)
            difference = active_fraction - inactive_fraction
            if abs(difference) >= 0.35:
                scored.append(
                    (
                        -abs(difference),
                        key,
                        active_count,
                        inactive_count,
                        active_fraction,
                        inactive_fraction,
                    )
                )
        scored.sort()
        lines = []
        for _, key, active_count, inactive_count, af, inf in scored[:limit]:
            pattern = MACCSkeys.smartsPatts.get(key, ("?", 0))[0]
            lines.append(
                f"  bit {key} ({pattern}): active={active_count}/{len(active)}, inactive={inactive_count}/{len(inactive)}, test={int(key in query.maccs_bits)}"
            )
        return "\n".join(lines) if lines else "  no strong MACCS contrast"

    @staticmethod
    def _stereochemical_matched_pairs(
        query: MoleculeRecord,
        records: list[MoleculeRecord],
        train_idx: list[int],
        limit: int = 6,
    ) -> str:
        groups: dict[str, list[int]] = {}
        for idx in train_idx:
            key = Chem.MolToSmiles(records[idx].mol, isomericSmiles=False)
            groups.setdefault(key, []).append(idx)
        query_key = Chem.MolToSmiles(query.mol, isomericSmiles=False)
        exact_lines = []
        related_lines = []
        for key, members in groups.items():
            active = [idx for idx in members if records[idx].label == "active"]
            inactive = [idx for idx in members if records[idx].label == "inactive"]
            if not active or not inactive:
                continue
            marker = (
                "TEST_CONSTITUTION_MATCH"
                if key == query_key
                else "RELATED_CONSTITUTION"
            )
            target = exact_lines if key == query_key else related_lines
            for a in active[:2]:
                for i in inactive[:2]:
                    target.append(
                        f"  [{marker}] active {Chem.MolToSmiles(records[a].mol, isomericSmiles=True)} pChEMBL={_fmt(records[a].pchembl)} stereo[{ReferenceJudge._stereo_signature(records[a])}] | inactive {Chem.MolToSmiles(records[i].mol, isomericSmiles=True)} pChEMBL={_fmt(records[i].pchembl)} stereo[{ReferenceJudge._stereo_signature(records[i])}]"
                    )
                    if len(target) >= limit:
                        break
                if len(target) >= limit:
                    break
        lines = exact_lines + related_lines
        return "\n".join(lines) if lines else "  no matched stereochemical pairs"

    @staticmethod
    def _same_scaffold_neighbors(
        query: MoleculeRecord,
        records: list[MoleculeRecord],
        train_idx: list[int],
        limit: int = 8,
    ) -> str:
        """Show measured references sharing the test's exact Murcko scaffold."""
        if not query.scaffold:
            return "  no Murcko scaffold; use the nearest local analogs"
        candidates = sorted(
            (idx for idx in train_idx if records[idx].scaffold == query.scaffold),
            key=lambda idx: (-tanimoto(query.morgan, records[idx].morgan), idx),
        )
        lines = []
        for idx in candidates[:limit]:
            ref = records[idx]
            lines.append(
                f"  ref={idx}; label={ref.label}; pChEMBL={_fmt(ref.pchembl)}; similarity={_fmt(tanimoto(query.morgan, ref.morgan))}; halogens={ReferenceJudge._halogen_count(ref)}; SMILES={Chem.MolToSmiles(ref.mol, isomericSmiles=True)}; linker[{ReferenceJudge._linker_type(ref)}]; sidechains[{ReferenceJudge._sidechain_signature(ref)}]"
            )
        return "\n".join(lines) if lines else "  no exact-scaffold reference"

    @staticmethod
    def _constitutional_isomer_evidence(
        query: MoleculeRecord,
        records: list[MoleculeRecord],
        train_idx: list[int],
        limit: int = 10,
    ) -> str:
        """Show measured same-formula isomers with different connectivity.

        These references isolate regioisomerism and scaffold placement while
        holding atom composition fixed.  The table is descriptive evidence for
        the LLM, not a label rule.
        """
        formula = rdMolDescriptors.CalcMolFormula(query.mol)
        candidates = [
            idx
            for idx in train_idx
            if rdMolDescriptors.CalcMolFormula(records[idx].mol) == formula
            and Chem.MolToSmiles(records[idx].mol, isomericSmiles=True)
            != Chem.MolToSmiles(query.mol, isomericSmiles=True)
        ]
        candidates.sort(
            key=lambda idx: (-tanimoto(query.morgan, records[idx].morgan), idx)
        )
        lines = []
        for idx in candidates[:limit]:
            ref = records[idx]
            lines.append(
                f"  ref={idx}; label={ref.label}; pChEMBL={_fmt(ref.pchembl)}; similarity={_fmt(tanimoto(query.morgan, ref.morgan))}; same_murcko_scaffold={bool(query.scaffold) and ref.scaffold == query.scaffold}; SMILES={Chem.MolToSmiles(ref.mol, isomericSmiles=True)}; scaffold={ref.scaffold or '(acyclic)'}; topology[{ReferenceJudge._topology_signature(ref)}]; linker[{ReferenceJudge._linker_type(ref)}]; functional_groups[{ReferenceJudge._functional_signature(ref)}]"
            )
        if not lines:
            return "  no same-formula constitutional isomer"
        return f"  query_formula={formula}\n" + "\n".join(lines)

    @staticmethod
    def _halogen_pattern_evidence(
        query: MoleculeRecord,
        records: list[MoleculeRecord],
        train_idx: list[int],
        limit: int = 4,
    ) -> str:
        """Show exact-count halogen analogs plus the closest one-halogen neighbor."""
        query_halogens = ReferenceJudge._halogen_count(query)
        if not query_halogens:
            return "  no query halogens"
        lines = []
        for label in CLASSES:
            candidates = [
                idx
                for idx in train_idx
                if records[idx].label == label
                and ReferenceJudge._halogen_count(records[idx]) == query_halogens
            ]
            candidates.sort(
                key=lambda idx: (-tanimoto(query.morgan, records[idx].morgan), idx)
            )
            for idx in candidates[:limit]:
                ref = records[idx]
                lines.append(
                    f"  same_count_{label}; ref={idx}; pChEMBL={_fmt(ref.pchembl)}; similarity={_fmt(tanimoto(query.morgan, ref.morgan))}; same_murcko_scaffold={bool(query.scaffold) and ref.scaffold == query.scaffold}; linker[{ReferenceJudge._linker_type(ref)}]; sidechains[{ReferenceJudge._sidechain_signature(ref)}]; SMILES={Chem.MolToSmiles(ref.mol, isomericSmiles=True)}"
                )
        nearest = sorted(
            train_idx,
            key=lambda idx: (-tanimoto(query.morgan, records[idx].morgan), idx),
        )
        for idx in nearest:
            ref = records[idx]
            ref_halogens = ReferenceJudge._halogen_count(ref)
            if (
                abs(ref_halogens - query_halogens) == 1
                and bool(query.scaffold)
                and (ref.scaffold == query.scaffold)
            ):
                lines.append(
                    f"  closest_same_scaffold_one_halogen_delta; ref={idx}; label={ref.label}; halogens={ref_halogens}; pChEMBL={_fmt(ref.pchembl)}; similarity={_fmt(tanimoto(query.morgan, ref.morgan))}; sidechains[{ReferenceJudge._sidechain_signature(ref)}]; SMILES={Chem.MolToSmiles(ref.mol, isomericSmiles=True)}"
                )
                break
        return "\n".join(lines) if lines else "  no exact-count halogen analog"

    @staticmethod
    def _cip_signature(ref: MoleculeRecord) -> str:
        try:
            centers = Chem.FindMolChiralCenters(
                ref.mol, includeUnassigned=True, useLegacyImplementation=False
            )
        except TypeError:
            centers = Chem.FindMolChiralCenters(ref.mol, includeUnassigned=True)
        values = sorted((str(cip) for _, cip in centers))
        return "[" + ",".join(values) + "]" if values else "[]"

    @staticmethod
    def _topology_signature(ref: MoleculeRecord) -> str:
        """Describe generic linker length and heterocycle attachment topology."""
        mol = ref.mol
        rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
        aromatic = [
            ring
            for ring in rings
            if len(ring) >= 6
            and all((mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        ]

        def shortest_path(start: int, end: int) -> list[int] | None:
            queue = deque([(start, [start])])
            seen = {start}
            while queue:
                node, path = queue.popleft()
                if node == end:
                    return path
                for neighbor in mol.GetAtomWithIdx(node).GetNeighbors():
                    idx = neighbor.GetIdx()
                    if idx not in seen:
                        seen.add(idx)
                        queue.append((idx, path + [idx]))
            return None

        linker_atoms = 0
        for pos, first in enumerate(aromatic):
            for second in aromatic[pos + 1 :]:
                if first & second:
                    continue
                for start in first:
                    for end in second:
                        path = shortest_path(start, end)
                        if path is None:
                            continue
                        non_ring = sum(
                            (
                                1
                                for idx in path
                                if not any((idx in ring for ring in aromatic))
                            )
                        )
                        linker_atoms = max(linker_atoms, non_ring)
        attachments = []
        for ring in rings:
            if len(ring) != 5 or not any(
                (mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in ring)
            ):
                continue
            for attachment in ring:
                for neighbor in mol.GetAtomWithIdx(attachment).GetNeighbors():
                    anchor = neighbor.GetIdx()
                    if anchor in ring or not mol.GetAtomWithIdx(anchor).GetIsAromatic():
                        continue
                    aromatic_ring = next(
                        (candidate for candidate in aromatic if anchor in candidate),
                        None,
                    )
                    if aromatic_ring is None:
                        continue
                    for ring_atom in aromatic_ring:
                        for substituent in mol.GetAtomWithIdx(ring_atom).GetNeighbors():
                            sub_idx = substituent.GetIdx()
                            if sub_idx in aromatic_ring or sub_idx in ring:
                                continue
                            path = shortest_path(anchor, ring_atom)
                            if path is not None:
                                attachments.append(
                                    (len(path) - 1, substituent.GetSymbol())
                                )
        attachment_text = "none"
        if attachments:
            distance, element = min(attachments)
            position = {1: "ortho", 2: "meta", 3: "para"}.get(
                distance, f"distance_{distance}"
            )
            attachment_text = f"{position}({element})"
        return f"aryl_linker_atoms={linker_atoms}; terminal_heterocycle_attachment={attachment_text}"

    def build_halogen_deletion_prompt(
        self,
        dataset: str,
        records: list[MoleculeRecord],
        query_idx: int,
        train_idx: list[int],
    ) -> tuple[list[dict], list[int]] | None:
        """Build the halogen deletion prompt and return its reference rows.

        This is used when the closest same-scaffold reference has more
        halogens than the test.  The reference's measured class is on the
        more-halogenated side; the LLM must decide whether the deletion flips
        or retains that class.  Scaffold rigidity and shape are explicit
        evidence.
        """
        query = records[query_idx]
        if not query.scaffold:
            return None
        same_scaffold = [
            idx for idx in train_idx if records[idx].scaffold == query.scaffold
        ]
        if not same_scaffold:
            return None
        ordered = sorted(
            same_scaffold,
            key=lambda idx: (-tanimoto(query.morgan, records[idx].morgan), idx),
        )
        query_halogens = self._halogen_count(query)
        closest = ordered[0]
        if query_halogens >= self._halogen_count(records[closest]):
            return None
        reference_lines = []
        for idx in ordered:
            ref = records[idx]
            reference_lines.append(
                f"  ref={idx}; label={ref.label}; pChEMBL={_fmt(ref.pchembl)}; similarity={_fmt(tanimoto(query.morgan, ref.morgan))}; halogens={self._halogen_count(ref)}; SMILES={Chem.MolToSmiles(ref.mol, isomericSmiles=True)}; rings[{self._ring_signature(ref)}]; fraction_csp3={_fmt(ref.descriptors.get('FractionCSP3'))}; aromatic_rings={_fmt(ref.descriptors.get('NumAromaticRings'))}; aliphatic_rings={_fmt(ref.descriptors.get('NumAliphaticRings'))}; functional_groups[{self._functional_signature(ref)}]; sidechains[{self._sidechain_signature(ref)}]"
            )
        context = self.context
        system = "You are a medicinal chemist assessing a local halogen SAR change. Use only measured same-scaffold references, the local substitution pattern, and scaffold shape. Do not use global class prevalence or drug-likeness as the decision rule. Return only JSON."
        user = f"""TASK
Dataset: {dataset}
Endpoint: {context["endpoint"]}
Chemical series: {context["series"]}

TEST COMPOUND
SMILES: {Chem.MolToSmiles(query.mol, isomericSmiles=True)}
Murcko scaffold: {query.scaffold}
halogens: {query_halogens}
element_counts: {self._element_signature(query)}
ring_composition: {self._ring_signature(query)}
functional_groups: {self._functional_signature(query)}
sidechain_fragments: {self._sidechain_signature(query)}
Properties:
{chr(10).join((f"  {name}={_fmt(query.descriptors[name])}" for name in query.descriptors))}

SAME-SCAFFOLD TRAINING REFERENCES
{(chr(10).join(reference_lines) if reference_lines else "  none")}

QUESTION
The test has fewer halogens than the closest same-scaffold reference
(ref={closest}; measured_class={records[closest].label}). The reference's class
is on the more-halogenated side. Do not simply copy that class. Decide whether
the halogen deletion is likely to flip the class or retain it.
Explicitly consider scaffold rigidity and shape: a simple flexible
biaryl/benzophenone often depends more on halogen substitution for potency,
so a one-halogen deletion is more likely to flip the closest reference's
class: deletion from an active flexible benzophenone tends toward inactive,
while deletion from an inactive flexible benzophenone tends toward active.
A same-halogen reference is useful counter-evidence, but its absence is not by
itself a reason to retain the halogenated reference class. In a flexible aryl
ketone series, use the unsubstituted-H analog as the direct product of the
deletion and override the flip only for a measured local-pocket or substitution
effect, not merely because no exact H analog was measured.
A rigid fused/spiro polycyclic scaffold is more likely to retain the closest
reference's class after one halogen deletion. Use the measured same-scaffold
analogs, local substitution pattern, and the shape descriptors; override this
prior only when the local pattern gives a clear reason. Return active,
inactive, or Uncertain.

OUTPUT (JSON only)
{{
  "choice": "active" or "inactive" or "Uncertain",
  "confidence": 0.0 to 1.0,
  "scaffold_type": "one sentence describing flexibility/rigidity",
  "halogen_change_interpretation": "one sentence",
  "decisive_reference": "reference index",
  "rationale": "2-4 concise sentences"
}}
"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return messages, ordered

    def build_stereo_gate_pair_prompt(
        self,
        dataset: str,
        records: list[MoleculeRecord],
        query_idx: int,
        train_idx: list[int],
    ) -> tuple[list[dict], list[int]] | None:
        """Build the stereo gate prompt and return its reference rows.

        The comparison is deliberately limited to one active exact
        constitution and one inactive terminal-class reference.  The LLM
        decides whether the terminal context makes the opposite
        stereochemistry analogous; it is not given a global R/S rule.
        """
        query = records[query_idx]
        query_key = Chem.MolToSmiles(query.mol, isomericSmiles=False)
        query_isomer = Chem.MolToSmiles(query.mol, isomericSmiles=True)
        exact_active = [
            idx
            for idx in train_idx
            if records[idx].label == "active"
            and Chem.MolToSmiles(records[idx].mol, isomericSmiles=False) == query_key
            and (
                Chem.MolToSmiles(records[idx].mol, isomericSmiles=True) != query_isomer
            )
        ]
        if not exact_active:
            return None
        active_ref = max(
            exact_active,
            key=lambda idx: (tanimoto(query.morgan, records[idx].morgan), -idx),
        )
        query_sidechains = self._sidechain_signature(query)
        inactive = [idx for idx in train_idx if records[idx].label == "inactive"]
        if not inactive:
            return None
        inactive_ref = max(
            inactive,
            key=lambda idx: (
                bool(query.scaffold) and records[idx].scaffold == query.scaffold,
                self._terminal_similarity(query, records[idx]),
                tanimoto(query.morgan, records[idx].morgan),
                -idx,
            ),
        )
        active = records[active_ref]
        inactive_record = records[inactive_ref]
        terminal_similarity = self._terminal_similarity(query, inactive_record)
        context = self.context
        system = "You are resolving a stereochemical activity gate. The test is the opposite configuration of an active reference, so its class must not be copied directly. Compare the test's terminal substituent class with the closest inactive reference. If that inactive reference is a close terminal-class analog, choose inactive; if its terminal class differs substantially, choose active. Do not apply a global R/S rule. Return only JSON."
        user = f"""TASK
Dataset: {dataset}
Endpoint: {context["endpoint"]}
Chemical series: {context["series"]}

TEST COMPOUND
SMILES: {Chem.MolToSmiles(query.mol, isomericSmiles=True)}
Murcko scaffold: {query.scaffold or "(acyclic)"}
stereochemistry: {self._stereo_signature(query)}
chiral_signature: {self._cip_signature(query)}
element_counts: {self._element_signature(query)}
functional_groups: {self._functional_signature(query)}
sidechain_fragments: {query_sidechains}
Properties:
{chr(10).join((f"  {name}={_fmt(query.descriptors[name])}" for name in query.descriptors))}

EXACT OPPOSITE ACTIVE REFERENCE
ref={active_ref}; active; pChEMBL={_fmt(active.pchembl)};
similarity={_fmt(tanimoto(query.morgan, active.morgan))};
stereo={self._stereo_signature(active)};
same_scaffold={bool(query.scaffold) and active.scaffold == query.scaffold};
sidechains[{self._sidechain_signature(active)}];
SMILES={Chem.MolToSmiles(active.mol, isomericSmiles=True)}

CLOSEST INACTIVE REFERENCE FOR TERMINAL-CLASS COMPARISON
ref={inactive_ref}; inactive; pChEMBL={_fmt(inactive_record.pchembl)};
similarity={_fmt(tanimoto(query.morgan, inactive_record.morgan))};
terminal_similarity={_fmt(terminal_similarity)};
same_scaffold={bool(query.scaffold) and inactive_record.scaffold == query.scaffold};
stereo={self._stereo_signature(inactive_record)};
sidechains[{self._sidechain_signature(inactive_record)}];
SMILES={Chem.MolToSmiles(inactive_record.mol, isomericSmiles=True)}

QUESTION
The test is the opposite enantiomer of the exact active reference. Decide
whether the closest inactive reference is a genuine terminal-class analog
that justifies transferring the inactive class, or whether its terminal
group differs enough that the exact active scaffold should remain active.
Compare the full terminal substituent. Treat benzyl/aryl-methyl analogs with
different aryl substituents (for example Me versus OMe) as the same terminal
class. Treat a different N-alkyl chain length or a different aryl ring type
(for example benzene versus thiophene) as a substantial difference.
Do not use a global R/S rule.

OUTPUT (JSON only)
{{
  "choice": "active" or "inactive" or "Uncertain",
  "confidence": 0.0 to 1.0,
  "terminal_class_match": "one sentence",
  "decisive_reference": "active or inactive reference index",
  "rationale": "2-4 concise sentences"
}}
"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return messages, [active_ref, inactive_ref]

    @staticmethod
    def _sidechain_neighbors(
        query: MoleculeRecord,
        records: list[MoleculeRecord],
        train_idx: list[int],
        limit: int = 8,
    ) -> str:
        """Rank training analogs by Murcko sidechain string similarity."""
        query_sidechains = ReferenceJudge._sidechain_signature(query)
        candidates = []
        for idx in train_idx:
            ref = records[idx]
            sidechains = ReferenceJudge._sidechain_signature(ref)
            ratio = SequenceMatcher(None, query_sidechains, sidechains).ratio()
            candidates.append(
                (ratio, tanimoto(query.morgan, ref.morgan), idx, sidechains)
            )
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        lines = [f"  query_sidechains={query_sidechains}"]
        for ratio, similarity, idx, sidechains in candidates[:limit]:
            ref = records[idx]
            lines.append(
                f"  ref={idx}; sidechain_similarity={_fmt(ratio)}; molecule_similarity={_fmt(similarity)}; label={ref.label}; pChEMBL={_fmt(ref.pchembl)}; sidechains={sidechains}"
            )
        return "\n".join(lines)

    def build_prompt(
        self,
        dataset: str,
        records: list[MoleculeRecord],
        query_idx: int,
        active_ref: int,
        inactive_ref: int,
        pair_score: float,
        train_idx: list[int] | None = None,
        active_side: str = "A",
    ) -> list[dict]:
        query = records[query_idx]
        a = records[active_ref]
        i = records[inactive_ref]
        if active_side == "A":
            side_a, side_b = (a, i)
        else:
            side_a, side_b = (i, a)
        context = self.context
        query_auxiliary_line = ""

        def ref_block(letter: str, ref: MoleculeRecord) -> str:
            props = "\n".join(
                (
                    f"  {name}={_fmt(ref.descriptors[name])} (delta_vs_test={_fmt(ref.descriptors[name] - query.descriptors[name])})"
                    for name in ref.descriptors
                )
            )
            same_constitution = Chem.MolToSmiles(
                ref.mol, isomericSmiles=False
            ) == Chem.MolToSmiles(query.mol, isomericSmiles=False)
            same_isomer = Chem.MolToSmiles(
                ref.mol, isomericSmiles=True
            ) == Chem.MolToSmiles(query.mol, isomericSmiles=True)
            heavy_atom_delta = abs(
                ref.descriptors["HeavyAtomCount"] - query.descriptors["HeavyAtomCount"]
            )
            auxiliary_line = ""
            return f"""REFERENCE {letter}
supplied_label: {ref.label}
measured_pchembl: {_fmt(ref.pchembl)}
{auxiliary_line}SMILES: {Chem.MolToSmiles(ref.mol, isomericSmiles=True)}
Murcko_scaffold: {ref.scaffold or "(acyclic)"}
test_similarity={_fmt(tanimoto(query.morgan, ref.morgan))}
test_relation: constitution_match={same_constitution}, isomeric_match={same_isomer}
heavy_atom_delta={_fmt(heavy_atom_delta)}
element_counts: {self._element_signature(ref)}
ring_composition: {self._ring_signature(ref)}
functional_groups: {self._functional_signature(ref)}
sidechain_fragments: {self._sidechain_signature(ref)}
stereochemistry: {self._stereo_signature(ref)}
properties:
{props}"""

        prior_block = ""
        if train_idx:
            counts = Counter((records[idx].label for idx in train_idx))
            prior_block = f"""
TRAINING PRIOR
active={counts["active"]}, inactive={counts["inactive"]}. This is a weak calibration prior, not a substitute for local structural evidence. Do not infer activity from general drug-likeness, a known target class, or a basic amine alone; those claims must be supported by this pair.
"""
        structural_block = ""
        structural_block = f"""
STRUCTURAL DIFFERENCES (descriptive, not a decision rule)
element_counts(test): {self._element_signature(query)}
element_counts(A): {self._element_signature(side_a)}
element_counts(B): {self._element_signature(side_b)}
ring_composition(test): {self._ring_signature(query)}
ring_composition(A): {self._ring_signature(side_a)}
ring_composition(B): {self._ring_signature(side_b)}
stereochemistry(test): {self._stereo_signature(query)}
MACCS patterns differing between A and B:
{self._maccs_differences(query, side_a, side_b)}
"""
        system = "You are a medicinal chemist comparing a test compound with two measured reference compounds, one active and one inactive. Your job is not to pick the more attractive class; it is to judge which reference's measured potency the test is more likely to resemble. Use the structural difference, stereochemistry, and measured pChEMBL values. Treat the supplied labels as measured evidence, not as a recommendation to favor the active class. Return only the requested JSON object."
        user = f"""TASK
Dataset: {dataset}
Endpoint: {context["endpoint"]}
Chemical series: {context["series"]}
{prior_block}

TEST COMPOUND
SMILES: {Chem.MolToSmiles(query.mol, isomericSmiles=True)}
Murcko scaffold: {query.scaffold or "(acyclic)"}
Continuous target: withheld; do not assume it.
element_counts: {self._element_signature(query)}
functional_groups: {self._functional_signature(query)}
sidechain_fragments: {self._sidechain_signature(query)}
stereochemistry: {self._stereo_signature(query)}
{query_auxiliary_line}
Properties:
{chr(10).join((f"  {name}={_fmt(query.descriptors[name])}" for name in query.descriptors))}

PAIR (selection score {_fmt(pair_score)}; structural pair similarity is not a label)
{ref_block("A", side_a)}

{ref_block("B", side_b)}
{structural_block}

QUESTION
Which reference is the test more likely to resemble in measured potency?
Consider the smallest structural change, the shared scaffold/terminal group,
the element/ring differences, stereochemistry, and the measured pChEMBL
values. Do not choose the active reference merely because it is active or
because the compound class is generally associated with the target. If the
pair does not isolate a relevant change for this test, choose "Uncertain"
rather than forcing a vote; otherwise choose the closer reference even when
the structural change is imperfect. A reference with isomeric_match=False is
not an exact stereochemical match, even when constitution_match=True.

OUTPUT (JSON only)
{{
  "choice": "A" or "B" or "Uncertain",
  "confidence": 0.0 to 1.0,
  "pchembl_estimate": estimated test pChEMBL or null,
  "smallest_structural_difference": "one sentence",
  "why_this_pair_is_relevant": "one sentence",
  "counter_evidence": "strongest reason the other reference might be closer",
  "rationale": "2-4 sentences"
}}
"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def build_panel_prompt(
        self,
        dataset: str,
        records: list[MoleculeRecord],
        query_idx: int,
        train_idx: list[int],
        active_panel: list[int],
        inactive_panel: list[int],
    ) -> list[dict]:
        query = records[query_idx]
        context = self.context
        counts = Counter((records[idx].label for idx in train_idx))
        query_functional_signature = self._functional_signature(query)
        query_functional_counts = self._functional_group_counts(query)
        has_inactive_exocyclic_alkene = bool(
            query_functional_counts.get("aryl_methylene_carbocycle")
        ) and any(
            (
                records[idx].label == "inactive"
                and "aryl_exocyclic_alkene"
                in self._functional_group_counts(records[idx])
                for idx in train_idx
            )
        )
        methyl_groups = ("amine_methyl", "tertiary_amine")
        has_same_count_inactive_methyl = bool(
            any((query_functional_counts.get(name) for name in methyl_groups))
        ) and any(
            (
                records[idx].label == "inactive"
                and any(
                    (
                        query_functional_counts.get(name, 0)
                        == self._functional_group_counts(records[idx]).get(name, 0)
                        for name in methyl_groups
                        if query_functional_counts.get(name, 0)
                    )
                )
                for idx in train_idx
            )
        )
        query_halogens = self._halogen_count(query)
        same_halogen_active = any(
            (
                self._halogen_count(records[idx]) == query_halogens
                for idx in active_panel
            )
        )
        same_halogen_inactive = any(
            (
                self._halogen_count(records[idx]) == query_halogens
                for idx in inactive_panel
            )
        )
        use_functional_groups = True
        use_bond_signature = use_functional_groups
        key_descriptors = (
            "MolWt",
            "MolLogP",
            "TPSA",
            "NumHDonors",
            "NumHAcceptors",
            "NumRotatableBonds",
            "RingCount",
            "NumAromaticRings",
            "NumAliphaticRings",
            "FractionCSP3",
            "HeavyAtomCount",
            "NumHeteroatoms",
            "qed",
        )

        def panel_line(ref: MoleculeRecord) -> str:
            props = ", ".join(
                (f"{name}={_fmt(ref.descriptors[name])}" for name in key_descriptors)
            )
            auxiliary_text = ""
            topology_text = ""
            functional_text = (
                f"functional_groups[{self._functional_signature(ref)}]; "
                if use_functional_groups
                else ""
            )
            bond_text = (
                f"bonds[{self._bond_signature(ref)}]; " if use_bond_signature else ""
            )
            sidechain_text = (
                f"sidechains[{self._sidechain_signature(ref)}]; "
                if use_sidechain_evidence
                else ""
            )
            linker_text = ""
            return f"- {ref.label}: pChEMBL={_fmt(ref.pchembl)}; {auxiliary_text}similarity={_fmt(tanimoto(query.morgan, ref.morgan))}; constitution_match={Chem.MolToSmiles(ref.mol, isomericSmiles=False) == Chem.MolToSmiles(query.mol, isomericSmiles=False)}; isomeric_match={Chem.MolToSmiles(ref.mol, isomericSmiles=True) == Chem.MolToSmiles(query.mol, isomericSmiles=True)}; heavy_atom_delta={_fmt(abs(ref.descriptors['HeavyAtomCount'] - query.descriptors['HeavyAtomCount']))}; SMILES={Chem.MolToSmiles(ref.mol, isomericSmiles=True)}; elements[{self._element_signature(ref)}]; rings[{self._ring_signature(ref)}]; {functional_text}{sidechain_text}{linker_text}{bond_text}stereo[{self._stereo_signature(ref)}]; {topology_text}{props}"

        exact_block = ""
        local_block = ""
        transfer_block = ""
        question_block = """First name the most consistent structural or property contrast that
separates the active references from the inactive references. Then decide
whether the test lies on the active or inactive side of that contrast. If the
panel does not contain a meaningful contrast for this test, return
"Uncertain". If an exact TEST_CONSTITUTION_MATCH is present, treat it as
strong local evidence and explain why it does or does not override the broader
class contrast.
When constitution_match=True but isomeric_match=False, the reference is an
enantiomer/relative stereoisomer, not an exact match.
RELATED_CONSTITUTION lines are weaker: use them only when the test's terminal
substituent class is analogous, and do not transfer a stereochemical rule
across a major terminal-group change such as thiophene versus methoxybenzyl.
"""
        question_block += """
When SAME-SCAFFOLD MEASURED REFERENCES are present, compare the test with the
closest rows first. A different-scaffold reference should not override a
clear same-scaffold measured analog merely because of a broad prevalence
trend; explain any disagreement explicitly.
"""
        question_block += """
When SAME-FORMULA CONSTITUTIONAL ISOMERS are present, compare how the measured
label changes with connectivity or attachment position at constant atom
composition. A regioisomer pair is stronger local evidence for a positional
effect than a broad series trend. Do not copy a reference label when the test
has a different placement unless the local comparison supports that transfer.
"""
        if has_same_count_inactive_methyl:
            question_block += """
When amine_methyl counts differ, compare cumulative N-methylation explicitly.
A second N-methyl is not automatically a minor tolerated change: require a
measured active reference with the same N-methyl count before assuming that
the additional methylation retains activity.
When tertiary_amine counts differ, compare measured references with the same
count and hydrogen-bond-donor status. If the only same-count measured analog
is inactive, do not assume that an additional N-alkylation is tolerated just
because the added group is small.
If the test's tertiary_amine count is higher than every active reference and
equals a measured inactive analog, while no active reference has that same
count, use inactive as the local default. Override it only with a direct active
analog that shares the count, not merely because the added alkyl group is
small.
"""
        if query_functional_counts.get("aryl_exocyclic_alkene"):
            question_block += """
The test has an aryl exocyclic alkene. Treat that rigid unsaturated attachment
as a distinct local linker type, not as a minor bond-order change from a
single-bond alkyl analog. If no measured active reference shares an aryl
exocyclic alkene, do not infer active status by interpolating from a single-
bond alkyl neighbor. Use inactive as the local default unless a measured
active reference or a direct stereochemical comparison supports activity for
this exact linker type.
"""
        if (
            query_functional_counts.get("aryl_methylene_carbocycle")
            and has_inactive_exocyclic_alkene
        ):
            question_block += """
The test connects a carbocycle to the aryl ring through a saturated methylene.
Do not merge this linker with an exocyclic alkene or arylidene analog merely
because they differ by two hydrogens. Saturation can change shape and activity,
and a measured inactive dehydrated alkene does not establish that the saturated
methylene is inactive. Compare the saturated O-linked and direct-carbon
carbocycle analogs separately, and state why the linker's hybridization and
geometry place the test on one side of the local class boundary.
"""
        if query_halogens:
            question_block += f"""
The test carries {query_halogens} halogen atom(s). When HALOGEN PATTERN
REFERENCES are present, compare total halogen count and substitution pattern
across the full active and inactive panels, not only the same-scaffold rows. A
same-count active reference is positive evidence for that halogen pattern even
when its linker differs; a same-count inactive is counter-evidence. Here
same_count_active={str(same_halogen_active).lower()} and
same_count_inactive={str(same_halogen_inactive).lower()}. Do not infer active
status from a one-halogen count difference when no active reference shares the
test count. Conversely, do not discard a measured active halogen pattern only
because its linker differs. State which exact-count references and which
one-halogen-delta same-scaffold reference dominate, and why.
"""
        if query_functional_counts.get("aryl_hydroxyl"):
            question_block += """
When the test has a terminal hydroxyl, do not merge it with a methoxy analog
as one generic O-substituent class. A phenolic hydroxyl is a hydrogen-bond
donor and can retain activity when the corresponding methoxy is inactive.
Compare the nearest H analog, the measured hydroxyl/donor context, and the
methoxy analog separately; state explicitly whether donor status explains the
local class change.
"""
        if query_functional_counts.get("aliphatic_hydroxyl"):
            question_block += """
The test has an aliphatic terminal hydroxyl. Compare it with measured
hydroxymethyl, aminomethyl, and methoxymethyl analogs sharing the same scaffold
before treating the hydroxyl as an active donor. Do not transfer the active
phenolic-hydroxyl behavior to an aliphatic alcohol, and do not infer activity
merely from the O atom. If several close small polar terminal analogs are
inactive and no active reference shares the aliphatic hydroxyl, state why the
test differs before placing it active.
"""
        use_sidechain_evidence = (
            self._sidechain_fragment_count(query) >= self.sidechain_min_fragments
        )
        if use_sidechain_evidence:
            question_block += f"""
When a SIDECHAIN NEAREST NEIGHBORS table is present, compare the test's exact
Murcko sidechain fragments with those references. A close sidechain with a
measured pChEMBL below {self.context["threshold"]} is stronger local evidence than a broader
size/lipophilicity trend. Do not treat an extended substituent as active
merely because it is larger; use the closest measured sidechain values.
"""
        if self._has_opposite_exact_reference(query, records, train_idx):
            question_block += f"""
STEREOCHEMICAL CAUTION: the closest exact-constitution reference is an
opposite enantiomer, not an exact match. Do not transfer its activity
directly. Estimate the potency gap from analogous matched stereoisomer pairs
and apply that gap; if the active enantiomer is only slightly above the {self.context["threshold"]}
threshold, the opposite enantiomer may fall below it. A different terminal
substituent class can change the gap, so do not apply one R/S rule globally.
"""
        regression_block = ""
        key_pairs_block = ""
        if use_sidechain_evidence:
            sidechain_block = f"""
SIDECHAIN NEAREST NEIGHBORS (training references only)
These references are ranked by similarity of their Murcko-core substituent
fragments, not by overall molecule similarity. Use the measured pChEMBL values
of the closest sidechains to judge whether the test's exact substituent is
closer to the active or inactive local series.
{self._sidechain_neighbors(query, records, train_idx)}
"""
        else:
            sidechain_block = ""
        local_analog_block = ""
        stereo_transfer_block = ""
        one_fragment_cliff_block = ""
        same_scaffold_block = f"""
SAME-SCAFFOLD MEASURED REFERENCES (training references only)
These references share the test's exact Murcko scaffold. Use them before a
broad class-prevalence contrast. A different-scaffold reference is weaker
local evidence unless the structural change it isolates is explicit.
{self._same_scaffold_neighbors(query, records, train_idx)}
"""
        retrieval_consistency_block = ""
        constitutional_isomer_block = f"""
SAME-FORMULA CONSTITUTIONAL ISOMERS (training references only)
These measured references have the same molecular formula as the test but a
different connectivity or attachment position. Use them to test whether the
local positional change is activity-critical.
{self._constitutional_isomer_evidence(query, records, train_idx)}
"""
        functional_group_neighbor_block = ""
        halogen_pattern_block = (
            f"""
HALOGEN PATTERN REFERENCES (training references only)
Rows labeled same_count_active/inactive have the test's exact total halogen
count. The last row is the closest same-scaffold reference with a one-halogen
delta. Compare substitution position and linker context before transferring a
class.
{self._halogen_pattern_evidence(query, records, train_idx)}
"""
            if self._halogen_count(query)
            else ""
        )
        query_auxiliary_line = ""
        query_topology_line = ""
        query_sidechain_line = (
            f"sidechain_fragments: {self._sidechain_signature(query)}\n"
            if use_sidechain_evidence
            else ""
        )
        query_functional_line = (
            f"functional_groups: {query_functional_signature}\n"
            if use_functional_groups
            else ""
        )
        query_bond_line = (
            f"bonds: {self._bond_signature(query)}\n" if use_bond_signature else ""
        )
        auxiliary_block = ""
        topology_question = ""
        system = "You are a medicinal chemist performing a class-contrast analysis. The active and inactive panels share a chemical series. Identify the structural/property direction that separates the two measured classes, then place the test compound on that direction. Do not choose active because there are more active references or because the test is drug-like. When local matched pairs are available, give them priority over a broad prevalence contrast. Return only the requested JSON object. Auxiliary assays are non-target measurements; use them as supporting evidence, never as the target label. A local regression estimate is a numerical consistency check, not a final classification rule. A single local terminal-substituent change that flips a matched pair is stronger evidence than a broad prevalence trend."
        counts = Counter((records[idx].label for idx in train_idx))
        prior_block = f"""
TRAINING PRIOR
active={counts["active"]}, inactive={counts["inactive"]}. This is only a
calibration prior; the class contrast below is the primary evidence.
"""
        system = system.replace(
            " Auxiliary assays are non-target measurements; use them as supporting evidence, never as the target label.",
            "",
        )
        user = f"""TASK
Dataset: {dataset}
Endpoint: {context["endpoint"]}
Chemical series: {context["series"]}

TEST COMPOUND
SMILES: {Chem.MolToSmiles(query.mol, isomericSmiles=True)}
Murcko scaffold: {query.scaffold or "(acyclic)"}
element_counts: {self._element_signature(query)}
ring_composition: {self._ring_signature(query)}
{query_functional_line}
{query_sidechain_line}
{query_bond_line}
stereochemistry: {self._stereo_signature(query)}
{query_topology_line}
chiral_signature: {self._cip_signature(query)}
{query_auxiliary_line}
Properties:
{chr(10).join((f"  {name}={_fmt(query.descriptors[name])}" for name in query.descriptors))}

{prior_block}
{auxiliary_block}

{exact_block}

NEAREST ACTIVE REFERENCES
{chr(10).join((panel_line(records[i]) for i in active_panel)) or "- none"}

NEAREST INACTIVE REFERENCES
{chr(10).join((panel_line(records[i]) for i in inactive_panel)) or "- none"}

{same_scaffold_block}

{constitutional_isomer_block}

{retrieval_consistency_block}

{functional_group_neighbor_block}
{halogen_pattern_block}

CLASS-CONSISTENT STRUCTURAL FEATURES (training references only)
These are descriptive prevalence differences, not a decision rule. The test
value is shown as 0/1.
{self._class_contrast_features(query, records, train_idx)}

{regression_block}

{key_pairs_block}

{local_block}

{sidechain_block}

{local_analog_block}

{stereo_transfer_block}

{one_fragment_cliff_block}

STEREOCHEMICAL MATCHED PAIRS (training references only)
Compounds with identical non-isomeric constitution but opposite measured
classes. Use these as local stereochemical evidence when the test has the same
constitution. Only a line marked TEST_CONSTITUTION_MATCH is directly
applicable. RELATED_CONSTITUTION lines are context only and must not change
the class; when there is no TEST_CONSTITUTION_MATCH, use the non-stereochemical
scaffold/series contrast instead of inferring a class from R/S alone.
{self._stereochemical_matched_pairs(query, records, train_idx)}

{transfer_block}

QUESTION
{question_block}
{topology_question}
When measured pChEMBL values are available, estimate the test pChEMBL by local
interpolation and use the {self.context["threshold"]} threshold as a calibration check; do not invent
a value when the panel has no measurements. Keep the rationale under 120
words.

OUTPUT (JSON only)
{{
  "choice": "active" or "inactive" or "Uncertain",
  "confidence": 0.0 to 1.0,
  "pchembl_estimate": estimated test pChEMBL or null,
  "class_contrast": "one sentence naming the active-vs-inactive direction",
  "test_position": "one sentence locating the test on that direction",
  "counter_evidence": "strongest reason the other class might fit",
  "rationale": "2-4 concise sentences"
}}
"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def judge_panel(
        self, messages: list[dict], *, tag: str, force_decision: bool = False
    ) -> FinalChoice:
        raw = ""
        last_error = None
        current = messages
        for attempt in range(2):
            try:
                kwargs = dict(
                    messages=current,
                    temperature=self.temperature,
                    tag=f"{tag}_panel_a{attempt}",
                    max_tokens=self.max_tokens,
                )
                if self.use_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                raw = self.llm.chat(**kwargs) or ""
                call = dict(getattr(self.llm, "last_call", None) or {})
                if call.get("truncated"):
                    last_error = "response was truncated at max_tokens; return only the complete JSON object and omit repeated analysis"
                    if attempt == 0:
                        current = messages + [
                            {
                                "role": "user",
                                "content": "Return one complete JSON object under 500 tokens. Do not repeat the prompt, do not enumerate all references, and keep the rationale under 80 words.",
                            }
                        ]
                    continue
                parsed = safe_parse_json(raw)
                if parsed is None:
                    last_error = "response was not valid JSON"
                elif parsed.get("choice") not in (*CLASSES, "Uncertain"):
                    last_error = f"invalid choice {parsed.get('choice')!r}"
                elif force_decision and parsed.get("choice") == "Uncertain":
                    last_error = "the answer was Uncertain, but this final panel decision must be active or inactive"
                else:
                    try:
                        confidence = float(parsed.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        confidence = 0.0
                    confidence = max(0.0, min(1.0, confidence))
                    parsed["confidence"] = confidence
                    return FinalChoice(
                        label=str(parsed["choice"]),
                        confidence=confidence,
                        parsed=parsed,
                        raw_response=raw,
                        call=call,
                        parsed_ok=True,
                    )
            except ReplayError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                current = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"The previous response was unusable: {last_error}. Return the complete JSON object with choice active "
                        + (
                            "or inactive; do not return Uncertain."
                            if force_decision
                            else ", inactive, or Uncertain."
                        ),
                    },
                ]
        return FinalChoice(
            label="Inconclusive",
            confidence=0.0,
            parsed={},
            raw_response=raw,
            call=dict(getattr(self.llm, "last_call", None) or {}),
            parsed_ok=False,
            error=last_error,
        )

    def judge(
        self,
        messages: list[dict],
        *,
        pair_id: int,
        active_ref: int,
        inactive_ref: int,
        tag: str,
        active_side: str = "A",
    ) -> PairChoice:
        raw = ""
        last_error = None
        current = messages
        for attempt in range(2):
            try:
                kwargs = dict(
                    messages=current,
                    temperature=self.temperature,
                    tag=f"{tag}_a{attempt}",
                    max_tokens=self.max_tokens,
                )
                if self.use_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                raw = self.llm.chat(**kwargs) or ""
                call = dict(getattr(self.llm, "last_call", None) or {})
                if call.get("truncated"):
                    last_error = "response was truncated at max_tokens"
                    raw = ""
                    if attempt == 0:
                        current = messages + [
                            {"role": "user", "content": "Return one concise, complete JSON object."}
                        ]
                    continue
                parsed = safe_parse_json(raw)
                if parsed is None:
                    last_error = "response was not valid JSON"
                elif parsed.get("choice") not in ("A", "B", "Uncertain"):
                    last_error = f"invalid choice {parsed.get('choice')!r}"
                else:
                    try:
                        confidence = float(parsed.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        confidence = 0.0
                    confidence = max(0.0, min(1.0, confidence))
                    parsed["confidence"] = confidence
                    return PairChoice(
                        pair_id=pair_id,
                        active_ref=active_ref,
                        inactive_ref=inactive_ref,
                        choice=str(parsed["choice"]),
                        confidence=confidence,
                        parsed=parsed,
                        raw_response=raw,
                        call=call,
                        parsed_ok=True,
                        active_side=active_side,
                    )
            except ReplayError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                current = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"The previous response was unusable: {last_error}. Return the complete JSON object with choice A, B, or Uncertain.",
                    },
                ]
        return PairChoice(
            pair_id=pair_id,
            active_ref=active_ref,
            inactive_ref=inactive_ref,
            choice="Uncertain",
            confidence=0.0,
            parsed={},
            raw_response=raw,
            call=dict(getattr(self.llm, "last_call", None) or {}),
            parsed_ok=False,
            error=last_error,
            active_side=active_side,
        )

    @staticmethod
    def aggregate(choices: list[PairChoice]) -> tuple[str, float, dict]:
        weights = {cls: 0.0 for cls in CLASSES}
        counts = {cls: 0 for cls in CLASSES}
        for choice in choices:
            if not choice.parsed_ok:
                continue
            if choice.choice == "A":
                label = "active" if choice.active_side == "A" else "inactive"
            elif choice.choice == "B":
                label = "inactive" if choice.active_side == "A" else "active"
            else:
                continue
            weights[label] += max(choice.confidence, 1e-06)
            counts[label] += 1
        if not any(weights.values()) or weights["active"] == weights["inactive"]:
            return ("Inconclusive", 0.0, {"weights": weights, "counts": counts})
        total = sum(weights.values())
        label = max(weights, key=weights.get)
        return (label, weights[label] / total, {"weights": weights, "counts": counts})
