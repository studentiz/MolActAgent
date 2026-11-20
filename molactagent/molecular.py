"""Binary activity classification from labelled reference SMILES."""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

try:
    from ._molecules import CLASSES, make_record, tanimoto
    from ._molecular_evidence import FinalChoice, ReferenceJudge
except ModuleNotFoundError as exc:
    if exc.name == "rdkit":
        raise ImportError("Install MolActAgent with the [chem] extra to classify SMILES.") from exc
    raise

from ._molecular_recording import CallRecorder, ReplayClient, digest, runtime_versions
from .config import MOLECULAR_MAX_TOKENS


@dataclass(frozen=True)
class MolecularResult:
    """Decision, reference panel, individual judgments and model-call metadata."""

    identifier: str
    smiles: str
    label: str
    confidence: float
    selected_view: str
    references: list[dict]
    judgments: list[dict]
    calls: list[dict]
    reason: str | None = None
    decision: dict = field(default_factory=dict)
    evidence_references: list[dict] = field(default_factory=list)


def _panel_decision(choices: list[FinalChoice]):
    counts = Counter(choice.label for choice in choices if choice.parsed_ok)
    opposing = {label: counts[label] for label in CLASSES}
    usable = sum(opposing.values())
    required = len(choices) // 2 + 1
    weights = {
        label: sum(choice.confidence for choice in choices if choice.parsed_ok and choice.label == label)
        for label in CLASSES
    }
    decision = {
        "planned": len(choices),
        "counts": {**opposing, "uncertain": counts["Uncertain"],
                   "failed": len(choices) - usable - counts["Uncertain"]},
        "decisive": usable,
        "required": required,
        "weight_totals": weights,
    }
    if usable < required:
        return "Inconclusive", 0.0, {**decision, "method": "insufficient_votes"}
    if opposing["active"] == opposing["inactive"]:
        if math.isclose(weights["active"], weights["inactive"], abs_tol=1e-9):
            return "Inconclusive", 0.0, {**decision, "method": "tie"}
        label = max(CLASSES, key=weights.get)
        return label, weights[label] / sum(weights.values()), {**decision, "method": "score_tiebreak"}
    label = max(CLASSES, key=opposing.get)
    return label, weights[label] / opposing[label], {**decision, "method": "majority"}


def _reference_rows(records, query_index, indices, reference_ids):
    query = records[query_index]
    return [
        {
            "reference": reference_ids[index],
            "position": index,
            "smiles": records[index].canonical,
            "label": records[index].label,
            "similarity": tanimoto(query.morgan, records[index].morgan),
            "pchembl": records[index].pchembl,
        }
        for index in indices
    ]


class MolecularClassifier:
    """Predict active or inactive by contrasting groups of measured references.

    The default compares a class-balanced panel in five independent model
    calls. The focused profile also enables local halogen and stereo views.
    """

    def __init__(
        self,
        llm_client,
        *,
        endpoint: str = "Binary activity classification.",
        series: str = "",
        task_name: str = "Molecular activity",
        profile: Literal["panel", "focused"] = "panel",
        neighbors: int = 8,
        repeats: int = 5,
        temperature: float = 0.0,
        max_tokens: int = MOLECULAR_MAX_TOKENS,
        response_format: bool = True,
        pchembl_threshold: float | None = None,
        labels_from_pchembl: bool = False,
        workers: int = 4,
    ):
        if profile not in ("panel", "focused"):
            raise ValueError("Profile must be panel or focused.")
        for name, value in (
            ("neighbors", neighbors), ("repeats", repeats), ("max_tokens", max_tokens),
            ("workers", workers)
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if pchembl_threshold is not None and not math.isfinite(pchembl_threshold):
            raise ValueError("pChEMBL threshold must be finite.")
        if labels_from_pchembl and pchembl_threshold is None:
            raise ValueError("Supply a pChEMBL cutoff when the labels are defined by that cutoff.")
        self.client = llm_client
        self.task_name = task_name
        self.profile = profile
        self.neighbors = neighbors
        self.repeats = repeats
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.workers = workers
        self._context = {
            "endpoint": endpoint,
            "series": series,
            "threshold": pchembl_threshold,
            "labels_from_pchembl": labels_from_pchembl,
        }
        self._records = None
        self._reference_ids = []
        self.recording = None

    @property
    def settings(self):
        return {
            "endpoint": self._context["endpoint"],
            "series": self._context["series"],
            "pchembl_threshold": self._context["threshold"],
            "labels_from_pchembl": self._context["labels_from_pchembl"],
            "task_name": self.task_name,
            "profile": self.profile,
            "neighbors": self.neighbors,
            "repeats": self.repeats,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "workers": self.workers,
        }

    @staticmethod
    def _identifiers(frame, id_column, prefix):
        if id_column is None:
            return [f"{prefix}{position + 1}" for position in range(len(frame))]
        if id_column not in frame.columns:
            raise ValueError(f"Missing column {id_column!r}.")
        raw = frame[id_column]
        identifiers = raw.astype(str).tolist()
        if raw.isna().any() or any(not value.strip() for value in identifiers) or len(set(identifiers)) != len(raw):
            raise ValueError(f"Column {id_column!r} must contain unique, non-empty IDs.")
        return identifiers

    def fit(
        self,
        reference: pd.DataFrame,
        *,
        smiles_column: str = "smiles",
        label_column: str = "label",
        id_column: str | None = None,
        pchembl_column: str | None = None,
    ) -> MolecularClassifier:
        cutoff_labels = self._context["labels_from_pchembl"]
        cutoff = self._context["threshold"]
        if cutoff_labels and not pchembl_column:
            raise ValueError("Supply a pChEMBL column when the labels are defined by its cutoff.")
        required = [smiles_column, label_column] + ([pchembl_column] if pchembl_column else [])
        for name in required:
            if name not in reference.columns:
                raise ValueError(f"Missing reference column {name!r}.")
        labels = reference[label_column].astype(str).str.strip().str.lower().tolist()
        ids = self._identifiers(reference, id_column, "R")
        for identifier, label in zip(ids, labels):
            if label not in CLASSES:
                raise ValueError(f"Reference {identifier!r} needs an active or inactive label.")
        if set(labels) != set(CLASSES):
            raise ValueError("Reference labels must include both active and inactive.")
        values = reference[pchembl_column] if pchembl_column else [None] * len(ids)
        records = []
        by_structure = {}
        for row, (smiles, identifier, label, measured) in enumerate(
            zip(reference[smiles_column], ids, labels, values)
        ):
            potency = None
            if not pd.isna(measured) and not (isinstance(measured, str) and not measured.strip()):
                try:
                    potency = float(measured)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid pChEMBL value at {identifier!r}.") from exc
                if not math.isfinite(potency):
                    raise ValueError(f"Invalid pChEMBL value at {identifier!r}.")
            if cutoff_labels:
                if potency is None:
                    raise ValueError(f"Missing pChEMBL value at {identifier!r} for the declared cutoff.")
                expected = "active" if potency >= cutoff else "inactive"
                if label != expected:
                    raise ValueError(f"Label at {identifier!r} conflicts with active at pChEMBL ≥ {cutoff:g}.")
            record = make_record(smiles, row, identifier, label, potency)
            previous_id, previous_label = by_structure.setdefault(record.canonical, (identifier, label))
            if previous_label != label:
                raise ValueError(
                    f"References {previous_id!r} and {identifier!r} have conflicting labels for {record.canonical}."
                )
            records.append(record)
        self._records = records
        self._reference_ids = ids
        self._reference_digest = digest([
            (identifier, row.canonical, row.label, row.pchembl) for identifier, row in zip(ids, records)
        ])
        return self

    def predict(
        self,
        test: pd.DataFrame,
        *,
        smiles_column: str = "smiles",
        id_column: str | None = None,
        record: bool = False,
    ) -> list[MolecularResult]:
        if self._records is None:
            raise ValueError("Call fit with labelled reference molecules first.")
        if smiles_column not in test.columns or test.empty:
            raise ValueError(f"The test table needs at least one row with column {smiles_column!r}.")
        ids = self._identifiers(test, id_column, "Q")
        references = self._records
        queries = [
            make_record(smiles, len(references) + row, identifier)
            for row, (smiles, identifier) in enumerate(zip(test[smiles_column], ids))
        ]
        self.recording = None
        inputs = {
            "references": self._reference_digest,
            "queries": digest([(identifier, row.canonical) for identifier, row in zip(ids, queries)]),
        }
        if isinstance(self.client, ReplayClient):
            self.client.begin(self.settings, inputs)
        recorder = CallRecorder(self.client, keep_responses=record)
        judge = self._make_judge(recorder)
        results = []
        records = references + queries
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for ordinal, (query, identifier) in enumerate(zip(queries, ids)):
                indices = [ref.row for ref in references if ref.canonical != query.canonical]
                result = self._predict_fold(
                    records, query.row, indices, self._reference_ids, identifier,
                    f"query:{ordinal}", judge, pool,
                )
                results.append(result)
        if isinstance(self.client, ReplayClient):
            self.client.verify_complete()
        if record:
            self.recording = {
                "schema": 2,
                "model": getattr(self.client, "model_name", "unknown"),
                "versions": runtime_versions(),
                "settings": self.settings,
                "inputs": inputs,
                "calls": recorder.calls,
            }
        return results

    def _make_judge(self, recorder):
        return ReferenceJudge(
            recorder, self._context, self.temperature, self.max_tokens, self.response_format,
        )

    def _predict_fold(
        self, records, query_index, indices, reference_ids, identifier, tag, judge, pool,
    ):
        query = records[query_index]
        if {records[index].label for index in indices} != set(CLASSES):
            return MolecularResult(
                identifier, query.canonical, "Inconclusive", 0.0, "none", [], [], [],
                "Both reference classes are required after excluding identical query structures.",
                {"method": "missing_reference_class"},
            )

        unique_references = len({records[index].canonical for index in indices})
        size = 4 if self.neighbors == 8 and unique_references <= 6 else self.neighbors
        active, inactive = judge.select_panel(records, query_index, indices, n_per_class=size)
        messages, reference_positions = judge.build_panel_prompt(
            self.task_name, records, query_index, indices, active, inactive,
        )
        futures = [
            pool.submit(
                judge.judge_panel, messages, tag=f"{tag}:panel:r{repeat}",
                reference_positions=reference_positions,
            )
            for repeat in range(self.repeats)
        ]
        panel_choices = [future.result() for future in futures]
        label, confidence, decision = _panel_decision(panel_choices)
        selected = "panel"
        judgments = [
            self._judgment("panel", repeat, vote) for repeat, vote in enumerate(panel_choices)
        ]
        decision["focused"] = []
        if self.profile == "focused" and decision["decisive"] >= decision["required"]:
            focused = self._focused_views(records, query_index, indices, reference_ids, tag, judge, pool)
            for view, choice, evidence in focused:
                judgments.extend(evidence)
                eligible = choice is not None and choice.parsed_ok and choice.label in CLASSES
                applied = {"view": view, "applied": False}
                if not eligible:
                    applied["reason"] = "no_decisive_focused_judgment"
                elif view == "halogen_deletion" and choice.confidence < (0.6 if choice.label == label else 0.5):
                    applied["reason"] = "below_score_threshold"
                else:
                    label, confidence, selected = choice.label, choice.confidence, view
                    applied["applied"] = True
                decision["focused"].append(applied)

        panel = _reference_rows(records, query_index, (*active, *inactive), reference_ids)
        references = _reference_rows(records, query_index, reference_positions, reference_ids)
        calls = [
            value["metadata"] for key, value in sorted(judge.llm.calls.items())
            if key.startswith(f"{tag}:")
        ]
        reason = None
        if label not in CLASSES:
            if decision["method"] == "tie":
                reason = "The decisive judgments had equal class counts and model score totals."
            else:
                reason = "Too few model judgments supplied a class with usable evidence."
        return MolecularResult(
            identifier, query.canonical, label, confidence, selected,
            panel, judgments, calls, reason, decision, references,
        )

    @staticmethod
    def _judgment(view, number, decision, *, references=None):
        judgment = {
            "view": view,
            "repeat": number,
            "label": decision.label,
            "confidence": decision.confidence,
            "evidence": {
                key: decision.parsed[key]
                for key in ("class_contrast", "test_position", "counter_evidence", "rationale",
                            "scaffold_type", "halogen_change_interpretation", "terminal_class_match",
                            "smallest_structural_difference", "why_this_pair_is_relevant",
                            "reference_ids", "counter_reference_ids")
                if key in decision.parsed
            },
            "parsed": decision.parsed_ok,
            "error": decision.error,
            "call_tag": decision.call.get("tag"),
        }
        if references is not None:
            judgment["references"] = references
        return judgment

    def _focused_views(self, records, query_index, indices, reference_ids, tag, judge, pool):
        views = []
        deletion = judge.build_halogen_deletion_prompt(self.task_name, records, query_index, indices)
        if deletion:
            messages, related = deletion
            choice = judge.judge_panel(messages, tag=f"{tag}:halogen_deletion", reference_positions=related)
            references = _reference_rows(records, query_index, related, reference_ids)
            evidence = self._judgment("halogen_deletion", 0, choice, references=references)
            views.append(("halogen_deletion", choice, [evidence]))
        stereo = judge.build_stereo_gate_pair_prompt(self.task_name, records, query_index, indices)
        if not stereo:
            return views
        pairs = judge.select_pairs(records, query_index, indices, 3, strategy="contrast")
        if not pairs:
            return views
        pair_futures = []
        for pair_index, (active, inactive, score) in enumerate(pairs):
            active_side = "A" if (query_index + pair_index) % 2 == 0 else "B"
            prompt = judge.build_prompt(
                self.task_name, records, query_index, active, inactive, score,
                train_idx=indices, active_side=active_side,
            )
            pair_futures.append(pool.submit(
                judge.judge, prompt, pair_id=pair_index, active_ref=active,
                inactive_ref=inactive, tag=f"{tag}:pair{pair_index}", active_side=active_side,
            ))
        messages, related = stereo
        choice = judge.judge_panel(messages, tag=f"{tag}:stereo_gate_pair", reference_positions=related)
        choices = [future.result() for future in pair_futures]
        evidence = [
            {
                "view": "stereo_pair", "repeat": pair_index,
                "label": ("active" if vote.choice == vote.active_side else "inactive")
                if vote.choice in ("A", "B") else vote.choice,
                "confidence": vote.confidence,
                "evidence": {key: value for key, value in vote.parsed.items()
                             if key in ("rationale", "reference_ids", "counter_reference_ids", "counter_evidence",
                                        "smallest_structural_difference", "why_this_pair_is_relevant")},
                "parsed": vote.parsed_ok, "error": vote.error,
                "call_tag": vote.call.get("tag"),
                "references": _reference_rows(
                    records, query_index, (vote.active_ref, vote.inactive_ref), reference_ids,
                ),
            }
            for pair_index, vote in enumerate(choices)
        ]
        references = _reference_rows(records, query_index, related, reference_ids)
        evidence.append(self._judgment("stereo_gate_pair", 0, choice, references=references))
        if choice.parsed_ok and choice.label in CLASSES and choice.confidence >= 0.6:
            views.append(("stereo_gate_pair", choice, evidence))
        else:
            label, confidence, summary = judge.aggregate(choices)
            eligible = sum(summary["counts"].values()) >= len(choices) // 2 + 1
            decision = FinalChoice(label, confidence, {}, "", {}, eligible and label in CLASSES)
            views.append(("stereo_pair", decision, evidence))
        return views
