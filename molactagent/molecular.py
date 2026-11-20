"""Binary activity classification from labelled reference SMILES."""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import pandas as pd

try:
    from ._molecules import CLASSES, make_record, tanimoto
    from ._molecular_evidence import FinalChoice, ReferenceJudge
except ModuleNotFoundError as exc:
    if exc.name == "rdkit":
        raise ImportError("Install MolActAgent with the [chem] extra to classify SMILES.") from exc
    raise

from ._molecular_recording import CallRecorder, ReplayClient


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


def _panel_decision(choices: list[FinalChoice]):
    valid = [choice for choice in choices if choice.parsed_ok and choice.label in CLASSES]
    counts = Counter(choice.label for choice in valid)
    weights = {
        label: sum(max(choice.confidence, 1e-6) for choice in valid if choice.label == label)
        for label in CLASSES
    }
    if weights["active"] == weights["inactive"]:
        return "Inconclusive", 0.0
    if counts["active"] == counts["inactive"]:
        label = max(weights, key=weights.get)
        return label, weights[label] / sum(weights.values())
    label = max(CLASSES, key=lambda candidate: counts[candidate])
    confidence = sum(choice.confidence for choice in valid if choice.label == label)
    return label, confidence / counts[label]


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
        max_tokens: int = 16384,
        response_format: bool = True,
        pchembl_threshold: float = 7.0,
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
        if not math.isfinite(pchembl_threshold):
            raise ValueError("pChEMBL threshold must be finite.")
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
        }
        self._records = None
        self._reference_ids = []
        self.recording = None

    @staticmethod
    def _identifiers(frame, id_column, prefix):
        if id_column is None:
            return [f"{prefix}{position + 1}" for position in range(len(frame))]
        if id_column not in frame.columns:
            raise ValueError(f"Missing column {id_column!r}.")
        raw = frame[id_column]
        identifiers = raw.astype(str).str.strip().tolist()
        if raw.isna().any() or "" in identifiers or len(set(identifiers)) != len(raw):
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
        required = [smiles_column, label_column] + ([pchembl_column] if pchembl_column else [])
        for name in required:
            if name not in reference.columns:
                raise ValueError(f"Missing reference column {name!r}.")
        labels = reference[label_column].astype(str).str.strip().str.lower().tolist()
        if set(labels) != set(CLASSES):
            raise ValueError("Reference labels must include both active and inactive.")
        ids = self._identifiers(reference, id_column, "R")
        values = reference[pchembl_column] if pchembl_column else [None] * len(ids)
        records = []
        for row, (smiles, identifier, label, measured) in enumerate(
            zip(reference[smiles_column], ids, labels, values)
        ):
            potency = None
            if not pd.isna(measured):
                try:
                    potency = float(measured)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid pChEMBL value at {identifier!r}.") from exc
                if not math.isfinite(potency):
                    raise ValueError(f"Invalid pChEMBL value at {identifier!r}.")
            records.append(make_record(smiles, row, identifier, label, potency))
        self._records = records
        self._reference_ids = ids
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
                "schema": 1,
                "model": getattr(self.client, "model_name", "unknown"),
                "settings": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "response_format": self.response_format,
                },
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
                "Both reference classes are required after excluding the query molecule.",
            )

        unique_references = len({records[index].canonical for index in indices})
        size = 4 if self.neighbors == 8 and unique_references <= 6 else self.neighbors
        active, inactive = judge.select_panel(records, query_index, indices, n_per_class=size)
        messages = judge.build_panel_prompt(
            self.task_name, records, query_index, indices, active, inactive,
        )
        futures = [
            pool.submit(
                judge.judge_panel, messages, tag=f"{tag}:panel:r{repeat}", force_decision=True,
            )
            for repeat in range(self.repeats)
        ]
        panel_choices = [future.result() for future in futures]
        label, confidence = _panel_decision(panel_choices)
        selected = "panel"
        judgments = [
            self._judgment("panel", repeat, vote) for repeat, vote in enumerate(panel_choices)
        ]
        if self.profile == "focused":
            focused = self._focused_views(records, query_index, indices, tag, judge, pool)
            for view, choice, evidence in focused:
                judgments.extend(evidence)
                if choice is None or not choice.parsed_ok or choice.label not in CLASSES:
                    continue
                if view == "halogen_deletion":
                    minimum = 0.6 if choice.label == label else 0.5
                    if choice.confidence < minimum:
                        continue
                label, confidence, selected = choice.label, choice.confidence, view

        panel = [
            {
                "reference": reference_ids[index],
                "position": index,
                "smiles": records[index].canonical,
                "label": records[index].label,
                "similarity": tanimoto(query.morgan, records[index].morgan),
                "pchembl": records[index].pchembl,
            }
            for index in (*active, *inactive)
        ]
        calls = [
            value["metadata"] for key, value in sorted(judge.llm.calls.items())
            if key.startswith(f"{tag}:")
        ]
        reason = "The model judgments did not produce a decisive vote." if label not in CLASSES else None
        return MolecularResult(
            identifier, query.canonical, label, confidence, selected, panel, judgments, calls, reason,
        )

    @staticmethod
    def _judgment(view, number, decision):
        return {
            "view": view,
            "repeat": number,
            "label": decision.label,
            "confidence": decision.confidence,
            "evidence": {
                key: decision.parsed[key]
                for key in ("class_contrast", "test_position", "counter_evidence", "rationale",
                            "pchembl_estimate", "scaffold_type")
                if key in decision.parsed
            },
            "parsed": decision.parsed_ok,
            "error": decision.error,
        }

    def _focused_views(self, records, query_index, indices, tag, judge, pool):
        views = []
        deletion = judge.build_halogen_deletion_prompt(self.task_name, records, query_index, indices)
        if deletion:
            choice = judge.judge_panel(deletion, tag=f"{tag}:halogen_deletion")
            views.append(("halogen_deletion", choice, [self._judgment("halogen_deletion", 0, choice)]))
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
        choice = judge.judge_panel(stereo, tag=f"{tag}:stereo_gate_pair")
        choices = [future.result() for future in pair_futures]
        evidence = [
            {
                "view": "stereo_pair", "repeat": pair_index,
                "label": ("active" if vote.choice == vote.active_side else "inactive")
                if vote.choice in ("A", "B") else "Inconclusive",
                "confidence": vote.confidence, "evidence": vote.parsed,
                "parsed": vote.parsed_ok, "error": vote.error,
            }
            for pair_index, vote in enumerate(choices)
        ]
        evidence.append(self._judgment("stereo_gate_pair", 0, choice))
        if choice.parsed_ok and choice.label in CLASSES and choice.confidence >= 0.6:
            views.append(("stereo_gate_pair", choice, evidence))
        else:
            label, confidence, _ = judge.aggregate(choices)
            decision = FinalChoice(label, confidence, {}, "", {}, label in CLASSES)
            views.append(("stereo_pair", decision, evidence))
        return views
