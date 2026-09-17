"""Prompt templates for MolActAgent.

Three prompts drive the system:

  1. ``build_tuple_prompt``
     A single per-tuple call that asks the LLM to identify the key
     discriminative drivers, justify why each one matters in domain
     terms, weigh counter-evidence, and emit a structured JSON verdict
     across M candidate classes. Used by classification directly and
     — twice per test sample — by the regression backend (CCR), which
     reformulates regression as two multi-class classifications over
     target bins and feature clusters.

  2. ``build_explanation_prompt``
     A single end-of-pipeline call that turns the aggregated per-tuple
     reasoning into a Markdown report (tables, sections, narrative)
     for classification.

  3. ``build_regression_explanation_prompt``
     The regression-flavoured analogue of the explanation prompt;
     consumes the CCR result schema and produces a numeric-prediction
     report with per-feature direction (increases / decreases / mixed)
     derived from the per-feature Pearson correlation with the target.

A small ``build_feedback_messages`` helper supports retrying the
contrastive call when a rule-based check fails.
"""

from __future__ import annotations

import numpy as np


class PromptBuilder:
    def __init__(
        self,
        feature_names: list[str],
        feature_descriptions: dict[str, str] | None = None,
        background: str | None = None,
    ):
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions
        self.background = background

    # ── value formatting helpers ─────────────────────────────────────

    @staticmethod
    def _fmt_value(v, precision: int = 4) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if v == int(v) and abs(v) < 1e10:
                return str(int(v))
            return f"{v:.{precision}f}"
        return str(v)

    def format_reference_neighbourhood_block(
        self,
        ref_label_id: str,
        ref_value: str,
        ref_features: np.ndarray,
        neighbour_ids: list[str],
        neighbour_values: list[str],
        neighbour_features: list[np.ndarray],
        feature_names: list[str] | None = None,
        precision: int = 4,
    ) -> str:
        """Render a per-reference local-context block.

        The block shows the reference together with its k_local
        nearest reference-set neighbours, all sharing the same column
        layout. The LLM perceives the local "texture" by reading
        across the neighbour rows: a stable cluster looks like a
        column of identical values/labels; a boundary looks like a
        mix; a gradient looks like a monotone trend.

        ``ref_value`` and ``neighbour_values`` carry whatever the
        task at hand calls a "value" — class label for
        classification, target number for regression — already
        formatted as strings.
        """
        names = feature_names if feature_names is not None else self.feature_names
        col_w = max(precision + 6, 10)

        # Determine widths so the table aligns even when ids/values
        # have different lengths between rows.
        all_ids = [ref_label_id] + list(neighbour_ids)
        all_vals = [ref_value] + list(neighbour_values)
        id_w = max(8, max((len(s) for s in all_ids), default=8))
        val_w = max(8, max((len(s) for s in all_vals), default=8))

        header_cells = (
            f"{'Identifier':<{id_w}} | {'Value':>{val_w}} | "
            + " | ".join(f"{n:>{col_w}}" for n in names)
        )
        sep = "-" * len(header_cells)
        rows = [header_cells, sep]

        def _row(idx: str, val: str, feats: np.ndarray) -> str:
            feat_cells = " | ".join(
                f"{self._fmt_value(feats[i], precision):>{col_w}}"
                for i in range(len(names))
            )
            return f"{idx:<{id_w}} | {val:>{val_w}} | {feat_cells}"

        rows.append(_row(ref_label_id, ref_value, ref_features))
        for nid, nval, nfeats in zip(
            neighbour_ids, neighbour_values, neighbour_features,
        ):
            rows.append(_row(nid, nval, nfeats))

        return "\n".join(rows)

    def format_tuple_table(
        self,
        references: list[np.ndarray],
        class_names: list[str],
        test_sample: np.ndarray,
        precision: int = 4,
        feature_names: list[str] | None = None,
    ) -> str:
        """Render a side-by-side table with one column per class reference.

        Layout:

            Feature                  |  ClassA_Ref |  ClassB_Ref |        Test
            -------------------------+-------------+-------------+-------------
            sepal_length             |         5.0 |         6.5 |         6.8
            ...

        Column widths grow to fit the longest header so multi-class
        class names (e.g. ``versicolor_Ref``, 14 chars) keep the
        table cleanly aligned for both the LLM and any human reader.
        """
        names = feature_names if feature_names is not None else self.feature_names
        ref_headers = [f"{cls}_Ref" for cls in class_names]
        # Column width must accommodate the widest header AND a typical
        # numeric value (precision + 6 covers a sign, 5 integer digits,
        # the decimal point, and ``precision`` decimal digits).
        max_header_w = max((len(h) for h in ref_headers), default=0)
        col_w = max(precision + 6, 10, max_header_w, len("Test"))

        header_cells = [f"{h:>{col_w}}" for h in ref_headers] + [f"{'Test':>{col_w}}"]
        hdr = f"{'Feature':<24} | " + " | ".join(header_cells)
        sep = "-" * len(hdr)
        lines = [hdr, sep]
        for i, name in enumerate(names):
            ref_vals = [
                f"{self._fmt_value(refs[i], precision):>{col_w}}"
                for refs in references
            ]
            test_v = f"{self._fmt_value(test_sample[i], precision):>{col_w}}"
            lines.append(f"{name:<24} | " + " | ".join(ref_vals + [test_v]))
        return "\n".join(lines)

    # ── tuple prompt (single call, structured JSON output) ──────────

    def build_tuple_prompt(
        self,
        references: list[np.ndarray],
        class_names: list[str],
        test_sample: np.ndarray,
        feature_names: list[str] | None = None,
        n_total_features: int | None = None,
        n_aligned: dict[str, int] | None = None,
        n_neutral: int | None = None,
        baseline_class: str | None = None,
        reference_neighbourhoods: list[dict] | None = None,
    ) -> list[dict]:
        """Single LLM call that emits a structured multi-way verdict.

        ``n_aligned`` maps each candidate class to the count of
        discriminative features whose value in Test is numerically
        closest to that class's reference. The sums plus ``n_neutral``
        equal the total number of discriminative features. The Python
        side computes these counts deterministically and surfaces them
        as **ground truth** so the LLM does not need to recount; the
        ``baseline_class`` is the argmax (with ties giving
        ``"Unknown"``).

        Output JSON contains:
          label, confidence, key_drivers (with per-feature
          justification), supporting_features, opposing_features,
          summary, counter_evidence, uncertainty.
        """
        system_msg = self._build_system_message(
            reasoning_focus=True, class_names=class_names,
        )
        table = self.format_tuple_table(
            references, class_names, test_sample,
            feature_names=feature_names,
        )

        names = feature_names if feature_names is not None else self.feature_names
        n_disc = len(names)
        filter_note = ""
        if n_total_features is not None:
            filter_note = (
                f"(Showing {n_disc} discriminative features out of "
                f"{n_total_features} total. Non-discriminative features "
                f"where every reference shares the same value are "
                "omitted.)\n"
            )

        desc_lines = []
        if self.feature_descriptions:
            for n in names:
                d = self.feature_descriptions.get(n)
                if d:
                    desc_lines.append(f"  {n}: {d}")
        desc_block = ""
        if desc_lines:
            desc_block = (
                "Feature descriptions for the discriminative subset:\n"
                + "\n".join(desc_lines) + "\n\n"
            )

        baseline_block = ""
        if n_aligned is not None and baseline_class is not None:
            tally_lines = []
            for cls in class_names:
                cnt = int(n_aligned.get(cls, 0))
                tally_lines.append(
                    f"  Test is numerically closest to {cls}_Ref on "
                    f"{cnt} feature(s)."
                )
            neutral = n_neutral if n_neutral is not None else 0
            tally_lines.append(
                f"  Test is equidistant (no closest reference) on "
                f"{neutral} feature(s)."
            )
            baseline_block = (
                "DETERMINISTIC ALIGNMENT TALLY (computed numerically, "
                "given as ground truth)\n"
                + "\n".join(tally_lines) + "\n"
                f"  Baseline verdict from this tally: **{baseline_class}**.\n"
                "  -> Default to this baseline class unless one or more "
                "drivers below give a STRONG, domain-grounded reason to "
                "override it (rare).\n\n"
            )

        # Per-reference texture blocks: each reference together with
        # its k_local nearest reference-set neighbours and their class
        # labels. Lets the LLM perceive the local "neighbourhood
        # composition" — whether each reference is deep in its own
        # class territory or sitting near a class boundary — without
        # any explicit purity computation in Python.
        texture_block = ""
        if reference_neighbourhoods:
            blocks = []
            for nb in reference_neighbourhoods:
                blocks.append(
                    f"=== {nb['ref_label_id']} (class {nb['ref_value']}) ===\n"
                    + self.format_reference_neighbourhood_block(
                        ref_label_id=nb["ref_label_id"],
                        ref_value=nb["ref_value"],
                        ref_features=nb["ref_features"],
                        neighbour_ids=nb["neighbour_ids"],
                        neighbour_values=nb["neighbour_values"],
                        neighbour_features=nb["neighbour_features"],
                        feature_names=feature_names,
                    )
                )
            texture_block = (
                "PER-REFERENCE LOCAL CONTEXT (each chosen reference "
                "and its nearest reference-set neighbours)\n"
                "Use these to gauge whether each reference sits deep "
                "in its own class territory (neighbours all the same "
                "class) or near a boundary (mixed). Treat boundary "
                "references as weaker evidence than territory-deep "
                "ones.\n\n"
                + "\n\n".join(blocks)
                + "\n\n"
            )

        candidate_list = ", ".join(f'"{c}"' for c in class_names)
        user_msg = (
            "You are comparing one Test sample against one reference "
            "sample from each candidate class. Below is a table of "
            "ONLY the discriminative features (i.e. features whose "
            "values are not all identical across the references).\n\n"
            f"Candidate classes for this comparison: {candidate_list}.\n\n"
            f"{filter_note}\n"
            f"{table}\n\n"
            f"{desc_block}"
            f"{baseline_block}"
            f"{texture_block}"
            "TASK\n"
            "----\n"
            "Identify the 3-6 features that most strongly drive the "
            "classification and EXPLAIN WHY each one matters. Use the "
            "domain context you were given to weigh which alignments "
            "are most informative — not every aligned feature is equally "
            "important.\n\n"
            "Reasoning steps to perform internally:\n"
            "  1. Note the deterministic alignment tally above; it tells "
            "     you which class the raw numerical similarity favours.\n"
            "  2. Pick the 3-6 features whose alignment carries the most "
            "     domain weight (use the feature descriptions and the "
            "     domain context).\n"
            "  3. For each driver, name the class it favours and explain "
            "     in one sentence WHY this feature is informative and "
            "     what its alignment implies. If a driver mostly "
            "     separates one pair of classes and is silent on others, "
            "     say so in your explanation.\n"
            "  4. Look for counter-evidence: which features point the "
            "     OTHER way? Could they overturn the baseline? Almost "
            "     always the answer is NO (the tally already integrates "
            "     all features); state this honestly.\n"
            "  5. Choose your final label. It MUST equal the baseline "
            "     class unless the baseline is clearly contradicted by "
            "     a decisive, domain-grounded counter-argument.\n\n"
            "OUTPUT FORMAT (JSON only, no prose, no code fences)\n"
            "-----------------------------------------------------\n"
            "{\n"
            f'  "label": one of [{candidate_list}, "Unknown"],\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "key_drivers": [\n'
            "    {\n"
            '      "feature": "<feature name from the table>",\n'
            f'      "favored_class": one of [{candidate_list}],\n'
            '      "strength": <float 0.0-1.0, how decisive this driver is>,\n'
            '      "test_value": <number>,\n'
            '      "favored_value": <number, the favoured class\'s reference value>,\n'
            '      "why": "<one sentence: why this feature matters '
            'for the classification, citing domain knowledge if relevant>"\n'
            "    }\n"
            "    // 3-6 entries; ordered by strength descending\n"
            "  ],\n"
            '  "supporting_features": ["<feature>", ...],\n'
            '  "opposing_features": ["<feature>", ...],\n'
            '  "summary": "<2-3 sentences on why Test aligns with the '
            'chosen class, mentioning the strongest 1-2 drivers>",\n'
            '  "counter_evidence": "<what would overturn this '
            'verdict; or \'none significant\' if no real counter-signal>",\n'
            '  "uncertainty": "<remaining uncertainty in 1 sentence>"\n'
            "}\n\n"
            "RULES\n"
            "-----\n"
            "- Output ONLY the JSON object. No markdown fences. No prose.\n"
            "- ``label`` MUST equal the baseline class above unless "
            "decisively contradicted; state any override in "
            "``counter_evidence``.\n"
            "- ``confidence`` should reflect the alignment-tally margin "
            "and your certainty about WHY the drivers matter. A "
            "lopsided tally (e.g. 30 favours one class vs 5 each for "
            "the others) with strong rationale is ~0.9; a near-tied "
            "tally is ~0.5-0.6.\n"
            "- Pick Unknown ONLY when the tally is essentially tied "
            "across multiple classes AND no feature is decisively "
            "informative.\n"
            "- ``key_drivers`` MUST contain 3-6 entries, ordered by "
            "``strength`` (highest first); each ``feature`` MUST appear "
            "in the input table; each ``favored_class`` MUST be one of "
            f"the candidate classes [{candidate_list}]."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def build_feedback_messages(
        self,
        original_messages: list[dict],
        original_response: str,
        issues: list[str],
    ) -> list[dict]:
        """Re-ask the LLM to fix specific issues found by rule-checks."""
        issues_text = "\n".join(f"- {iss}" for iss in issues)
        return original_messages + [
            {"role": "assistant", "content": original_response},
            {
                "role": "user",
                "content": (
                    "Your previous JSON had the following issues:\n\n"
                    f"{issues_text}\n\n"
                    "Please correct them and emit the full revised JSON "
                    "object — same schema, no prose, no code fences."
                ),
            },
        ]

    # ── explanation prompt (single call → Markdown report) ──────────

    def build_explanation_prompt(
        self,
        result: dict,
        background: str | None = None,
        class_labels: list[str] | None = None,
        ref_feature_stats: dict | None = None,
    ) -> list[dict]:
        """Build a Markdown report from already-aggregated tuple evidence.

        ``result`` must contain (produced by ``MolActAgent``):
          - final_label, confidence, confidence_basis
          - reference_info (with per-class sample counts and tuple counts)
          - test_feature_values
          - tuple_results (each with key_drivers, summary, etc.)
          - driver_attribution (the new per-class weighted attribution)

        ``ref_feature_stats`` provides per-class mean and range for the
        top driver features. Schema:

            {
              "<feature>": {
                  "<class_name>": {
                      "mean": <float>,
                      "range": (<lo>, <hi>),
                  },
                  ...
              }
            }
        """
        ref_stats = ref_feature_stats or {}
        attribution = result.get("driver_attribution", [])
        tuples = result.get("tuple_results") or result.get("triple_results", [])
        ri = result.get("reference_info", {})
        cb = result.get("confidence_basis", {})

        n_total = ri.get("total_reference_samples", "?")
        per_class_counts = ri.get("class_neighbor_counts", {}) or {}
        per_class_ref_counts = ri.get("class_reference_counts", {}) or {}
        n_comp = ri.get("n_comparisons", "?")
        n_degen = ri.get("n_degenerate_excluded", 0)

        cls_list = class_labels or sorted(per_class_ref_counts.keys()) or []

        # Per-class tuple-vote counts
        per_class_votes: dict[str, int] = {c: 0 for c in cls_list}
        n_uv = 0
        for t in tuples:
            lbl = t.get("label", "Unknown")
            if lbl in per_class_votes:
                per_class_votes[lbl] += 1
            else:
                n_uv += 1
        valid_tuples = max(1, sum(per_class_votes.values()) + n_uv)

        verdict = result.get("final_label", "Inconclusive")
        conf = result.get("confidence", 0.0)

        system_msg = (
            "You are an analyst writing a classification report for a "
            "client who submitted a single sample for review. The client "
            f"wants to know: which class does the sample belong to "
            f"(candidate classes: {', '.join(cls_list)}), and WHY?\n\n"
            "Write the report ABOUT THE SUBMITTED SAMPLE — its measured "
            "properties, how those properties compare to known examples "
            "of each class, and what that comparison implies. Do NOT "
            "describe the classifier's internals as the topic; treat "
            "yourself as the analyst who personally weighed the "
            "evidence.\n\n"
            "RULES\n"
            "-----\n"
            "- Use the domain class names exactly as listed above. "
            "Never write Pos / Neg / Unknown.\n"
            "- When citing a feature, give: the full descriptive name, "
            "the sample's actual value, AND the reference range for "
            "each class. Avoid vague claims.\n"
            "- Use the supplied domain context to explain WHY each "
            "driver matters in plain language. If no domain context "
            "applies, describe features generically.\n"
            "- Output ONLY Markdown. Do not wrap the entire response "
            "in a code fence.\n"
            "- You MAY use Markdown tables (pipe syntax). Use them "
            "where they improve clarity (e.g., the per-driver "
            "comparison).\n"
            "- Use ## for top-level section headings and ### for "
            "subsections within sections. Never use # alone.\n"
            "- Be honest about uncertainty. If the verdict is "
            "Inconclusive or low-confidence, say so directly."
        )

        data: list[str] = []

        bg = background or self.background
        if bg:
            data.append(f"=== Domain Context ===\n{bg}\n")

        if self.feature_descriptions:
            lines = [f"  {n}: {d}" for n, d in self.feature_descriptions.items()]
            data.append("=== Feature Descriptions ===\n" + "\n".join(lines) + "\n")

        # ── Verdict header ──
        data.append(
            "=== RESOLVED VERDICT ===\n"
            f"Classification: {verdict}\n"
            f"Confidence: {conf:.0%}\n"
            f"Evidence basis: {cb.get('basis_text', f'{valid_tuples} tuple comparisons against {n_total} reference samples')}\n"
            f"Confidence rationale: {cb.get('rationale', '')}\n"
        )

        # ── Sample feature profile (top drivers) ──
        test_vals = result.get("test_feature_values", {})
        if test_vals and ref_stats and cls_list:
            data.append(
                "=== Sample Feature Profile (top drivers) ===\n"
                "For each top driver, the sample's value is shown next "
                "to the mean and full range observed among the reference "
                "samples of every candidate class."
            )
            cls_col_w = 14
            header = (
                f"{'Feature':<24} {'Sample':>10}"
                + "".join(
                    f"  {cls + ' mean':>{cls_col_w}} {cls + ' range':>{cls_col_w + 6}}"
                    for cls in cls_list
                )
            )
            data.append(header)

            def _fv(v):
                if v is None:
                    return "—"
                if isinstance(v, float) and v == int(v) and abs(v) < 1e8:
                    return str(int(v))
                if isinstance(v, float):
                    return f"{v:.2f}"
                return str(v)

            def _fr(r):
                if r is None:
                    return "—"
                return f"{_fv(r[0])}-{_fv(r[1])}"

            for fa in attribution[:12]:
                fn = fa["feature"]
                tv = test_vals.get(fn)
                if tv is None:
                    continue
                stats = ref_stats.get(fn, {})
                row = f"  {fn:<22} {_fv(tv):>10}"
                for cls in cls_list:
                    cs = stats.get(cls, {}) if isinstance(stats, dict) else {}
                    row += (
                        f"  {_fv(cs.get('mean')):>{cls_col_w}} "
                        f"{_fr(cs.get('range')):>{cls_col_w + 6}}"
                    )
                data.append(row)
            data.append("")

        # ── Reference dataset & comparison logic ──
        per_class_ref_text = ", ".join(
            f"{n} {cls}" for cls, n in per_class_ref_counts.items()
        ) or "(per-class breakdown unavailable)"
        per_class_nbr_text = ", ".join(
            f"{n} {cls}" for cls, n in per_class_counts.items()
        ) or "(per-class breakdown unavailable)"
        comp_text = (
            "=== Comparison Setup ===\n"
            f"Total reference samples: {n_total} ({per_class_ref_text}).\n"
            f"Tuple comparisons performed: {n_comp} "
            f"(formed from the most-similar neighbours per candidate "
            f"class: {per_class_nbr_text})."
        )
        if n_degen:
            comp_text += (
                f" {n_degen} tuple(s) were excluded because a reference "
                "was numerically identical to the sample."
            )
        data.append(comp_text + "\n")

        # ── Driver attribution (weighted, only cited drivers) ──
        if attribution:
            data.append("=== Cross-tuple Driver Attribution ===")
            data.append(
                "Features below were CITED AS KEY DRIVERS by the per-tuple "
                "analyses. Each line shows: how many tuples cited it, the "
                "class those citations favour, and the weighted strength "
                "of the citations.\n"
            )
            for fa in attribution[:12]:
                desc = f" ({fa['description']})" if fa.get("description") else ""
                dom = fa.get("dominant_class") or fa.get("direction") or "Mixed"
                data.append(
                    f"  {fa['feature']}{desc}: cited in "
                    f"{fa.get('n_cited', 0)}/{n_comp} tuples, "
                    f"net weighted strength = {fa.get('weighted_score', 0):+.2f} "
                    f"toward {dom}"
                )
            data.append("")

        # ── Per-tuple narratives ──
        data.append("=== Per-tuple Reasoning Records ===")
        for i, tr in enumerate(tuples):
            label = tr.get("label", "Unknown")
            cands = tr.get("candidate_classes") or cls_list
            label_disp = label if label in cands else (
                "Inconclusive" if label == "Unknown" else label
            )
            tr_conf = tr.get("confidence", 0.5)
            summary = tr.get("summary", "") or tr.get("reason", "")
            cand_text = f" [classes: {', '.join(cands)}]" if cands else ""
            data.append(
                f"[Tuple {i+1}]{cand_text} verdict = {label_disp} "
                f"(LLM-confidence {tr_conf:.0%}); {summary}"
            )
            drivers = tr.get("key_drivers", [])
            if drivers:
                for d in drivers[:6]:
                    fn = d.get("feature", "?")
                    why = d.get("why", "")
                    favoured = d.get("favored_class", "?")
                    data.append(f"    - {fn} -> {favoured}: {why}")
            ce = tr.get("counter_evidence")
            if ce and ce.lower() not in ("", "none significant", "none"):
                data.append(f"    counter-evidence: {ce}")
        tally_text = ", ".join(
            f"{cnt} -> {cls}" for cls, cnt in per_class_votes.items()
        )
        data.append(f"\nVote tally: {tally_text}, {n_uv} inconclusive.\n")

        # ── Writing instructions ──
        per_class_col_specs = " · ".join(
            f"**{cls} mean (range)**" for cls in cls_list
        )
        data.append(
            "=== Report Instructions ===\n"
            "Produce a Markdown report with the following sections, in "
            "this exact order. Do NOT invent extra sections.\n"
            "\n"
            "## Executive Summary\n"
            f"2-3 sentences. State the verdict ({verdict}) and confidence "
            f"({conf:.0%}). Mention the evidence basis honestly "
            f"({valid_tuples} tuple comparisons against {n_total} reference "
            "samples). One sentence on the headline reason.\n"
            "\n"
            "## Key Drivers\n"
            "Render a Markdown table with these columns: "
            "**Feature** · **Sample value** · "
            f"{per_class_col_specs} · **Verdict**.\n"
            "Include 5-8 rows for the strongest drivers from the "
            "Sample Feature Profile. The Verdict column states which "
            "class the sample's value aligns with for that feature.\n"
            "\n"
            "## Why These Drivers Matter\n"
            "For each of the top 4-6 drivers, write a `### <Feature>` "
            "subsection (1 short paragraph) that weaves together: the "
            "sample's measured value, the reference range for each "
            "class, the alignment verdict, AND a domain-grounded "
            "explanation of WHY this feature is informative for the "
            "candidate classes. Use the per-tuple `why` rationales "
            "above; don't just restate numbers.\n"
            "\n"
            "## Robustness Across Comparisons\n"
            f"Explain that the verdict was checked against {n_comp} "
            "distinct multi-class reference tuples. Cite the vote tally "
            "exactly. Then NAME the features that the per-tuple analyses "
            "CONSISTENTLY cited as drivers in the same direction — these "
            "are the most reliable indicators. Use an unordered list "
            "(`- item`).\n"
            "\n"
            "## Counter-evidence Considered\n"
            "Honestly summarize features or tuple-verdicts that pointed "
            "toward OTHER classes. Explain why they did not overturn "
            "the verdict (or, if confidence is low, acknowledge that "
            "they weaken it).\n"
            "\n"
            "## Confidence and Caveats\n"
            "1-2 sentences contextualizing the "
            f"{conf:.0%} confidence: what it does and does NOT mean "
            f"given a {n_total}-sample reference set and {n_comp} "
            "evaluated tuples. Mention whether more reference data "
            "would materially change the picture.\n"
            "\n"
            "## Recommended Next Steps\n"
            "Numbered list (1. 2. 3.) of 2-4 concrete actions a "
            "domain expert should take next."
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "\n".join(data)},
        ]

    # ── regression explanation (single Markdown call) ────────────────

    def build_regression_explanation_prompt(
        self,
        result: dict,
        background: str | None = None,
        target_name: str = "target",
    ) -> list[dict]:
        """Build a Markdown report from a CCR regression result.

        Consumes the schema produced by
        :class:`molactagent.cross_classification.CrossClassificationRegressor`,
        which is also the schema ``MolActAgent.predict_one`` returns
        for ``task='regression'``:

          - ``predicted_value`` (float or None for refused predictions)
          - ``interval`` (tuple)
          - ``confidence``
          - ``confidence_basis`` (dict)
          - ``reference_info`` (target_range, n_target_bins,
             n_feature_clusters, etc.)
          - ``test_feature_values`` (feature_name -> value)
          - ``tuple_results`` (one row per CCR phase — Phase 1 over
             target bins and Phase 2 over feature clusters — each
             with the phase's mean target across its picked refs,
             the empirical interval, the phase's drivers and a
             rationale; older bootstrap-tuple-shaped results are
             also rendered correctly)
          - ``driver_attribution`` (per-feature direction-aware,
             ``dominant_direction`` set from the feature -> target
             Pearson correlation)
        """
        attribution = result.get("driver_attribution", [])
        tuples = result.get("tuple_results") or []
        ri = result.get("reference_info", {})
        cb = result.get("confidence_basis", {})

        n_total = ri.get("total_reference_samples", "?")
        n_comp = ri.get("n_comparisons", "?")
        target_range = ri.get("target_range")

        predicted = result.get("predicted_value")
        interval = result.get("interval") or (None, None)
        conf = result.get("confidence", 0.0) or 0.0

        if predicted is None:
            verdict_line = (
                f"Predicted {target_name}: REFUSED — the assembled "
                "evidence was too thin or contradictory to produce a "
                "responsible estimate."
            )
        else:
            verdict_line = (
                f"Predicted {target_name}: "
                f"{self._fmt_value(predicted, 4)} "
                f"(interval [{self._fmt_value(interval[0], 4)}, "
                f"{self._fmt_value(interval[1], 4)}])"
            )

        system_msg = (
            "You are an analyst writing a regression-prediction "
            "report for a client who submitted a single sample. The "
            f"client wants to know: what value of {target_name} does "
            "the sample most likely take, and WHY?\n\n"
            "Write the report ABOUT THE SUBMITTED SAMPLE — its "
            "measured properties, where it sits relative to known "
            "examples, and what that position implies for the target "
            "value. Treat yourself as the analyst who personally "
            "weighed the evidence; do NOT describe the system's "
            "internals as the topic.\n\n"
            "RULES\n"
            "-----\n"
            "- Output ONLY Markdown. Do not wrap the entire response "
            "in a code fence.\n"
            "- Use ## for top-level section headings and ### for "
            "subsections within sections. Never use # alone.\n"
            "- When citing a feature, give: the full descriptive "
            "name, the sample's actual value, AND how the predicted "
            "value depends on it (increases / decreases / stable). "
            "Avoid vague claims.\n"
            "- Use the supplied domain context to explain WHY each "
            "driver matters in plain language.\n"
            "- Be honest about uncertainty. A wide interval, a "
            "low confidence, or an outright REFUSED verdict deserves "
            "a frank discussion of why the data did not support a "
            "tighter answer."
        )

        data: list[str] = []

        bg = background or self.background
        if bg:
            data.append(f"=== Domain Context ===\n{bg}\n")
        if self.feature_descriptions:
            lines = [f"  {n}: {d}" for n, d in self.feature_descriptions.items()]
            data.append(
                "=== Feature Descriptions ===\n" + "\n".join(lines) + "\n"
            )

        # ── Verdict header ──
        basis_text = cb.get(
            "basis_text",
            f"{len(tuples)} tuple comparisons against {n_total} "
            "reference samples",
        )
        data.append(
            "=== RESOLVED PREDICTION ===\n"
            f"{verdict_line}\n"
            f"Confidence: {conf:.0%}\n"
            f"Evidence basis: {basis_text}\n"
            f"Confidence rationale: {cb.get('rationale', '')}\n"
        )

        # ── Test sample features ──
        test_vals = result.get("test_feature_values", {})
        if test_vals:
            data.append("=== Sample Feature Profile ===")
            for fn, v in list(test_vals.items())[:25]:
                data.append(f"  {fn}: {self._fmt_value(v, 4)}")
            data.append("")

        # ── Reference / target context ──
        if target_range is not None:
            data.append(
                "=== Reference Target Range ===\n"
                f"Observed target values across the reference set "
                f"span [{self._fmt_value(target_range[0], 4)}, "
                f"{self._fmt_value(target_range[1], 4)}].\n"
            )

        # ── Driver attribution ──
        if attribution:
            data.append("=== Cross-tuple Driver Attribution ===")
            data.append(
                "Features cited as key drivers across the tuples. "
                "``direction`` shows whether higher values of this "
                "feature push the prediction up or down.\n"
            )
            for fa in attribution[:12]:
                desc = f" ({fa['description']})" if fa.get("description") else ""
                direction = fa.get("dominant_direction") or fa.get("direction") or "mixed"
                data.append(
                    f"  {fa['feature']}{desc}: cited in "
                    f"{fa.get('n_cited', 0)}/{n_comp} tuples, net "
                    f"weighted strength = "
                    f"{fa.get('weighted_score', 0):+.2f}, direction "
                    f"= {direction}"
                )
            data.append("")

        # ── Per-tuple narratives ──
        data.append("=== Per-tuple Reasoning Records ===")
        for i, tr in enumerate(tuples):
            pv = tr.get("predicted_value")
            iv = tr.get("interval") or (None, None)
            tr_conf = tr.get("confidence", 0.5)
            rationale = tr.get("rationale", "") or tr.get("summary", "")
            extrap = " [extrapolation flagged]" if tr.get("extrapolation_flag") else ""
            data.append(
                f"[Tuple {i+1}] predicted = "
                f"{self._fmt_value(pv, 4)}  "
                f"interval = [{self._fmt_value(iv[0], 4)}, "
                f"{self._fmt_value(iv[1], 4)}]  "
                f"confidence = {tr_conf:.0%}{extrap}; "
                f"{rationale}"
            )
            for d in (tr.get("drivers") or [])[:6]:
                fn = d.get("feature", "?")
                why = d.get("why", "")
                direction = d.get("direction", "?")
                data.append(f"    - {fn} ({direction}): {why}")
        data.append("")

        # ── Writing instructions ──
        data.append(
            "=== Report Instructions ===\n"
            "Produce a Markdown report with the following sections, "
            "in this exact order. Do NOT invent extra sections.\n"
            "\n"
            "## Executive Summary\n"
            f"2-3 sentences. State the prediction ({verdict_line}) "
            f"and the {conf:.0%} confidence honestly. One sentence "
            "on the headline reason.\n"
            "\n"
            "## Where the Sample Sits\n"
            "Describe in plain language where the sample sits "
            "relative to the references whose target values "
            "MolActAgent considered. Mention the observed target "
            "range and whether the sample's predicted value falls "
            "comfortably inside it or near the edges.\n"
            "\n"
            "## Key Drivers\n"
            "Render a Markdown table with columns: **Feature** · "
            "**Sample value** · **Direction** (increases / "
            "decreases / mixed) · **Net weighted strength**. "
            "Include 4-6 rows for the strongest drivers from the "
            "Cross-tuple Driver Attribution.\n"
            "\n"
            "## Why These Drivers Matter\n"
            "For each of the top 3-5 drivers, write a `### "
            "<Feature>` subsection (1 short paragraph) that weaves "
            "together: the sample's value, the references' values, "
            "the local trend (increasing / decreasing / stable), and "
            "a domain-grounded explanation of WHY this feature "
            "drives the prediction.\n"
            "\n"
            "## Confidence and Caveats\n"
            "1-2 sentences contextualising the confidence level. "
            "Mention how tight the interval is, whether any "
            "extrapolation was flagged, and what would tighten the "
            "estimate (e.g., more reference data near the sample's "
            "feature region).\n"
            "\n"
            "## Recommended Next Steps\n"
            "Numbered list (1. 2. 3.) of 2-4 concrete actions a "
            "domain expert should take next."
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "\n".join(data)},
        ]

    # ── helpers ───────────────────────────────────────────────────────

    def _build_system_message(
        self,
        reasoning_focus: bool = False,
        class_names: list[str] | None = None,
    ) -> str:
        msg = (
            "You are MolActAgent, a multi-class classification analyst. "
            "You compare a Test sample against one reference example "
            "from each candidate class and produce evidence-based "
            "verdicts.\n\n"
            "Your judgments must be grounded in the SPECIFIC feature "
            "values shown to you, not generic priors. When domain "
            "context is provided, USE it to weigh which features matter "
            "most — do not treat all features as equally informative."
        )
        if class_names:
            msg += (
                "\n\nThe candidate classes for this comparison are: "
                + ", ".join(class_names) + "."
            )
        if reasoning_focus:
            msg += (
                "\n\nFor every classification, you must (a) name the "
                "specific drivers that pushed the verdict, (b) name the "
                "class each driver favours, (c) explain why each driver "
                "is informative, and (d) actively consider "
                "counter-evidence."
            )
        if self.background:
            msg += (
                "\n\n=== Domain Context ===\n"
                + self.background
                + "\n(Use this to interpret which features are "
                "meaningful in the relevant domain, not as a rigid "
                "checklist.)"
            )
        if self.feature_descriptions:
            msg += "\n\n=== Feature Descriptions ===\n"
            for name, desc in self.feature_descriptions.items():
                msg += f"  {name}: {desc}\n"
        return msg
