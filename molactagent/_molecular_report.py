"""Render a standalone chemical comparison from the recorded decision."""

from __future__ import annotations

from base64 import b64encode
from collections import Counter
from functools import lru_cache
from html import escape
from math import isfinite

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from ._molecules import CLASSES
from .pp.descriptors import compute

_VIEW_NAMES = {
    "panel": "Reference panel",
    "halogen_deletion": "Halogen substitution comparison",
    "stereo_gate_pair": "Stereochemistry and terminal group comparison",
    "stereo_pair": "Individual stereochemical comparisons",
}

_EVIDENCE_HEADINGS = {
    "test_position": "How the test compound fits",
    "class_contrast": "What distinguishes the labeled references",
    "halogen_change_interpretation": "Effect of the change in halogen substitution",
    "terminal_class_match": "Comparison of terminal groups",
    "scaffold_type": "Scaffold shape and flexibility",
    "smallest_structural_difference": "Smallest structural difference",
    "why_this_pair_is_relevant": "Why these references are relevant",
    "rationale": "Chemical rationale",
    "counter_evidence": "Counter-evidence and limitations",
}

_PROPERTIES = {
    "NumHDonors": ("Hydrogen-bond donors", 0),
    "NumHAcceptors": ("Hydrogen-bond acceptors", 0),
    "MolLogP": ("Calculated logP", 2),
    "TPSA": ("Polar surface area (Å²)", 1),
    "MolWt": ("Molecular weight (Da)", 1),
}


def _text(value):
    return escape(str(value)) if value is not None else "—"


def _view_name(view):
    return _VIEW_NAMES.get(view, view.replace("_", " ").capitalize())


@lru_cache(maxsize=256)
def _structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}
    properties = compute(mol)
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(340, 160)
        drawer.drawOptions().fixedBondLength = 35
        drawer.drawOptions().updateAtomPalette({
            8: (0.66, 0.13, 0.14),
            9: (0.00, 0.39, 0.28),
            17: (0.00, 0.39, 0.28),
        })
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return b64encode(drawer.GetDrawingText()).decode("ascii"), properties
    except (AttributeError, RuntimeError, ValueError):
        return None, properties


def _compound_name(identifier, name):
    return f"{_text(name)} <small>{_text(identifier)}</small>" if name and name != identifier else _text(identifier)


def _diagram(smiles, name):
    png, _ = _structure(smiles)
    if png is None:
        return '<span class="diagram-missing">Structure drawing unavailable</span>'
    return (f'<img src="data:image/png;base64,{png}" width="340" height="160" '
            f'alt="2D structure of {_text(name)}">')


def _molecule(smiles, identifier, name, role, style, *, potency=None, note=""):
    measured = f"<span>Input pChEMBL {_text(potency)}</span>" if potency is not None else ""
    return f"""<figure class="structure {_text(style)}">
  <figcaption>
    <span class="role">{_text(role)}</span>
    <strong>{_compound_name(identifier, name)}</strong>
  </figcaption>
  {_diagram(smiles, name or identifier)}
  <div class="structure-meta">{note}<code>{_text(smiles)}</code>{measured}</div>
</figure>"""


def _reference_table(rows, names):
    if not rows:
        return ""
    measured = any(row["pchembl"] is not None for row in rows)
    potency_header = '<th scope="col">Input pChEMBL</th>' if measured else ""
    body = []
    for row in rows:
        position = row["position"]
        name = names.get(position)
        potency = f'<td>{_text(row["pchembl"])}</td>' if measured else ""
        body.append(f"""<tr>
  <th scope="row">{_compound_name(row["reference"], name)}<small>Row {_text(position)}</small></th>
  <td>{_text(row["label"].capitalize())}</td>
  <td class="table-structure">{_diagram(row["smiles"], name or row["reference"])}
    <code>{_text(row["smiles"])}</code></td>
  <td>{row["similarity"]:.3f}</td>{potency}
</tr>""")
    return f"""<div class="table-scroll" role="region" aria-label="Reference compounds" tabindex="0">
<table><thead><tr><th scope="col">Compound</th><th scope="col">Input label</th>
<th scope="col">Structure</th><th scope="col">Structural similarity</th>{potency_header}
</tr></thead><tbody>{"".join(body)}</tbody></table></div>"""


def _property_table(result, references):
    compounds = [(result.identifier, result.smiles)]
    compounds.extend((row["reference"], row["smiles"]) for row in references)
    columns = "".join(f'<th scope="col">{_text(identifier)}</th>' for identifier, _ in compounds)
    descriptors = [_structure(smiles)[1] for _, smiles in compounds]
    rows = []
    for key, (name, digits) in _PROPERTIES.items():
        values = [properties.get(key, float("nan")) for properties in descriptors]
        cells = "".join(f"<td>{value:.{digits}f}</td>" if isfinite(value) else "<td>—</td>" for value in values)
        rows.append(f'<tr><th scope="row">{name}</th>{cells}</tr>')
    return f"""<details class="record">
  <summary>Calculated molecular properties</summary>
  <div class="table-scroll" role="region" aria-label="Calculated molecular properties" tabindex="0">
  <table><thead><tr><th scope="col">Property</th>{columns}</tr></thead>
  <tbody>{"".join(rows)}</tbody></table></div>
</details>"""


def _evidence(vote, field):
    value = vote["evidence"].get(field) if vote else None
    return value.strip() if isinstance(value, str) else ""


def _judgment_for(votes, label):
    candidates = [vote for vote in votes if vote["parsed"] and vote["label"] == label]
    return max(
        candidates,
        key=lambda vote: (
            sum(bool(_evidence(vote, field)) for field in _EVIDENCE_HEADINGS),
            vote["confidence"],
        ),
        default=None,
    )


def _decision_summary(result, votes):
    if not votes:
        return _text(result.reason or "No judgments were recorded for this compound.")
    counts = Counter(vote["label"] for vote in votes if vote["parsed"])
    used = sum(counts[label] for label in CLASSES)
    summary = []
    if result.label in CLASSES:
        if result.selected_view != "panel":
            summary.append(f"{_view_name(result.selected_view)} supplied the final call.")
        summary.append(f"{counts[result.label]} of {used} usable model judgments favored {result.label}.")
        if result.selected_view == "panel" and counts["active"] == counts["inactive"]:
            summary.append("Model scores resolved the tied vote.")
        if result.selected_view == "stereo_pair":
            summary.append("The total model score for each label determined the call.")
    elif used:
        summary.append(f"{counts['active']} active and {counts['inactive']} inactive model judgments "
                       "left the decision unresolved.")
    else:
        summary.append("The model responses yielded no usable activity label.")
    if used < len(votes):
        summary.append(f"{len(votes) - used} of {len(votes)} judgments supplied no usable label.")
    return _text(" ".join(summary))


def _reference_cards(result, name, vote, names):
    panel = result.selected_view == "panel"
    rows = result.references if panel else (vote or {}).get("references", [])
    cited = str((vote or {}).get("evidence", {}).get("decisive_reference", "")).strip()
    cited_row = next((row for row in rows if str(row["position"]) == cited), None)
    by_class = {}
    for label in CLASSES:
        same_label = [row for row in rows if row["label"] == label]
        preferred = cited_row if cited_row and cited_row["label"] == label else None
        reference = preferred or max(same_label, key=lambda row: row["similarity"], default=None)
        if reference:
            by_class[label] = reference
    cards = [_molecule(result.smiles, result.identifier, name, "Test compound", "query")]
    for label, row in by_class.items():
        note = "<span>Cited by this judgment</span>" if row is cited_row else ""
        role = f"Closest {label} reference" if panel and not note else f"{label.capitalize()} reference"
        cards.append(_molecule(
            row["smiles"], row["reference"], names.get(row["position"]), role, label,
            potency=row["pchembl"], note=note,
        ))
    return f'<div class="structures">{"".join(cards)}</div>', list(by_class.values())


def _judgment_id(anchor, vote):
    return f"{anchor}-{vote['view']}-{vote['repeat'] + 1}"


def _source(anchor, vote):
    number = vote["repeat"] + 1
    return f'<a href="#{_text(_judgment_id(anchor, vote))}">Read recorded judgment {number}</a>'


def _excerpt(anchor, vote, *, omit=()):
    statements = []
    for field, heading in _EVIDENCE_HEADINGS.items():
        value = _evidence(vote, field)
        if value and field not in omit:
            statements.append(f'<dt>{heading}</dt><dd>{_text(value)}</dd>')
    if not statements:
        return ""
    return f'<dl class="evidence">{"".join(statements)}</dl><p class="source">{_source(anchor, vote)}</p>'


def _interpretation(anchor, result, chosen, selected):
    if result.label in CLASSES:
        main = _excerpt(anchor, chosen, omit=("test_position", "counter_evidence"))
        counter = _evidence(chosen, "counter_evidence")
        if not main:
            if counter or _evidence(chosen, "test_position"):
                main = f'<p class="source">{_source(anchor, chosen)}</p>'
            else:
                main = "<p>The supporting judgments provided class labels and scores only.</p>"
        if counter:
            main = (f'<div class="interpretation"><div>{main}</div>'
                    f'<aside class="counter"><h4>Counter-evidence and limitations</h4>'
                    f'<p>{_text(counter)}</p></aside></div>')
        opposite = next(label for label in CLASSES if label != result.label)
        alternate = _judgment_for(selected, opposite)
        comparison = "Another judgment in this comparison"
        if alternate is None:
            other = [vote for vote in result.judgments if vote["view"] != result.selected_view]
            alternate = _judgment_for(other, opposite)
            if alternate:
                comparison = f"A judgment from the {_view_name(alternate['view']).lower()}"
        if alternate:
            explanation = _excerpt(anchor, alternate) or f'<p class="source">{_source(anchor, alternate)}</p>'
            main += (f'<section class="counter"><h4>{_text(comparison)} favored {opposite}</h4>'
                     f'{explanation}</section>')
        return main

    sides = []
    for label in CLASSES:
        vote = _judgment_for(selected, label)
        if vote:
            excerpt = _excerpt(anchor, vote) or f'<p class="source">{_source(anchor, vote)}</p>'
            sides.append(f"<section><h4>Case for {label}</h4>{excerpt}</section>")
    return "".join(sides)


def _judgments(anchor, result, names):
    blocks = []
    for vote in result.judgments:
        evidence = "".join(
            f"<dt>{_text(_EVIDENCE_HEADINGS.get(field, field.replace('_', ' ').capitalize()))}</dt>"
            f"<dd>{_text(value)}</dd>"
            for field, value in vote["evidence"].items()
        )
        if vote["error"]:
            evidence += f'<dt>Response error</dt><dd>{_text(vote["error"])}</dd>'
        status = (f'{_text(vote["label"].capitalize())} · model score {vote["confidence"]:.2f}'
                  if vote["parsed"] else "Unusable response")
        refs = vote.get("references", [])
        table = _reference_table(refs, names) if refs else ""
        blocks.append(f"""<details class="judgment">
  <summary>{_text(_view_name(vote["view"]))} · judgment {vote["repeat"] + 1} · {status}</summary>
  <dl class="evidence" id="{_text(_judgment_id(anchor, vote))}">{evidence}</dl>{table}
</details>""")
    calls = result.calls
    truncated = sum(bool(call.get("truncated")) for call in calls)
    errors = sum(bool(call.get("error") or call.get("retry_failures")) for call in calls)
    models = ", ".join(sorted({str(call["response_model"]) for call in calls if call.get("response_model")}))
    backend = f" · Reported model {_text(models)}" if models else ""
    return f"""<details class="record" id="{anchor}-judgments">
  <summary>Recorded judgments ({len(blocks)})</summary>
  <p>Uncalibrated model evidence score {result.confidence:.2f} on a 0–1 scale.</p>
  <p>{len(calls)} recorded calls · {truncated} truncated · {errors} with request errors{backend}</p>
  {"".join(blocks)}
</details>"""


def _compound_report(anchor, result, name, names):
    selected = [vote for vote in result.judgments if vote["view"] == result.selected_view]
    decided = result.label in CLASSES
    chosen = _judgment_for(selected, result.label) if decided else None
    lead = _evidence(chosen, "test_position")
    summary = _decision_summary(result, selected)
    structures, references = _reference_cards(result, name, chosen, names)
    judgment_details = _interpretation(anchor, result, chosen, selected)
    heading = "Why this label?" if decided else "What the evidence shows"
    status = f"Predicted {result.label}" if decided else "Inconclusive"
    style = result.label if decided else "inconclusive"
    lead_html = f'<p class="lead">{_text(lead)}</p>' if lead else ""
    panel = f"""<details class="record"><summary>Reference panel ({len(result.references)})</summary>
<p>Labels and any pChEMBL values come from the reference data. Structural similarity is on a 0–1 scale.</p>
{_reference_table(result.references, names)}</details>""" if result.references else ""
    used_refs = (chosen or {}).get("references", [])
    all_used = (f'<details class="record"><summary>References in the deciding comparison ({len(used_refs)})</summary>'
                f'{_reference_table(used_refs, names)}</details>') if used_refs else ""
    scorecard = _judgments(anchor, result, names) if result.judgments else ""
    reading = "\n    ".join(filter(None, (lead_html, structures, judgment_details)))
    records = "\n    ".join(filter(None, (_property_table(result, references), all_used, panel, scorecard)))
    return f"""<article id="{anchor}" aria-labelledby="{anchor}-title">
  <header class="decision">
    <div>
      <h2 id="{anchor}-title">{_compound_name(result.identifier, name)}</h2>
      <p>{summary}</p>
    </div>
    <strong class="status {style}">{_text(status)}</strong>
  </header>
  <section class="reading">
    <h3>{heading}</h3>
    {reading}
  </section>
  <div class="records">
    {records}
  </div>
</article>"""


_STYLE = """
:root {
  color-scheme: light;
  --ink: #1b2c32;
  --quiet: #4a6068;
  --line: #b7c7ca;
  --blue-wash: #e9f0f5;
  --active-wash: #e9f2e7;
  --active-ink: #2c5233;
  --inactive-wash: #f8eee7;
  --inactive-ink: #713e2b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #f3f6f6;
  color: var(--ink);
  font: 1rem/1.55 system-ui, sans-serif;
}
main { max-width: 69rem; margin: auto; padding: 1.5rem clamp(.75rem, 3.6vw, 3rem) 3.5rem; }
.masthead { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .5rem 1rem;
  padding-bottom: .55rem; border-bottom: 1px solid var(--line); }
.masthead b { letter-spacing: .015em; }
.masthead span, small, .source, .structure-meta { color: var(--quiet); }
h1, h2, h3, h4 { line-height: 1.2; }
h1 { margin: 1.25rem 0 1rem; font-size: 2.35rem; letter-spacing: -.035em; }
h2 { margin: 0; font-size: 1.65rem; letter-spacing: -.025em; }
h2 small { font-size: .9rem; letter-spacing: 0; white-space: normal; }
h3 { margin: 0 0 .6rem; font-size: 1.2rem; }
h4 { margin: 0 0 .5rem; font-size: 1.02rem; }
p { margin: 0; }
p + p { margin-top: .6em; }
code { overflow-wrap: anywhere; font-size: .88em; }
a { color: #22516d; text-underline-offset: .2em; }
a:hover { color: #123344; }
:focus-visible { outline: 2px solid #22516d; outline-offset: 3px; }
::selection { background: #d3e7ef; }
.assay { display: grid; grid-template-columns: 12rem 1fr; gap: .45rem 1rem;
  padding: .8rem 1rem; background: var(--blue-wash); border-block: 1px solid var(--line); }
.assay p { max-width: 74ch; white-space: pre-line; }
.contents { margin: 1rem 0; }
.contents ul { display: flex; flex-wrap: wrap; gap: .3rem 1.5rem; padding-left: 1.2rem; }
article { background: #fff; border: 1px solid var(--line); margin-top: 1.6rem; overflow-wrap: anywhere; }
.decision, .reading, .records { padding: 1.3rem clamp(.85rem, 2.7vw, 1.8rem); }
.decision { display: flex; align-items: start; justify-content: space-between; flex-wrap: wrap;
  gap: .65rem 1rem; border-bottom: 1px solid var(--line); }
.decision p { margin-top: .45rem; color: var(--quiet); max-width: 73ch; }
.status { font-size: .94rem; padding: .32rem .65rem; border: 1px solid currentColor;
  border-radius: .2rem; white-space: nowrap; }
.active { color: var(--active-ink); background: var(--active-wash); }
.inactive { color: var(--inactive-ink); background: var(--inactive-wash); }
.status.inconclusive, .query { background: var(--blue-wash); }
.lead { font-size: 1.08rem; max-width: 72ch; }
.structures { display: grid; grid-auto-columns: minmax(0, 1fr); grid-auto-flow: column;
  margin: 1rem 0 1.2rem; gap: 1px; background: var(--line); border: 1px solid var(--line); }
.structures:has(.active) .active { grid-column: 1; grid-row: 1; }
.structures:has(.active) .query { grid-column: 2; grid-row: 1; }
.structure { min-width: 0; margin: 0; padding: .7rem .8rem .65rem; }
.structure .role, .structure-meta, .structure strong small { display: block; font-size: .82rem; }
.structure .role { font-weight: 650; }
.structure figcaption strong { color: var(--ink); }
.structure strong small { display: inline; margin-left: .25rem; font-weight: 400; }
.structure img { display: block; width: 100%; height: auto; margin: .45rem 0;
  background: #fff; border: 1px solid var(--line); }
.structure:only-child img { max-width: 22rem; }
.structure-meta span { display: block; }
.diagram-missing { display: block; padding: .65rem 0; color: var(--quiet); }
.evidence { margin: 0; max-width: 76ch; }
.evidence dt { font-weight: 670; }
.evidence dd { margin: .22rem 0 .85rem; white-space: pre-line; }
.counter { margin-top: .95rem; padding: .8rem 1rem; background: #f5f3ee; border-block: 1px solid #d7cdb9; }
.counter p { max-width: 76ch; white-space: pre-line; }
.interpretation { display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(15rem, 1fr);
  gap: 1rem 1.5rem; align-items: start; }
.interpretation .counter { margin-top: 0; }
.source { font-size: .88rem; margin-top: .55rem; }
.records { padding-top: 0; }
.record, .judgment { border-top: 1px solid var(--line); padding: .7rem 0; }
.record:last-child { border-bottom: 1px solid var(--line); }
.record p, .judgment { margin-top: .65rem; }
.record p { font-size: .92rem; }
summary { cursor: pointer; font-weight: 650; }
summary:hover { color: #22516d; }
.judgment .evidence { margin-top: .5rem; }
.table-scroll { overflow: auto; margin-top: .6rem; }
table { border-collapse: collapse; width: 100%; text-align: left; font-size: .9rem;
  font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: .45rem .55rem; }
thead th { background: #edf2f3; }
th small { display: block; font-weight: 400; }
.table-structure { min-width: 6rem; }
.table-structure img { display: block; width: 7rem; max-width: 100%; height: auto; }
.run { margin-top: 1.4rem; color: var(--quiet); font-size: .9rem; }
@media (max-width: 46rem) {
  .interpretation { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 36rem) {
  h1 { font-size: 1.9rem; }
  .assay { display: block; }
  .assay p { margin-top: .25rem; }
  .structures { grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-flow: row; }
  .structures .query, .structures:has(.active) .query { grid-column: 1 / -1; grid-row: 1; }
  .structures:has(.active) .active { grid-row: 2; }
  .structure.query img { max-width: 17rem; }
  th, td { padding-inline: .32rem; }
}
@media print {
  body { background: white; }
  main { max-width: none; padding: 0; }
  .structures, .counter, h2, h3 { break-inside: avoid; }
}
"""


def render_molecular_report(results, *, endpoint, run, series="", reference_names=None, query_names=()):
    names = reference_names or {}
    sections = []
    contents = []
    for index, result in enumerate(results):
        anchor = f"compound-{index + 1}"
        name = query_names[index] if index < len(query_names) else ""
        sections.append(_compound_report(anchor, result, name, names))
        contents.append(f'<li><a href="#{anchor}">{_text(name or result.identifier)}</a> '
                        f'· {_text(result.label.capitalize())}</li>')
    navigation = (f'<nav class="contents" aria-label="Compounds"><ul>{"".join(contents)}</ul></nav>'
                  if len(contents) > 1 else "")
    series_html = f'<p>Compound series · {_text(series)}</p>' if series else ""
    mode = "Replay of recorded responses" if run["mode"] == "replay" else "Live prediction"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Molecular activity report | MolActAgent</title>
<style>{_STYLE}</style></head><body><main>
<header class="masthead"><b>MolActAgent</b><span>{mode}</span></header>
<h1>Molecular activity report</h1>
<div class="assay"><strong>Assay and label definition</strong><div><p>{_text(endpoint)}</p>{series_html}</div></div>
{navigation}
{"".join(sections)}
<details class="run">
  <summary>Run settings</summary>
  <p>Requested model <code>{_text(run["model"])}</code> · Profile {_text(run.get("profile"))}
    · Temperature {_text(run["temperature"])} · Maximum output tokens {_text(run["max_tokens"])}</p>
</details>
</main></body></html>
"""
