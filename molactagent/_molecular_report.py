"""Render recorded decisions, chemical structures, and evidence in standalone HTML."""

from __future__ import annotations

import re
import json
from base64 import b64encode
from collections import Counter
from functools import lru_cache
from html import escape
from math import isfinite
from urllib.parse import urlsplit

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from ._molecules import CLASSES
from .pp.descriptors import compute

_VIEW_NAMES = {
    "panel": "Reference panel",
    "halogen_deletion": "Halogen comparison",
    "stereo_gate_pair": "Stereochemistry and terminal-group comparison",
    "stereo_pair": "Paired stereochemistry comparisons",
}
_EVIDENCE_HEADINGS = {
    "test_position": "How the test compound fits",
    "class_contrast": "How the references differ",
    "halogen_change_interpretation": "Halogen comparison",
    "terminal_class_match": "Comparison of terminal groups",
    "scaffold_type": "Scaffold and flexibility",
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
_CALL_STATES = {
    "accepted": "Accepted judgment",
    "uncertain": "Model abstained",
    "truncated": "Truncated response",
    "invalid_response": "Invalid response",
    "authentication_error": "Authentication or permission error",
    "request_error": "API request failed",
    "timeout": "Request timed out",
}
_HIGHLIGHT = (0.98, 0.76, 0.38)


def _text(value):
    return escape(str(value)) if value is not None else "—"


def _view_name(view):
    return _VIEW_NAMES[view]


@lru_cache(maxsize=256)
def _terminal_sites(smiles):
    """Index unambiguous terminal single-bond sites by their marked structure."""
    mol = Chem.MolFromSmiles(smiles)
    sites = {}
    if mol is None:
        return sites
    for atom in mol.GetAtoms():
        if atom.GetDegree() != 1 or atom.GetAtomicNum() <= 1 or atom.GetAtomMapNum():
            continue
        bond = atom.GetBonds()[0]
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.GetBondDir() != Chem.BondDir.NONE:
            continue
        index = atom.GetIdx()
        neighbor = bond.GetOtherAtom(atom)
        marked = Chem.RWMol(mol)
        marked.ReplaceAtom(index, Chem.Atom(0))
        signature = Chem.MolToSmiles(marked, isomericSmiles=True)
        site = (neighbor.GetIdx(), index, neighbor.GetSymbol() + "–" + atom.GetSymbol())
        sites[signature] = None if signature in sites else site
    return sites


def _matched_ends(query, reference):
    query_sites, reference_sites = _terminal_sites(query), _terminal_sites(reference)
    pairs = [(left, reference_sites[key]) for key, left in query_sites.items()
             if left and reference_sites.get(key) and left[2] != reference_sites[key][2]]
    return pairs[0] if len(pairs) == 1 else None


@lru_cache(maxsize=256)
def _structure(smiles, site=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}, ""
    properties = compute(mol)
    note = f"Marked bond · {site[2]}" if site else ""

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(340, 176)
        drawer.drawOptions().fixedBondLength = 40
        drawer.drawOptions().setHighlightColour(_HIGHLIGHT)
        drawer.drawOptions().updateAtomPalette({8: (0.66, 0.13, 0.14), 9: (0, 0.39, 0.28)})
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(site[:2]) if site else [],
            highlightBonds=[mol.GetBondBetweenAtoms(*site[:2]).GetIdx()] if site else [],
        )
        drawer.FinishDrawing()
        return b64encode(drawer.GetDrawingText()).decode("ascii"), properties, note
    except (AttributeError, RuntimeError, ValueError):
        return None, properties, note


def _compound_name(identifier, name):
    return f"{_text(name)} <small>{_text(identifier)}</small>" if name and name != identifier else _text(identifier)


def _image(png, alternate):
    if png is None:
        return '<span class="diagram-missing">Structure drawing unavailable</span>'
    return (f'<img src="data:image/png;base64,{png}" width="340" height="176" '
            f'alt="{_text(alternate)}">')


def _molecule(smiles, identifier, name, role, style, *, row=None, row_url="", site=None):
    png, _, feature = _structure(smiles, site)
    alt = f"2D structure of {name or identifier}" + (f". {feature}." if feature else "")
    feature_html = f'<span class="feature">{_text(feature)}</span>' if feature else ""
    measured = _measurement(row) if row else ""
    ref = f'<span><a href="{_text(row_url)}">Reference record</a></span>' if row_url else ""
    return f"""<figure class="structure {_text(style)}">
  <figcaption><span class="role">{_text(role)}</span><strong>{_compound_name(identifier, name)}</strong></figcaption>
  {_image(png, alt)}
  <div class="structure-meta">{feature_html}{measured}{ref}<details><summary>SMILES</summary><code>{_text(smiles)}</code></details></div>
</figure>"""


def _source_link(row):
    url = str(row.get("source_url") or "").strip()
    try:
        parts = urlsplit(url)
        if parts.scheme in {"https", "http"} and parts.hostname and not (parts.username or parts.password):
            return f'<a href="{_text(url)}" rel="noreferrer">Measurement source</a>'
    except ValueError:
        pass
    return ""


def _measurement(row):
    observed = []
    if row.get("measurement"):
        observed.append(_text(row["measurement"]))
    if row.get("pchembl") is not None:
        observed.append(f'pChEMBL {_text(row["pchembl"])}')
    if source := _source_link(row):
        observed.append(source)
    return "<span>" + " · ".join(observed) + "</span>" if observed else ""


def _reference_url(anchor, position):
    return f"#{anchor}-ref-{position}"


def _reference_table(anchor, rows, names, panel_positions):
    if not rows:
        return ""
    measured = any(_measurement(row) for row in rows)
    potency_header = '<th scope="col">Reference measurement</th>' if measured else ""
    body = []
    for row in rows:
        position = row["position"]
        name = names.get(position)
        potency = f'<td>{_measurement(row) or "—"}</td>' if measured else ""
        views = "Retrieved panel" if position in panel_positions else "Additional comparison"
        png, _, _ = _structure(row["smiles"])
        body.append(f"""<tr id="{_reference_url(anchor, position)[1:]}">
  <th scope="row">{_compound_name(row["reference"], name)}</th><td>{_text(row["label"].capitalize())}</td>
  <td class="table-structure">{_image(png, f"2D structure of {name or row['reference']}")}
    <code>{_text(row["smiles"])}</code></td>
  <td>{row["similarity"]:.3f}</td>{potency}<td>{views}</td>
</tr>""")
    return f"""<div class="table-scroll" role="region" aria-label="Reference compounds" tabindex="0">
<table><thead><tr><th scope="col">Compound</th><th scope="col">Input label</th>
<th scope="col">Structure</th><th scope="col">Fingerprint similarity</th>{potency_header}
<th scope="col">Used in</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>"""


def _property_table(result, references, names):
    compounds = [(result.identifier, result.smiles)]
    compounds.extend((names.get(row["position"]) or row["reference"], row["smiles"]) for row in references)
    columns = "".join(f'<th scope="col">{_text(identifier)}</th>' for identifier, _ in compounds)
    descriptors = [_structure(smiles)[1] for _, smiles in compounds]
    rows = []
    for key, (name, digits) in _PROPERTIES.items():
        values = [properties.get(key, float("nan")) for properties in descriptors]
        cells = "".join(f"<td>{value:.{digits}f}</td>" if isfinite(value) else "<td>—</td>" for value in values)
        rows.append(f'<tr><th scope="row">{name}</th>{cells}</tr>')
    return f"""<details class="record">
  <summary>Properties calculated from structure</summary>
  <div class="table-scroll" role="region" aria-label="Calculated molecular properties" tabindex="0">
  <table><thead><tr><th scope="col">Property</th>{columns}</tr></thead><tbody>{"".join(rows)}</tbody></table>
</div></details>"""


def _evidence(vote, field):
    value = vote["evidence"].get(field) if vote else None
    return value.strip() if isinstance(value, str) else ""


def _evidence_groups(votes):
    """Keep model interpretations in request order, grouping exact evidence only."""
    groups = {}
    for vote in votes:
        if vote["parsed"]:
            key = (vote["view"], vote["label"], json.dumps(vote["evidence"], sort_keys=True))
            groups.setdefault(key, []).append(vote)
    return list(groups.values())


def _citations(anchor, positions, rows, names, label):
    links = []
    for position in positions:
        if position in rows:
            identifier = rows[position]["reference"]
            name = names.get(position)
            text = f"{name} ({identifier})" if name and name != identifier else identifier
            links.append(f'<a href="{_reference_url(anchor, position)}">{_text(text)}</a>')
    return f'<p class="ref-links"><strong>{label}</strong> {" · ".join(links)}</p>' if links else ""


def _linked_text(anchor, vote, field, rows, names):
    value = _evidence(vote, field)
    evidence = vote["evidence"]
    allowed = set(evidence.get("reference_ids", [])) | set(evidence.get("counter_reference_ids", []))
    fragments = []
    cursor = 0
    for match in re.finditer(r"\bref(?:erence)?\s*[#=]?\s*(\d+)\b", value, re.I):
        position = int(match.group(1))
        if position in allowed and position in rows:
            fragments.append(_text(value[cursor:match.start()]))
            name = names.get(position) or rows[position]["reference"]
            fragments.append(f'<a href="{_reference_url(anchor, position)}">{_text(name)}</a>')
            cursor = match.end()
    fragments.append(_text(value[cursor:]))
    return "".join(fragments)


def _decision_summary(result, selected):
    decision = result.decision
    counts = decision.get("counts", {})
    method = decision.get("method")
    if not counts:
        return result.reason or "This compound has no recorded judgments.", ""
    planned = decision["planned"]
    summary = [
        ("active", "active"),
        ("inactive", "inactive"),
        ("uncertain", "abstained"),
        ("failed", "failed"),
    ]
    tally = "".join(f"<li><strong>{counts[key]}</strong> {label}</li>" for key, label in summary)
    tally = f'<ul class="evidence-counts" aria-label="Main panel judgments">{tally}</ul>'
    if result.label not in CLASSES:
        if method == "insufficient_votes":
            text = (f"The main panel supplied {decision['decisive']} usable class judgments out of {planned}. "
                    f"This run required {decision['required']}.")
            if any(call.get("status") == "authentication_error" for call in result.calls):
                text += " The AI service returned an authentication or permission error."
            elif result.calls and all(call.get("error") for call in result.calls):
                text += " The AI service failed to return a response. Check the API and response log."
        elif method == "tie":
            text = "The active and inactive judgments tied on both vote count and total model score."
        else:
            text = result.reason or "The evidence left the decision unresolved."
    elif result.selected_view != "panel":
        same_label = sum(vote["parsed"] and vote["label"] == result.label for vote in selected)
        text = (f"The {_view_name(result.selected_view).lower()} supplied the final call. "
                f"{same_label} of {len(selected)} judgments in that comparison supported {result.label}.")
    elif method == "score_tiebreak":
        text = "The usable votes tied by count. Their total model scores resolved the tie."
    else:
        text = f"{counts[result.label]} of {planned} planned main-panel judgments supported {result.label}."
    return text, tally


def _reference_cards(anchor, result, name, references, names, selected):
    cited = set()
    for vote in selected:
        if vote["parsed"]:
            cited.update(vote["evidence"].get("reference_ids", []))
            cited.update(vote["evidence"].get("counter_reference_ids", []))
    displayed = []
    for label in CLASSES:
        candidates = [row for row in references if row["label"] == label]
        if candidates:
            displayed.append(max(candidates, key=lambda row: (row["position"] in cited, row["similarity"])))

    marks = {}
    query_site = None
    comparison = ""
    for row in sorted(displayed, key=lambda row: row["similarity"], reverse=True):
        ends = _matched_ends(result.smiles, row["smiles"])
        if ends:
            query_site, marks[row["position"]] = ends
            comparison = (f'<p class="bond-note"><strong>One structural change.</strong> '
                          f'The marked bond in {_text(name or result.identifier)} is {_text(ends[0][2])}. '
                          f'In <a href="{_reference_url(anchor, row["position"])}">'
                          f'{_text(names.get(row["position"]) or row["reference"])}</a> it is {_text(ends[1][2])}. '
                          'The structures match when the two terminal atoms are replaced by the same marker.</p>')
            break
    cards = [_molecule(result.smiles, result.identifier, name, "Candidate · activity to assess", "query",
                       site=query_site)]
    for row in displayed:
        role = f'{row["label"].capitalize()} reference'
        cards.append(_molecule(row["smiles"], row["reference"], names.get(row["position"]), role, row["label"],
                               row=row, row_url=_reference_url(anchor, row["position"]),
                               site=marks.get(row["position"])))
    return f'<div class="structures">{"".join(cards)}</div>{comparison}', displayed


def _judgment_id(anchor, vote):
    return f"{anchor}-{vote['view']}-{vote['repeat'] + 1}"


def _source(anchor, vote):
    title = _view_name(vote["view"])
    return (f'<a href="#{_judgment_id(anchor, vote)}">'
            f'Read {_text(title.lower())} judgment {vote["repeat"] + 1}</a>')


def _interpretation(anchor, group, rows, names, title):
    vote = group[0]
    evidence = vote["evidence"]
    supports = _citations(anchor, evidence.get("reference_ids", []), rows, names, "Cited references")
    counter = _citations(anchor, evidence.get("counter_reference_ids", []), rows, names, "Cited for the limitation")
    same = f" · Identical evidence in {len(group)} judgments" if len(group) > 1 else ""
    reasoning = _linked_text(anchor, vote, "rationale", rows, names)
    caveat = _linked_text(anchor, vote, "counter_evidence", rows, names)
    if not reasoning:
        reasoning = "This response supplied no written rationale."
    if not caveat:
        caveat = "This response supplied no written limitation."
    return f"""<section class="argument"><h4>{_text(title)}</h4><div class="interpretation">
  <div><p>{reasoning}</p>{supports}<p class="source">{_source(anchor, vote)}{same}</p></div>
  <aside class="counter"><h5>What limits this interpretation?</h5><p>{caveat}</p>{counter}</aside>
</div></section>"""


def _judgments(anchor, result, rows, names):
    blocks = []
    for vote in result.judgments:
        fields = "".join(
            f"<dt>{heading}</dt><dd>{_text(_evidence(vote, field))}</dd>"
            for field, heading in _EVIDENCE_HEADINGS.items() if _evidence(vote, field)
        )
        if vote["error"]:
            fields += f'<dt>Response error</dt><dd>{_text(vote["error"])}</dd>'
        if not vote["parsed"]:
            status = "Unusable response"
        elif vote["label"] == "Uncertain":
            status = "Abstained"
        else:
            status = f'{_text(vote["label"].capitalize())} · model score {vote["confidence"]:.2f}'
        evidence = vote["evidence"]
        citations = _citations(anchor, evidence.get("reference_ids", []), rows, names, "Cited references")
        citations += _citations(anchor, evidence.get("counter_reference_ids", []), rows, names, "Counter-evidence references")
        blocks.append(f"""<details class="judgment">
  <summary>{_text(_view_name(vote["view"]))} · judgment {vote["repeat"] + 1} · {status}</summary>
  <dl class="evidence" id="{_judgment_id(anchor, vote)}">{fields}</dl>{citations}
</details>""")
    return f"""<details class="record" id="{anchor}-judgments">
  <summary>Individual LLM judgments ({len(blocks)})</summary>
  <p class="scale-note">Model scores are supplied by the LLM on a 0–1 scale. They have no probability calibration.</p>
  {"".join(blocks)}
</details>"""


def _call_log(calls):
    if not calls:
        return ""
    counts = Counter(call.get("status") for call in calls)
    summary = ", ".join(f"{count} {state.lower()}" for key, state in _CALL_STATES.items()
                        if (count := counts[key]))
    entries = []
    for call in calls:
        state = _CALL_STATES.get(call.get("status"), "Recorded response")
        code = call.get("detail") or call.get("error")
        details = f" · {_text(code)}" if code else ""
        attempts = call.get("http_attempts")
        attempted = f" · HTTP attempts {_text(attempts)}" if attempts is not None else ""
        retries = call.get("retry_failures", [])
        failed_attempts = ", ".join(
            f"HTTP {retry['status']}" if retry.get("status") else retry.get("type", "Request error")
            for retry in retries
        )
        if failed_attempts:
            attempted += f" · Request errors {_text(failed_attempts)}"
        backend = call.get("response_model")
        reported = f" · Reported model {_text(backend)}" if backend else ""
        entries.append(f"<li><strong>{state}</strong>{details}{attempted}{reported}</li>")
    return f"""<details class="record">
  <summary>API and response log ({len(calls)})</summary>
  <p>{_text(summary.capitalize())}</p><ol>{"".join(entries)}</ol>
</details>"""


def _compound_report(anchor, result, name, names, notes):
    selected = [vote for vote in result.judgments if vote["view"] == result.selected_view]
    decided = result.label in CLASSES
    summary, tally = _decision_summary(result, selected)
    panel = result.evidence_references or result.references
    rows = {row["position"]: row for row in panel}
    for vote in result.judgments:
        rows.update({row["position"]: row for row in vote.get("references", [])})
    rows = {pos: {**row, **{key: notes.get(pos, {}).get(key, "") for key in ("measurement", "source_url")}}
            for pos, row in rows.items()}

    # A focused view can use compounds outside the original panel.
    comparison = {row["position"] for vote in selected for row in vote.get("references", [])}
    compared = [rows[pos] for pos in sorted(comparison)] if comparison else [rows[r["position"]] for r in panel]
    structures, displayed = _reference_cards(anchor, result, name, compared, names, selected)
    groups = _evidence_groups(result.judgments)
    preferred = result.label if decided else "Uncertain"
    groups.sort(key=lambda g: (g[0]["view"] != result.selected_view, g[0]["label"] != preferred))
    readings = []
    seen = set()
    other_readings = []
    for group in groups:
        label = group[0]["label"]
        if label == "Uncertain":
            title = "Why the model abstained"
        elif decided and label != result.label:
            title = f"Another judgment favored {label}"
        else:
            title = f"Evidence considered for {label}"
        argument = _interpretation(anchor, group, rows, names, title)
        standpoint = (group[0]["view"], label)
        if standpoint in seen:
            other_readings.append(argument)
        else:
            readings.append(argument)
            seen.add(standpoint)
    if other_readings:
        readings.append(f'<details class="record"><summary>Other recorded interpretations ({len(other_readings)})'
                        f'</summary>{"".join(other_readings)}</details>')
    if not readings:
        readings.append('<p>No usable chemical interpretation was recorded for this candidate.</p>')

    first_vote = groups[0][0] if groups else None
    lead = _linked_text(anchor, first_vote, "test_position", rows, names) if first_vote else ""
    lead_html = f'<p class="lead"><strong>Model interpretation.</strong> {lead}</p>' if lead else ""
    panel_positions = {row["position"] for row in result.references}
    reference_details = f"""<details class="record"><summary>All references shown to the LLM ({len(rows)})</summary>
  <p>Labels and measurements come from the reference input. Fingerprint similarity is calculated from the structures.</p>
  {_reference_table(anchor, rows.values(), names, panel_positions)}
</details>""" if rows else ""
    status = f"Predicted {result.label}" if decided else "Inconclusive"
    style = result.label if decided else "inconclusive"
    if decided:
        meaning = f"The chemical comparisons favor the {result.label} class in this assay."
    elif any(vote["parsed"] for vote in selected):
        meaning = "The available comparisons leave this candidate’s activity unresolved."
    elif result.reason and not result.calls:
        meaning = result.reason
    else:
        meaning = "The run produced too few usable responses to assess this candidate."
    comparison_note = ("The closest cited reference in each available class is shown. Where a class has no citation, "
                       "its closest available comparison is shown.") if displayed else ""
    repeats_note = ""
    if result.decision.get("planned", 0) > 1:
        repeats_note = ("<p class=\"scale-note\">Repeated requests examine the same main-panel compounds. "
                        "Identical written interpretations are shown together.</p>")
    return f"""<article id="{anchor}" aria-labelledby="{anchor}-title">
  <header class="decision">
    <div><h2 id="{anchor}-title">{_compound_name(result.identifier, name)}</h2>
      <p>{_text(meaning)}</p></div>
    <strong class="status {style}">{_text(status)}</strong>
  </header>
  <section class="reading" id="{anchor}-comparison">
    <h3>Which compounds can you compare?</h3>
    {structures}<p class="scale-note">{comparison_note}</p>
    {_property_table(result, displayed, names) if displayed else ""}
  </section>
  <section class="reading" id="{anchor}-interpretations">
    <h3>What supports the call, and what is still open?</h3>{lead_html}{"".join(readings)}
    <details class="record"><summary>How the final call was reached</summary><p>{_text(summary)}</p>{tally}{repeats_note}</details>
  </section>
  <div class="records">
    {reference_details}
    {_judgments(anchor, result, rows, names) if result.judgments else ""}{_call_log(result.calls)}
  </div>
</article>"""


_STYLE = """
:root {
  color-scheme: light;
  --ink: #163228;
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
  background: #f6f7f2;
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
  border-radius: .2rem; }
.active { color: var(--active-ink); background: var(--active-wash); }
.inactive { color: var(--inactive-ink); background: var(--inactive-wash); }
.status.inconclusive, .query { background: var(--blue-wash); }
.lead { font-size: 1.08rem; max-width: 72ch; }
.structure-meta .feature { color: var(--ink); font-weight: 650; margin-bottom: .3rem; }
.evidence-counts { display: flex; flex-wrap: wrap; gap: .35rem 1rem; list-style: none; padding: 0; margin: .65rem 0 0; font-size: .89rem; }
.evidence-counts strong { font-variant-numeric: tabular-nums; }
.ref-links { font-size: .9rem; margin: .45rem 0; }
.scale-note { color: var(--quiet); font-size: .9rem; }
[id]:target { scroll-margin-top: 2rem; }
tr:target, dl:target { background: #e6f0ea; }
.run code { overflow-wrap: anywhere; }
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
.argument { border-top: 1px solid var(--line); padding: .9rem 0 .7rem; }
.argument h4 { font-size: 1.1rem; margin-bottom: .45rem; }
.argument h5 { font-size: 1rem; margin: 0; }
.bond-note { color: #503814; background: #fff7e2; padding: .65rem .9rem; }
.structure-meta details { margin-top: .35rem; }
.structure-meta summary { font-weight: 500; }
.structure-meta code { overflow-wrap: anywhere; }
.reading .lead { max-width: 72ch; }
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


_OPEN_REFERENCES = """
<script>
function revealCitation() {
  const id = location.hash.slice(1);
  const target = document.getElementById(id);
  if (!target) return;
  for (let parent = target.parentElement; parent; parent = parent.parentElement) {
    if (parent.tagName === "DETAILS") parent.open = true;
  }
  target.scrollIntoView({block: "nearest"});
}
addEventListener("hashchange", revealCitation);
revealCitation();
</script>
"""


def render_molecular_report(results, *, endpoint, run, series="", reference_names=None, query_names=(),
                            reference_notes=None):
    names = reference_names or {}
    sections = []
    contents = []
    for index, result in enumerate(results):
        anchor = f"compound-{index + 1}"
        name = query_names[index] if index < len(query_names) else ""
        sections.append(_compound_report(anchor, result, name, names, reference_notes or {}))
        contents.append(f'<li><a href="#{anchor}">{_text(name or result.identifier)}</a> '
                        f'· {_text(result.label.capitalize())}</li>')
    navigation = (f'<nav class="contents" aria-label="Compounds"><ul>{"".join(contents)}</ul></nav>'
                  if len(contents) > 1 else "")
    cutoff = run.get("pchembl_threshold")
    rule = " · Active at or above this value" if run.get("labels_from_pchembl") else ""
    cutoff_html = f'<p>Assay pChEMBL cutoff · {_text(cutoff)}{rule}</p>' if cutoff is not None else ""
    series_html = f'<p>Compound series · {_text(series)}</p>' if series else ""
    mode = "From saved model responses" if run["mode"] == "replay" else "Live model assessment"
    versions = run.get("versions", {})
    build = (f'<p>MolActAgent {_text(versions["molactagent"])} · RDKit {_text(versions["rdkit"])}</p>'
             if versions else "")
    measurements_credit = ""
    for row in (reference_notes or {}).values():
        if _source_link(row) and urlsplit(str(row["source_url"]).strip()).hostname == "www.ebi.ac.uk":
            if urlsplit(str(row["source_url"]).strip()).path.startswith("/chembl/"):
                measurements_credit = ('<p>ChEMBL reference data · <a href="https://www.ebi.ac.uk/chembl/">EMBL-EBI</a>'
                                       ' · <a href="https://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a></p>')
                break
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Molecular activity report | MolActAgent</title>
<style>{_STYLE}</style></head><body><main>
<header class="masthead"><b>MolActAgent</b><span>{mode}</span></header>
<h1>Molecular activity report</h1>
<div class="assay"><strong>Assay and label definition</strong><div><p>{_text(endpoint)}</p>{cutoff_html}{series_html}</div></div>
{navigation}
{"".join(sections)}
<details class="run">
  <summary>Run settings</summary>
  <p>Requested model <code>{_text(run["model"])}</code> · Profile {_text(run.get("profile"))}
    · Temperature {_text(run["temperature"])} · Maximum output tokens {_text(run["max_tokens"])}</p>
  {build}
</details>
{measurements_credit}
</main>{_OPEN_REFERENCES}</body></html>
"""
