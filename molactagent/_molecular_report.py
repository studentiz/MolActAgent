"""Render model judgments and reference panels as a standalone HTML report."""

from __future__ import annotations

from html import escape


def _text(value):
    return escape(str(value)) if value is not None else "—"


def _references(result):
    return "".join(
        f"""<tr><td>{_text(row["position"])}</td><th scope="row">{_text(row["reference"])}</th>
        <td><code>{_text(row["smiles"])}</code></td><td>{_text(row["label"])}</td>
        <td>{row["similarity"]:.3f}</td><td>{_text(row["pchembl"])}</td></tr>"""
        for row in result.references
    )


def _judgments(result):
    blocks = []
    for judgment in result.judgments:
        evidence = "".join(
            f"<dt>{_text(name.replace('_', ' ').capitalize())}</dt><dd>{_text(value)}</dd>"
            for name, value in judgment["evidence"].items()
        )
        if judgment["error"]:
            evidence += f"<dt>Error</dt><dd>{_text(judgment['error'])}</dd>"
        blocks.append(
            f"""<details><summary>{_text(judgment["view"].replace("_", " ").capitalize())}
            · {judgment["repeat"] + 1} · {_text(judgment["label"])}
            · confidence {judgment["confidence"]:.2f}</summary><dl>{evidence}</dl></details>"""
        )
    return "".join(blocks)


def render_molecular_report(results, *, endpoint, run):
    sections = []
    for result in results:
        status = f"<p>{_text(result.reason)}</p>" if result.reason else ""
        sections.append(f"""
<article>
  <header class="decision"><div><h2>{_text(result.identifier)}</h2>
    <code>{_text(result.smiles)}</code></div><strong>{_text(result.label.capitalize())}</strong></header>
  <p>Selected view: {_text(result.selected_view.replace("_", " "))} · Confidence: {result.confidence:.2f}</p>{status}
  <h3>Retrieved references</h3>
  <div class="table"><table><thead><tr><th scope="col">Row</th><th scope="col">ID</th>
  <th scope="col">SMILES</th><th scope="col">Label</th><th scope="col">Similarity</th>
  <th scope="col">pChEMBL</th></tr></thead><tbody>{_references(result)}</tbody></table></div>
  <h3>Model judgments</h3>
  {_judgments(result)}
</article>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Molecular activity evidence | MolActAgent</title>
<style>
body {{ margin: 0 auto; padding: 3rem clamp(1rem, 4vw, 4rem); max-width: 70rem;
       background: #f7faf8; color: #142822; font: 1rem/1.6 system-ui, sans-serif; }}
h1,h2,h3 {{ line-height: 1.15; }} article {{ margin-block: 2rem; padding: clamp(1rem, 3vw, 2rem);
border: 1px solid #71867c; background: white; }} code {{ overflow-wrap: anywhere; }}
.decision {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
             align-items: center; padding: 1em; background: #e8f1f9; }}
.decision h2 {{ margin: 0; }} .decision strong {{ border: 1px solid; padding: .3em .8em; }}
.table {{ overflow: auto; }} table {{ width: 100%; border-collapse: collapse; text-align: left; }}
td,th {{ border-bottom: 1px solid #aebcb4; padding: .65em; }}
details {{ margin-top: .65em; border: 1px solid #aebcb4; padding: .65em .9em; }}
summary {{ cursor: pointer; }} dt {{ font-weight: 650; }} dd {{ margin: 0 0 .85em; }}
</style></head><body><main>
<p>MolActAgent</p><h1>Molecular activity evidence</h1>
<p>{_text(endpoint)}</p>
<p>Mode: {_text(run["mode"])} · Model: <code>{_text(run["model"])}</code> ·
Temperature: {_text(run["temperature"])} · Max tokens: {_text(run["max_tokens"])}</p>
<p>Confidence describes the model judgments and their aggregation. Treat it as an evidence score.</p>
{"".join(sections)}
</main></body></html>
"""
