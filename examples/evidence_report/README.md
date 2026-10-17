# Offline evidence report example

This example exports the interactive HTML viewer through the unchanged
`molactagent.report.render_html_report` function. It needs an installed
MolActAgent package and makes **no API calls**.

From the repository root:

```bash
python examples/evidence_report/render.py
```

Open `outputs/evidence_report.html` inside this example directory. You can
also choose a destination:

```bash
python examples/evidence_report/render.py --output /tmp/molactagent-report.html
```

`illustrative_record.json` contains the complete example record, display
context, two local comparisons, opposing evidence, feature attributions
and uncertainty. **Activity labels, weights, scores and explanations are
synthetic teaching material. They are not measured activities or outputs
from an LLM.** Names and familiar chemical descriptors make the viewer
easy to explore without publishing unpublished study data.

Switch the comparison selector, select a comparison row, inspect the
feature values and open the raw record. Both supporting and opposing
explanations should remain available. See the [report guide](../../docs/reports.md)
for the context schema and instructions for real prediction records.
