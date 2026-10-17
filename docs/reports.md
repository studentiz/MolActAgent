# Reading and exporting an evidence report

MolActAgent can turn a saved classification record into an interactive,
self-contained HTML report. It shows the prediction alongside the chemical
comparisons and recorded explanations behind it. Exporting this report does
not call an LLM or change the prediction.

## Try it without an API key

After [installing the package](../README.md#installation), run this from the
repository root:

```bash
python examples/evidence_report/render.py
```

Open `examples/evidence_report/outputs/evidence_report.html` in a browser.
No server, API credentials, chemistry extra or network connection is needed
to view the exported file. JavaScript must be enabled.

The example is **illustrative**, with synthetic activity labels and
handwritten explanations. It demonstrates the report interface; it is not
an experimental result or an LLM prediction. The bundled JSON makes every
displayed field inspectable.

## Follow a decision

1. **Prediction overview** gives the compound name, assigned class, number
   of comparisons and aggregated evidence score. This score is not a
   calibrated probability of correctness.
2. **Supporting and opposing evidence** reproduces the saved decision basis,
   counter-evidence and uncertainty. Chemical names replace reference IDs
   in the displayed prose when a name mapping is supplied.
3. **Feature contributions** includes every saved feature attribution and
   all its rationales, together with the comparison numbers where it was cited.
4. **Comparison explorer** lets you select a reference combination. Read its
   complete interpretation, drivers, opposing features and validation
   details. Nothing is shortened to a headline or ellipsis.
5. **Chemical feature comparison** shows every discriminative feature in
   that comparison, including definitions, query and reference values, and
   the nearest reference class. A tie is displayed as neutral.
6. **Evidence aggregation** shows weighted class support, all local
   interpretations, score components, reference pools and retrieved
   neighbors. Select a comparison row with the mouse or keyboard to inspect it.
7. **Original structured prediction record** contains the prediction JSON
   without display-name substitutions. An additional explanation, if
   supplied to the renderer, is also retained in an expandable section.

The report exposes the model's recorded reasoning. Its chemical or
biological explanations do not establish causal mechanisms. Rendering
cannot recover explanations or reference values that were never saved.

## Export your own saved classification

The public function signature remains unchanged:

```python
from copy import deepcopy
import json
from pathlib import Path

from molactagent.report import render_html_report

record = json.loads(Path("prediction.json").read_text(encoding="utf-8"))
display_record = deepcopy(record)
display_record["report_context"] = {
    "sample_name": "My compound",
    "reference_names": {"0": "Phenol", "1": "Anisole"},
    "reference_feature_values": {
        "0": {"MolWt": 94.113, "NumHDonors": 1},
        "1": {"MolWt": 108.140, "NumHDonors": 0},
    },
    "feature_labels": {
        "MolWt": "Molecular mass (Da)",
        "NumHDonors": "Hydrogen-bond donors",
    },
    "feature_descriptions": {
        "MolWt": "Average molecular mass in daltons.",
        "NumHDonors": "Number of hydrogen-bond donor groups.",
    },
    "display_features": ["MolWt", "NumHDonors"],
    "input_description": "Describe the representation and assay context used.",
}

html = render_html_report(display_record, title="My compound — MolActAgent")
Path("compound_report.html").write_text(html, encoding="utf-8")
```

Replace the example names and values with the actual references from your
run. The JSON must be a **single engine prediction record**, not an entire
pipeline summary. For an estimator result, obtain it with
`record = result.to_dict()`; for a pipeline `SampleOutcome`, use
`record = outcome.raw_result`.

The optional `report_context` is presentation metadata:

| Key | Meaning |
| --- | --- |
| `sample_name` | Chemical name of the query. |
| `reference_names` | Reference index to chemical name; JSON object keys are strings. |
| `reference_feature_values` | Reference index to feature-value mapping. Use the same units and representation as `test_feature_values`. |
| `feature_labels` | Short readable labels; original feature codes remain visible. |
| `feature_descriptions` | Chemical meanings and units of the features. |
| `display_features` | Features to put first; all other discriminative features are still shown. |
| `initial_ref_indices` | Optional class-to-reference-index mapping for the comparison shown on opening. |
| `input_description`, `footer` | Optional description of the recorded run and report provenance. |

Reference indices are the indices stored in `tuple_results[*].ref_indices`,
not compound IDs or row numbers from a different data split. Missing
reference values are shown as unavailable; they are not inferred by the
renderer. Retain the original prediction JSON alongside the report.

## Existing reports and API compatibility

The named-compound viewer is enabled by adding a `report_context` mapping
to a classification record. Classification records without that mapping
keep their existing layout. Regression reports also keep their existing
renderer. The [regression backend is experimental](experimental-regression.md),
remains under development, and is not evaluated in the manuscript;
it is shared for community exploration and collaboration.
`class_labels`, `explanation` and `title` remain supported.

The CLI and prediction pipelines do not automatically build this optional
name/value mapping. Their existing `--per-sample` behavior remains as
described in the [tutorial](tutorial.md#9-adding-per-sample-reports).
That path can request an additional LLM-written narrative; rendering a
saved record with `render_html_report` requires no additional model call.

The HTML embeds its data, styling and script locally. Text is inserted
as text, and embedded JSON escapes HTML/script delimiters. A report still
contains the reference information and explanations supplied to it, so
check those contents before sharing it.
