# Reading and exporting an evidence report

A MolActAgent report helps you evaluate an activity assessment and decide
what to measure next. Open `report.html` in a browser and start with three
questions.

1. **What did the run conclude?** Read the assigned class in the context
   of your assay. `Inconclusive` means the run could not support a class.
2. **Which measured compounds support that assessment?** Follow the
   citations to the reference structures and any supplied measurements
   and source links. Check which compounds were tested under comparable
   conditions and which features they share with your candidate.
3. **What points the other way, and what would help resolve it?** Read
   the opposing evidence and any missing comparison. If the model points to a
   substitution, for example, consider whether measuring an analog
   with just that change in the same assay would test the explanation.

[Reports from molecular structures](#reports-from-smiles) · [Reports from numeric tables](#reports-from-numeric-tables)

## Reports from SMILES

### Inspect the molecular evidence

Read the assay definition and the candidate's verdict at the top of the
report. `Inconclusive` leaves the activity class unresolved.

- In **Which compounds can you compare?**, open a cited reference record
  to inspect the supplied structure and input class. The report's 2D
  drawings, fingerprint similarities and descriptors are calculated from
  the submitted structures. Reference measurements come from the input.
  Follow the measurement source to check its endpoint and units.
- In **What supports the call, and what is still open?**, read the visible
  arguments and their limitations. The report shows the first recorded
  argument for each comparison view and outcome, including opposing
  judgments and abstentions when present. It groups identical recorded
  evidence within each view and outcome. Open **Other recorded
  interpretations** for additional distinct arguments from the same
  views and outcomes.
- Expand **How the final call was reached** to see the decision rule and
  the main-panel counts for active, inactive, abstained and failed
  judgments when recorded. An inconclusive result can reflect a missing
  reference class, too few decisive judgments or a tied result. If a
  focused comparison supplied the final decision, review it alongside
  the original panel counts. [Molecular decision rules](design.md#check-the-replies-and-decide) · [Focused comparisons](design.md#focused-molecular-comparisons)
- Open **Individual LLM judgments** for each judgment's evidence,
  citations and any model score. Open **API and response log** for
  failed requests, truncated or invalid replies and retries. An abstention is a valid
  `Uncertain` reply with an explanation. A corrected reply belongs to
  the original judgment. Model scores are uncalibrated self-reports on
  a 0–1 scale.

Read the model's text as an interpretation of the class evidence for
your candidate. The report's measured values belong to the references. Keep
`predictions.json` with the HTML for reference rows, individual judgments
and call metadata. Rendering uses the saved results and makes no
additional model call. [SMILES tutorial](tutorial.md#predict-from-smiles)

### Understand the reference fields

| Field | How to read it |
| --- | --- |
| Reference ID and name | The CLI uses a supplied `--id-column` or generates row-based IDs. The optional `name` column gives a display name. Use the persistent ID to trace the compound to its source. |
| Class and pChEMBL | The class comes from the input label. A value supplied via `--pchembl-column` is numeric reference potency used in the model comparison. |
| `measurement` | Optional free text from the reference CSV, such as the original value with units and a comparison sign. It appears on the reference record for human review. |
| `source_url` | Optional HTTP(S) link from the reference CSV to the source for that reference. Follow it to check the assay, units and measurement. |
| Similarity and properties | Comparisons and descriptors calculated from the SMILES. Similarity helps locate close analogs and cannot establish that a measured activity transfers to the query. |

The CLI uses `measurement` and `source_url` only for display. The
model receives the supplied pChEMBL values as numeric potency evidence. The CSV and original source give the context for how
that potency was recorded. A molecular report can contain supplied
structures, measurements and model text, so review it before sharing.

### Replay a report or open an example

Replay a molecular recording with `--replay`, the matching RDKit version
and the original reference and query files. With `--output-dir`, the CLI
writes both `report.html` and `predictions.json`. An inconclusive
prediction exits with status 4 and still writes both outputs. v0.5.1
also accepts compatible v0.5.0 schema 2 recordings. File and request
checks still apply, so preserve the original files and headers.
[Offline replay tutorial](tutorial.md#replay-a-complete-prediction) · [How replay checks a recording](design.md#replay-molecular-decisions)

- **Published assay example.** The CDK2 report shows an inconclusive
  result. Expand **How the final call was reached** to see one active,
  zero inactive, four abstained and zero failed judgments.
  [CDK2 case report](https://studentiz.github.io/MolActAgent/cdk2-report.html) · [Case files](../examples/cdk2/)
- **Synthetic teaching example.** Compare the hydroxyl
  attachment in 4-fluorophenol and p-cresol with the side-chain hydroxyl
  in benzyl alcohol. The input labels follow a teaching rule.
  [Hydroxyl teaching report](https://studentiz.github.io/MolActAgent/molecular-example.html)

## Reports from numeric tables

The feature-table engine records comparisons between rows of numeric
descriptors or measured properties. Each comparison, called a tuple,
contains the query and one reference per candidate class. The
interactive named-compound viewer displays a saved classification
record with the feature values and explanations available for review.
It renders offline and makes no model requests.
[How the numeric engine decides](design.md#decide-a-tuple-from-its-feature-alignment)

### Try the viewer offline

Install MolActAgent, then run this from the repository root. Open
`examples/evidence_report/outputs/evidence_report.html` in a browser
with JavaScript enabled. The included JSON uses synthetic labels,
scores and handwritten explanations to demonstrate the viewer.
[Installation guide](tutorial.md#install-the-example-environment) · [Example files](../examples/evidence_report/)

```bash
python examples/evidence_report/render.py
```

### Follow a feature-table decision

1. **Prediction overview** shows the assigned class, number of
   comparisons and aggregate evidence score. The score summarizes the
   class weight margin, size of the smallest candidate reference group
   and share of tuples that gave a class. It has no probability
   calibration. [Score details](design.md#combine-tuple-results)
2. **Supporting and opposing evidence** shows the saved decision basis,
   counter-evidence and uncertainty. If you supply names, the viewer
   expands recognized references in the displayed prose.
3. **Comparison explorer and chemical feature comparison** show a
   selected tuple, its interpretation and its discriminative feature
   values. Check which reference value is closest to the query for
   each feature. Ties appear as neutral. Inspect the saved tuple
   label, numeric baseline and any fallback reason. The engine can
   record an explanation favoring one class while the label follows
   the numeric baseline.
4. **Feature contributions and evidence aggregation** show the saved
   feature attributions, their rationales and comparison numbers,
   weighted class support and the retrieved references. Select a
   comparison row with the mouse or keyboard to inspect it. A driver
   attribution describes the recorded interpretation. Experimental
   evidence is needed to establish a causal effect.
5. **Original structured prediction record** contains the input JSON
   with the original text. An additional explanation, when supplied
   to the renderer, is available in an expandable section. Fields
   missing from the saved record may be unavailable in the viewer.

### Export your own saved classification

Pass a single feature-table prediction record to
`molactagent.report.render_html_report`. Obtain it with
`result.to_dict()` from an estimator result or `outcome.raw_result`
from a pipeline `SampleOutcome`. Add `report_context` to supply
presentation details. The renderer reads that dictionary to enable
the named-compound viewer. Replace the example indices, names and values
below with those of your run.

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

| `report_context` key | What to supply |
| --- | --- |
| `sample_name` | Display name of the query. |
| `reference_names` | Reference index to display name. Use string keys in JSON. |
| `reference_feature_values` | Reference index to feature-value mapping in the same units and representation as `test_feature_values`. |
| `feature_labels`, `feature_descriptions` | Readable names, definitions and units. The feature codes stay visible. |
| `display_features` | Features to show first. The viewer also shows the other discriminative features. |
| `initial_ref_indices` | Optional class-to-reference-index mapping to choose the initial comparison. |
| `input_description`, `footer` | Optional run description and provenance. |

Use the indices found in `tuple_results[*].ref_indices` to map the
reference values and names. Missing reference values appear as
unavailable. Keep the original JSON alongside the HTML.

### Batch and per-sample outputs

`molactagent run --output-dir` can generate a batch summary and aggregate
interpretability reports from the saved comparisons. Add `--per-sample`
for individual HTML, Markdown and JSON reports. The CLI can request an
additional model-written narrative for HTML or Markdown. Use
`--per-sample --per-sample-format json` for the structured records alone.
Rendering a saved record with `render_html_report` uses the supplied data
and optional explanation. The function also accepts `class_labels` and `title`.

The CLI uses the standard feature-table layout. Supply `report_context`
yourself for the optional named-compound viewer. The renderer routes
regression records to a dedicated layout. Regression is experimental
and remains under development.
[Per-sample tutorial](tutorial.md#adding-per-sample-reports) · [Regression status](experimental-regression.md) · [Renderer source](../molactagent/report.py)
