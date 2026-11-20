# Changelog

## 0.6.0 — 2026-11-20

- Added a retrospective COS-1 cell-assay example with seven measured references and four saved candidate decisions. The project site and scientific animation follow the same compounds through to the chemist's shortlist.
- Molecular reference panels count a repeated isomeric structure once. Its records retain their individual measurements and sources. The prompt asks the model to establish where structures can be compared across series.
- Added a batch comparison to the standalone HTML report. A failed request, an explicit abstention and opposing valid judgments have separate outcomes.
- Live molecular batches save completed molecules and, when requested, their model replies. Use the original invocation with `--resume` to recover the batch after an interruption.
- Feature-table tuple judgments retain the accepted model label, cited numerical values, class tally and decision source. Invalid citations and missing or out-of-range model scores trigger response validation. An invalid response contributes no class vote. Users revisiting numeric work can inspect the original record or run the original software tag to reconstruct its historical decision rule.
- The starter configuration uses 16,384 output tokens. Numeric classification honors the configured client budget. Historical v0.5.0 and v0.5.1 molecular recordings use their original comparison instructions on replay.

## 0.5.1 — 2026-11-20

- The project site and README introduce the experimental decision that MolActAgent supports. The tutorial walks through a live assessment of your own molecules. A command reference covers numeric tables, multi-class workflows, Python use, automation and troubleshooting.
- Molecular reports show a candidate with labeled references, original measurement text and optional source links. When two displayed compounds differ at one unambiguous terminal atom, the report marks the corresponding bonds in both structures. Repeated identical written interpretations share one entry. Other interpretations, limitations, abstentions and failed calls remain available for inspection.
- Added an optional public, single-assay example with a withheld query measurement, four traceable ChEMBL references, model responses and the resulting inconclusive assessment.
- Molecular recordings from v0.5.0 can also be replayed on v0.5.1 with matching RDKit and original inputs. Replay verifies the request digests. The molecular prompts and class decision rules are shared by these two releases.

## 0.5.0 — 2026-11-20

- Rebuilt GitHub Pages and the public guides around an illustrated example of the SMILES agent. The report now leads with the class, evidence coverage, structures, rationale and missing comparisons.
- Added validated reference citations, explicit LLM abstention and a minimum of usable class judgments from more than half of the requested main-panel votes.
- Preserved CSV identifiers and checked conflicting labels on identical canonical structures. Matched stereochemical comparisons require assigned configurations and preserve isotope identity. Users can declare and check a pChEMBL cutoff label rule.
- Added preflight checks for output paths and cleanup of incomplete output files. Completed batches with inconclusive queries write all results and exit with code 4.
- Added schema 2 recordings with file fingerprints and effective settings. Schema 1 recordings use the v0.4.1 release and the original run options.
- Aligned the SMILES default output cap at 16,384 tokens and constrained RDKit to compatible versions. The request client records HTTP attempts, retries transient API failures, and always sends the initial request with `--max-retries 0`.

## 0.4.1 — 2026-11-20

- Redesigned SMILES evidence reports for experimental review with 2D query and reference structures, calculated properties, and recorded chemical rationale, counter-evidence and disagreement.
- Added optional compound names in SMILES reports and retained the reference rows for each focused model judgment.

## 0.4.0 — 2026-11-20

- Added an opt-in SMILES binary classifier for labelled active and inactive references. Python and CLI use the same molecular engine.
- Added per-class reference retrieval, structural and property comparison, repeated model judgments, and evidence reports.
- Added an offline replay of recorded responses for a synthetic-label teaching assay and instructions for live prediction.
- Rebuilt GitHub Pages around the SMILES workflow and updated the tutorial and design notes.

### Feature-table Documentation and Reports

- Focused the README on classification and evidence reports. Marked the
  retained regression documentation, examples and API docstrings as an
  experimental community extension outside the manuscript's evaluation;
  the regression implementation and interfaces are unchanged.
- Added an optional interactive HTML evidence viewer for named compounds,
  enabled by presentation metadata in a classification record. The public
  `render_html_report` signature and existing classification/regression
  rendering paths remain compatible.
- Preserved complete supporting and opposing explanations, feature
  rationales, local comparisons, numerical values, aggregation details
  and the original prediction record in the viewer.
- Added an offline illustrative example and a report guide covering
  export, reference-name mappings, feature definitions and interpretation.
- Added report regression tests for lossless records, script delimiter
  escaping and template-token handling.

## 0.3.0 — 2026-09-17

- Rebranded the original codebase as MolActAgent.
- Renamed the canonical Python package, public engine/classes, CLI, environment variables, and logging identifiers.
- Positioned the project around small-data molecular activity prediction while retaining the generic tabular API.
- Added an original molecular-agent SVG logo and refreshed the documentation site copy.
- Added open-source project metadata, data provenance guidance, and a GitHub Actions test workflow.
