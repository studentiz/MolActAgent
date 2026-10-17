# Changelog

## Unreleased — 2026-10-17

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
