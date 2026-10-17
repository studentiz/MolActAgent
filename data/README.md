# Data

This directory contains small, reproducible inputs for MolActAgent's chemistry examples and benchmark scripts. The package itself does not write to this directory.

## Included inputs

| Path | Purpose |
| --- | --- |
| `antiTB_evidence_table.csv` | Small binary activity-classification example. |
| `BSLT_evidence_table.csv` | Small binary BSLT activity-classification input. |
| `BSLT_huigui.csv` | BSLT SMILES plus continuous target values for experimental regression scripts. |
| `benchmarks/` | Raw ChEMBL evidence tables used for chemistry benchmark experiments. |

Regression is an experimental extension, remains under development and
is not evaluated in the accompanying manuscript. These inputs are
retained for community exploration and collaboration; see
[status and limitations](../docs/experimental-regression.md).

Generated LOOCV results, reports, temporary tables, cached features, and downloaded raw sources are intentionally excluded from the public working tree and are covered by `.gitignore`.

For provenance and redistribution notes, see [`SOURCES.md`](SOURCES.md). Do not add private or unlicensed experimental data here.
