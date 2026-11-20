# An example assessment with measured references

This case lets you follow an inconclusive assessment back to four compounds tested in the same assay. The closest active reference has a hydroxyl where the candidate has bromine. Its measured 87 nM is close to the example's 100 nM active cutoff. The agent recorded one judgment favoring active and four abstentions. You can inspect what both sides cited. [Open the evidence report](https://studentiz.github.io/MolActAgent/cdk2-report.html)

## Where the measurements came from

The inputs are derived from the repository's ChEMBL 36 extract for human cyclin-dependent kinase 2. The source publication reports inhibition as IC₅₀. We took all five rows for assay CHEMBL897015 present in the **bundled extract**, in their source order. We reserved the compound with the highest numeric ChEMBL accession as the query and assigned it the neutral identifier Q1. Its input contains only that identifier and its structure. The other four rows supply the reference panel. This is a retrospective illustration with a published query structure. [Bundled source table](../../data/benchmarks/CDK2_CHEMBL301/CDK2_CHEMBL301_raw_evidence_table.csv) · [Original publication](https://doi.org/10.1016/j.bmcl.2006.12.011)

| Role | Compound | ChEMBL activity record | Measurement supplied |
| --- | --- | --- | --- |
| Inactive reference | CHEMBL151 | 1963778 | 20,000 nM |
| Active reference | CHEMBL232556 | 1963780 | 56 nM |
| Active reference | CHEMBL232557 | 1963802 | 87 nM |
| Inactive reference | CHEMBL50 | 1963774 | 40,000 nM |
| Query Q1 | CHEMBL392776 | 1963799 | Withheld |

The source extract labels IC₅₀ ≤ 100 nM as active and IC₅₀ ≥ 10,000 nM as inactive. Intermediate measurements fall outside those two groups. Each reference row contains the structure, class, the source's rounded pChEMBL value and the original measurement with its units. The last two CSV columns contain the human-readable measurement and link to its activity record for the report. [Reference input](reference.csv) · [Query input](query.csv) · [Assay definition](endpoint.txt)

ChEMBL structures and activity data come from EMBL-EBI under CC BY-SA 3.0. The example tables select, rename and format fields from that source and are shared under the same license. [ChEMBL licensing](https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing) · [License](https://creativecommons.org/licenses/by-sa/3.0/)

## Reproduce the public record

From the v0.5.1 repository root, install the Python 3.12 example environment and configure a model for new predictions as described in the tutorial. The saved run used an OpenAI-compatible Magpie Chat Completions service, the requested model group `group/deepseek-free`, temperature 0, a 16,384-token output limit and the default five judgments on a single panel. The service reported the group name on every response. The individual backend was undisclosed. All five responses completed with no truncation, transport retries or failed requests. A new live run can yield a different interpretation. [Live tutorial](../../docs/tutorial.md) · [Original decision and call records](predictions.json)

To inspect the exact saved result, run the following command from the same checkout.

```bash
molactagent molecular \
  --reference examples/cdk2/reference.csv \
  --test examples/cdk2/query.csv \
  --replay examples/cdk2/responses.json \
  --output-dir cdk2-replay
```

The command writes `cdk2-replay/report.html` and `cdk2-replay/predictions.json`, prints `Q1 Inconclusive` and exits with status 4. The software required three of the five judgments to supply a class. The recording includes the original file digests, request digests, model settings and replies. [Saved responses](responses.json)
