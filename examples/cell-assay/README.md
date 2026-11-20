# PPAR-alpha cell-assay example

This retrospective example asks MolActAgent to classify four small molecules
at a 1 micromolar EC50 cutoff. The candidates come from the same functional
human PPAR-alpha assay as the seven reference molecules. All 11 structures,
molecule IDs, activity IDs, EC50 relations and values, `nM` units, and
source-derived classes matched ChEMBL 37 in the 2026-11-20 provenance audit.

## Source assay and selection

The source is ChEMBL assay [CHEMBL763850](https://www.ebi.ac.uk/chembl/explore/assay/CHEMBL763850),
an in vitro transactivation assay for human PPAR-alpha using a PPAR-GAL4
chimera in COS-1 cells. The target is
[CHEMBL239](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL239). The
measurements trace to document `CHEMBL1136170` and Desai et al.,
"5-aryl thiazolidine-2,4-diones: discovery of PPAR dual alpha/gamma agonists
as antidiabetic agents," *Bioorganic & Medicinal Chemistry Letters* 13
(2003), 2795-2798. [Publication](https://doi.org/10.1016/S0960-894X(03)00505-5)
· [PubMed record](https://pubmed.ncbi.nlm.nih.gov/12873517/) ·
[ChEMBL 37 release](https://doi.org/10.6019/CHEMBL.database.37)

The ChEMBL assay contains exactly 11 activity records. This repository uses
each record once, with seven rows in `reference.csv` and four in `query.csv`.
The 7+4 role partition appears in these CSVs as a release construction. No
source record establishes whether the four candidates reflect a historical
selection order or prioritization, so this example makes no such claim.

| Example ID | Role | ChEMBL molecule | ChEMBL activity | Source EC50 | Example class |
| --- | --- | --- | ---: | --- | --- |
| R1 | Reference | `CHEMBL94496` | 108761 | > 3000 nM | inactive |
| R2 | Reference | `CHEMBL420441` | 108771 | = 180 nM | active |
| R3 | Reference | `CHEMBL121` | 108783 | > 3000 nM | inactive |
| R4 | Reference | `CHEMBL328615` | 116833 | = 156 nM | active |
| R5 | Reference | `CHEMBL94397` | 121673 | > 3000 nM | inactive |
| R6 | Reference | `CHEMBL97529` | 122905 | > 3000 nM | inactive |
| R7 | Reference | `CHEMBL96143` | 125484 | = 85 nM | active |
| Q1 | Query | `CHEMBL96758` | 122911 | > 3000 nM | inactive |
| Q2 | Query | `CHEMBL96451` | 129314 | = 66 nM | active |
| Q3 | Query | `CHEMBL92130` | 133208 | > 3000 nM | inactive |
| Q4 | Query | `CHEMBL318684` | 136815 | = 82 nM | active |

The example transformation defines active as EC50 at or below 1 micromolar
and inactive as EC50 above 1 micromolar. The source reports EC50 in
`nM`. A result reported as `>3000 nM` exceeds the cutoff. The tables
retain that lower bound on EC50.

## Recorded outcomes

The agent received candidate IDs and structures in `query.csv`.
The published candidate EC50 measurements stayed in `outcomes.csv` until
the predictions had been saved. In this example the next assay list would
contain Q2 and Q4.

| Candidate | Saved prediction | Held-back EC50 | Class from measurement |
| --- | --- | --- | --- |
| Q1 | inactive | >3000 nM | inactive |
| Q2 | active | 66 nM | active |
| Q3 | inactive | >3000 nM | inactive |
| Q4 | active | 82 nM | active |

## Run diagnostics

- The archived run used MolActAgent 0.6.0, response-recording schema 2,
  Python 3.12.12, RDKit 2025.09.6, and pandas 2.3.3.
- The requested model route was `group/deepseek-free`. Its individual backend
  was undisclosed. The run used temperature 0, response-format mode, an
  output limit of 16,384 tokens, four workers, eight neighbors, and five panel
  judgments per query with three required for a class decision.
- All 20 calls were accepted with no errors, truncation, or transport retries.
  Each query has one request digest across its five calls, and all five
  judgments agree on the predicted class.
- Offline replay reproduced the archived `predictions` array exactly. The
  only run-metadata difference was `mode` changing from `live` to `replay`.

The archived primary recordings were verified and copied byte for byte.

| File | SHA-256 |
| --- | --- |
| `reference.csv` | `78f88c6938397dfae3871f2267d5c805b0e67d95ea464f189d15b1c2f8ebd61d` |
| `query.csv` | `3e1e7e5b2372e21d62952aee6d1509de9f610bad459437ea50f0b434fc9c5e4c` |
| `endpoint.txt` | `8bf05f6a6c79478a193d03fd516d495f1f827d564d0307192114cd72ea94544f` |
| `outcomes.csv` | `0106986e65431aee3d7e566debf91823c8a71c2ce066acff6c6f2471ebaee632` |
| `predictions.json` | `d9b1bdc93353b815e02a1ea1f83fab8ae8ad5299c7eea21568b0f2b5a132f7be` |
| `responses.json` | `15036bedf9fd7ef26fa89d0e554e11bcc7a1d23cdc8a1db301498e4537d9d80a` |

## Run a new prediction

Configure your Chat Completions service from the release checkout. The archived run used Magpie's `group/deepseek-free` route. If you have access to that route, select it in your local settings file. The command below specifies the comparison and request settings used in that run. With your own provider or model it performs a new assessment of the same four structures. [API configuration](../../README.md#configuration)

```bash
molactagent molecular \
  --reference examples/cell-assay/reference.csv --test examples/cell-assay/query.csv \
  --id-column id --endpoint @examples/cell-assay/endpoint.txt \
  --molecular-profile panel --neighbors 8 --repeats 5 --workers 4 \
  --temperature 0 --max-tokens 16384 --timeout 120 --max-retries 2 \
  --llm-config llm_config.jsonc --record my-cell-responses.json --output-dir my-cell-run
```

Open `my-cell-run/report.html` to review the four decisions. An additional live release check with the same route and settings produced the same four classes. Individual explanations can vary between calls. [Published batch report](https://studentiz.github.io/MolActAgent/cell-assay.html)

## Reproduce the saved result

From a fresh clone of v0.6.0 with Python 3.12, install the example environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[chem]" -c examples/molecular/constraints.txt
molactagent --version
```

Then replay the recorded responses without making a live model request.

```bash
molactagent molecular \
  --reference examples/cell-assay/reference.csv \
  --test examples/cell-assay/query.csv \
  --endpoint @examples/cell-assay/endpoint.txt \
  --replay examples/cell-assay/responses.json \
  --output-dir cell-assay-replay
```

The command writes `cell-assay-replay/report.html` and
`cell-assay-replay/predictions.json` and exits with status 0.

## Data license and attribution

ChEMBL 37 data are distributed under the
[CC BY-SA 3.0 Unported license](https://creativecommons.org/licenses/by-sa/3.0/).
The example tables preserve the ChEMBL molecule and activity IDs and are
shared under the same license. Use this required attribution.

> ChEMBL data is from https://www.ebi.ac.uk/chembl; the version of ChEMBL is
> chembl_37.

See the [ChEMBL data licensing statement](https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing),
the [ChEMBL release DOI](https://doi.org/10.6019/CHEMBL.database.37), and the
[ChEMBL citation](https://doi.org/10.1093/nar/gky1075). The repository's
software is separately described in the root [`LICENSE`](../../LICENSE).
