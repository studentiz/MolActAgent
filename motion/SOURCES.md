# Sources, provenance, and limitations

## Scientific inputs

The checked-in release fixture supplies the assay definition, structures, reference values,
queries, held-out measurements, and public ChEMBL record links:

- examples/cell-assay/endpoint.txt
- examples/cell-assay/reference.csv
- examples/cell-assay/query.csv
- examples/cell-assay/outcomes.csv
- examples/cell-assay/predictions.json

SHA-256:

- endpoint: 8bf05f6a6c79478a193d03fd516d495f1f827d564d0307192114cd72ea94544f
- reference: 78f88c6938397dfae3871f2267d5c805b0e67d95ea464f189d15b1c2f8ebd61d
- query: 3e1e7e5b2372e21d62952aee6d1509de9f610bad459437ea50f0b434fc9c5e4c
- outcomes: 0106986e65431aee3d7e566debf91823c8a71c2ce066acff6c6f2471ebaee632
- predictions: d9b1bdc93353b815e02a1ea1f83fab8ae8ad5299c7eea21568b0f2b5a132f7be

The published v0.6.0 primary prediction record is checked in at
`examples/cell-assay/predictions.json`.

Its four calls, copied into data/evidence.json, are:

| Candidate | Predicted class | Recorded confidence |
| --- | --- | ---: |
| Q1 | inactive | 0.800 |
| Q2 | active | 0.856 |
| Q3 | inactive | 0.790 |
| Q4 | active | 0.754 |

The film does not display confidence values as if they were experimental uncertainty.

## Verified Q2 statements

`scripts/verify_evidence.py` cross-checks the checked-in summary against the release fixture
and reads `examples/cell-assay/predictions.json` by default. It checks the prediction
record's SHA-256, all four labels, the exact recorded Q2 judgment, and the Q2 comparator
classes. `SOURCE_PREDICTIONS` can point to another copy explicitly.

- Q2 structure: CCCc1cc(Oc2ccc(F)cc2)ccc1OCCCOc1cccc(C2SC(=O)NC2=O)c1
- R2 structure: same scaffold with para-Cl; measured 180 nM, active.
- R4 structure: same scaffold with para-Cl plus meta-Me; measured 156 nM, active.
- R5 structure: same scaffold with para-OH; measured >3000 nM, inactive.
- R7 structure: para-F plus meta-methyl; measured 85 nM, active.
- Q2 recorded prediction: active.
- Q2 held-out outcome in the source fixture: 66 nM, active. The film does not display the
  held-out outcomes and ends on the illustrative prospective assay decision.

For Q4, the matched measured comparator is R4 (`para-Cl` + `meta-Me`, 156 nM active). Q4
keeps `para-Cl` and changes `meta-Me` to `meta-Cl`. Q4 is untested in the displayed decision
sequence and predicted active by MolActAgent.

The important claim is deliberately narrow: para-F is measured in R7 alongside a meta-methyl;
Q2 tests the F-only analogue without that neighboring methyl. The comparison at the para
position is measured, while the exact Q2 analogue is predicted until the held-out outcome
is opened.

The recorded Q2 language-model statement checked by the script is:

> On the shared scaffold, a single para halogen on the terminal phenoxy ring distinguishes
> the active analogs from the inactive para-hydroxy and para-alkyl analogs.

## ChEMBL records

The fixture carries these ChEMBL reference activity URLs and query activity URLs:

- R1: https://www.ebi.ac.uk/chembl/explore/activity/108761
- R2: https://www.ebi.ac.uk/chembl/explore/activity/108771
- R3: https://www.ebi.ac.uk/chembl/explore/activity/108783
- R4: https://www.ebi.ac.uk/chembl/explore/activity/116833
- R5: https://www.ebi.ac.uk/chembl/explore/activity/121673
- R6: https://www.ebi.ac.uk/chembl/explore/activity/122905
- R7: https://www.ebi.ac.uk/chembl/explore/activity/125484
- Q1: https://www.ebi.ac.uk/chembl/explore/activity/122911
- Q2: https://www.ebi.ac.uk/chembl/explore/activity/129314
- Q3: https://www.ebi.ac.uk/chembl/explore/activity/133208
- Q4: https://www.ebi.ac.uk/chembl/explore/activity/136815

During this build, the public Explore URLs above returned HTTP 404. The rendered science
therefore verifies against the supplied release fixture rather than a newly downloaded
ChEMBL record. No replacement API query, model call, or new prediction was made.

ChEMBL states that its database data is provided under the Creative Commons
Attribution-ShareAlike 3.0 Unported license:

- https://www.ebi.ac.uk/chembl/
- https://creativecommons.org/licenses/by-sa/3.0/

The ChEMBL-derived scientific figure and measured-record selection should retain attribution
and share-alike terms.

Official ChEMBL 37 citation:

> Zdrazil B, Felix E, Hunter F, Manners EJ, Blackshaw J, Corbett S, de Veij M, Ioannidis H,
> Lopez DM, Mosquera JF, Magarinos MP, Bosc N, Arcila R, Kizilören T, Gaulton A, Bento AP,
> Adasme MF, Monecke P, Landrum GA, Leach AR. The ChEMBL Database in 2023: a drug discovery
> platform spanning multiple bioactivity data types and time periods. Nucleic Acids Res.
> 2024;52(D1):D1180-D1192. doi:10.1093/nar/gkad1004.

## Software and fonts

- RDKit 2025.09.2, BSD 3-Clause: https://www.rdkit.org/
- Remotion 4.0.545: https://www.remotion.dev/
- React 19.3.1, MIT: https://react.dev/
- DM Sans, SIL Open Font License 1.1. Local license: licenses/dm-sans-license.txt.
- TypeScript 5.9.3, Apache-2.0: https://www.typescriptlang.org/

Exact JavaScript dependency versions are pinned in package.json and package-lock.json.

## Limitations

- The assay is one PPAR-alpha COS-1 reporter endpoint with a fixed 1 µM cutoff.
- The four-candidate held-out return is a bounded retrospective, not a broad accuracy claim.
  It remains in the result files and report but is not shown as a separate film scene.
- The final Q2/Q4 next-assay list is an explicit decision illustration, not a claim about a
  historical laboratory scheduling event.
- Structures are RDKit 2D depictions from the supplied neutral SMILES. Protonation, tautomer,
  experimental conformer, salt form, and drawing-dependent effects are not inferred.
- The public ChEMBL activity pages were unavailable at build time; provenance remains the
  supplied release fixture and its recorded identifiers.
- The film is silent and noninteractive by design. It contains no microscopy image, fitted
  response curve, or synthetic experimental trace.
