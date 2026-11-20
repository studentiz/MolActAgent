# Data sources and redistribution notes

## Measured CDK2 example

The four references and one query in the public example are adapted from
the ChEMBL 36 CDK2 extract. Its provenance table maps each row to a ChEMBL
activity record and the published assay. Derivative data use CC BY-SA 3.0.
[Case provenance](../examples/cdk2/README.md) · [ChEMBL licensing](https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing)

## PPAR-alpha cell-assay example

`examples/cell-assay/` is an exhaustive snapshot of the 11 activities in
ChEMBL 37 assay `CHEMBL763850`, an in vitro human PPAR-alpha transactivation
assay using a PPAR-GAL4 chimera in COS-1 cells. The target is `CHEMBL239` and
the source document is `CHEMBL1136170`, from Desai et al., *Bioorganic &
Medicinal Chemistry Letters* 13 (2003), 2795-2798. The source uses EC50 in
`nM` with `>` and `=` relations. The bundled structures passed a canonical
structure comparison with the corresponding ChEMBL records.

The release assigns all 11 source activities exactly once to seven references
and four queries. That 7+4 partition is encoded in the example CSVs and is a
retrospective release construction. ChEMBL does not document a historical
selection order for the four candidates. The example defines its binary
classes with a 1 micromolar EC50 cutoff.

The ChEMBL data are licensed under [CC BY-SA 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/).
The derived example tables preserve ChEMBL molecule and activity IDs and are
shared under the same license. Use this required attribution.

> ChEMBL data is from https://www.ebi.ac.uk/chembl; the version of ChEMBL is
> chembl_37.

Source links are [assay](https://www.ebi.ac.uk/chembl/explore/assay/CHEMBL763850)
· [target](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL239) ·
[publication](https://doi.org/10.1016/S0960-894X(03)00505-5) ·
[ChEMBL 37 DOI](https://doi.org/10.6019/CHEMBL.database.37) · [licensing statement](https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing)
· [case provenance](../examples/cell-assay/README.md)

## SMILES teaching example

`examples/molecular/` contains hand-authored SMILES for common molecules
and synthetic labels for the presence of a phenolic hydroxyl. The bundled
responses were recorded from live model requests on this teaching example.
The example files are distributed with the project under its MIT license.

## ChEMBL benchmark tables

The files under `data/benchmarks/` are raw evidence-table extracts from ChEMBL 36. Each dataset directory contains its target, query filters, row counts, and retrieval timestamp. ChEMBL is provided by the European Bioinformatics Institute; consult the [ChEMBL data documentation](https://www.ebi.ac.uk/chembl/) and current terms before redistributing a release.

The benchmark tables retain ChEMBL identifiers, activity fields, compound structures, and publication metadata. They are research inputs, not clinical or regulatory data, and should not be interpreted as validated drug-discovery results.

## BSLT and antiTB examples

The BSLT and antiTB CSVs are small example inputs inherited from the source project. Their original project-level provenance is recorded in the accompanying example documentation where available. Before a tagged public release, maintainers should add the upstream publication or dataset URL and confirm that redistribution is permitted.

## Data hygiene

Do not commit API keys, private assay records, unverified datasets, generated reports, or cached feature matrices. If a new dataset is added, document its source, version, retrieval date, transformations, license, and any restrictions in this file and its local README.
