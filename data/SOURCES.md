# Data sources and redistribution notes

## ChEMBL benchmark tables

The files under `data/benchmarks/` are raw evidence-table extracts from ChEMBL 36. Each dataset directory contains its target, query filters, row counts, and retrieval timestamp. ChEMBL is provided by the European Bioinformatics Institute; consult the [ChEMBL data documentation](https://www.ebi.ac.uk/chembl/) and current terms before redistributing a release.

The benchmark tables retain ChEMBL identifiers, activity fields, compound structures, and publication metadata. They are research inputs, not clinical or regulatory data, and should not be interpreted as validated drug-discovery results.

## BSLT and antiTB examples

The BSLT and antiTB CSVs are small example inputs inherited from the source project. Their original project-level provenance is recorded in the accompanying example documentation where available. Before a tagged public release, maintainers should add the upstream publication or dataset URL and confirm that redistribution is permitted.

## Data hygiene

Do not commit API keys, private assay records, unverified datasets, generated reports, or cached feature matrices. If a new dataset is added, document its source, version, retrieval date, transformations, license, and any restrictions in this file and its local README.
