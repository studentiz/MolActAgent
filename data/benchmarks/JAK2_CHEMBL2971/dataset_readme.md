# JAK2 CHEMBL2971 Raw Evidence Dataset

## Source

- Database: ChEMBL
- ChEMBL version: ChEMBL_36
- ChEMBL release date: 2025-07-28
- Target: CHEMBL2971, Tyrosine-protein kinase JAK2, Homo sapiens
- Query filter: target_chembl_id=CHEMBL2971; standard_type=IC50; standard_relation="="; pchembl_value not null; assay_type=B
- Download cap for this pilot: at most 500 unique molecules

## File

- JAK2_CHEMBL2971_raw_evidence_table.csv

One row equals one ChEMBL activity evidence record. The same molecule can appear more than once if it has multiple activity records. No activity labels, aggregation, balancing, SELFIES conversion, or model-oriented cleaning were applied.

Rows: 318
Columns: 34
Unique molecules: 317

Core activity fields: standard_type, standard_relation, standard_value, standard_units, pchembl_value.
Retrieved UTC: 2026-04-10T10:40:45+00:00

Class label: active if pchembl_value >= 7 ; inactive if pchembl_value =<5, else do not include
- Active: 305
- Inactive: 12
