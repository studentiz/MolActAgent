# Acetylcholinesterase (CHEMBL220) Raw Evidence Dataset

## Source

- Database: ChEMBL
- ChEMBL version: ChEMBL_36
- ChEMBL release date: 2025-07-28
- Target: CHEMBL220, Acetylcholinesterase
- Query filter: target_chembl_id=CHEMBL220; standard_type__in=IC50; standard_relation='='; assay_type__in=B; require_pchembl=True
- Download cap for this pilot: at most 500 unique molecules

## File

- ACHE_CHEMBL220_raw_evidence_table.csv

  One row equals one ChEMBL activity evidence record. The same molecule can appear more than once if it has multiple activity records. No aggregation or cleaning were applied beyond adding class_label.

  Rows: 219
  Columns: 34
  Unique molecules: 218
  

Core activity fields: standard_type, standard_relation, standard_value, standard_units, pchembl_value.
  Class label: active if pchembl_value >= 7 ; inactive if pchembl_value =<5, else do not include.

Active: 143
  Inactive: 75

Retrieved UTC: 2026-04-11T03:38:50+00:00
