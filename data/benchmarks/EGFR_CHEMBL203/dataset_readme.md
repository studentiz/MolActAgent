# Epidermal growth factor receptor (CHEMBL203) Raw Evidence Dataset

## Source

- Database: ChEMBL
- ChEMBL version: ChEMBL_36
- ChEMBL release date: 2025-07-28
- Target: CHEMBL203, Epidermal growth factor receptor
- Query filter: target_chembl_id=CHEMBL203; standard_type__in=IC50; standard_relation='='; assay_type__in=B; require_pchembl=True
- Download cap for this pilot: at most 500 unique molecules

## File

- EGFR_CHEMBL203_raw_evidence_table.csv

  One row equals one ChEMBL activity evidence record. The same molecule can appear more than once if it has multiple activity records. No aggregation or cleaning were applied beyond adding class_label.

  Rows: 258
  Columns: 35
  Unique molecules: 257

  Core activity fields: standard_type, standard_relation, standard_value, standard_units, pchembl_value.
  Class label: active if pchembl_value >= 7 ; inactive if pchembl_value =<5, else do not include.

  Active: 195

  Inactive: 62

  Retrieved UTC: 2026-04-11T03:55:05+00:00
