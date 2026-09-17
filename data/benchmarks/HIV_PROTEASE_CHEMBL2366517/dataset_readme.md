# HIV-1 protease (CHEMBL2366517) Raw Evidence Dataset

## Source

- Database: ChEMBL
- ChEMBL version: ChEMBL_36
- ChEMBL release date: 2025-07-28
- Target: CHEMBL2366517, HIV-1 protease
- Query filter: target_chembl_id=CHEMBL2366517; standard_type__in=None; standard_relation=None; assay_type__in=None; require_pchembl=False
- Download cap for this pilot: at most 500 unique molecules

## File

- HIV_PROTEASE_CHEMBL2366517_raw_evidence_table.csv

  One row equals one ChEMBL activity evidence record. The same molecule can appear more than once if it has multiple activity records. No aggregation or cleaning were applied beyond adding class_label.

  Rows: 287
  Columns: 35
  Unique molecules: 286

  Core activity fields: standard_type, standard_relation, standard_value, standard_units, pchembl_value.
  Class label: active if pchembl_value >= 8 else inactive.
  Retrieved UTC: 2026-04-11T03:53:03+00:00

  Active: 111
  Inactive: 175
