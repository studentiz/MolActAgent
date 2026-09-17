# scripts/

This directory holds **research, ablation, and benchmark scripts**
the MolActAgent authors use to validate the system, compare it
against baselines, and explore design alternatives. None of it is
part of the user-facing API and none of it is shipped in the
package wheel.

For end-user examples, look at:

- `examples/iris_flowers/` — multi-class classification.
- `examples/penguins/` — regression on numeric morphometrics.
- `examples/bslt/` — chemistry regression (SMILES → descriptors → CCR).
- `docs/tutorial.md` — the canonical step-by-step tutorial.

## What's here, in rough categories

### Data preparation (occasionally useful)

| Script | Purpose |
|---|---|
| `prepare_features.py` | Convert a SMILES + label CSV into a descriptors + MACCS feature matrix on disk. (The package now exposes the same logic as `ma.pp.smiles_to_features`; this script remains for batch/CI workflows.) |
| `prepare_penguins_subsample.py` | Build the bundled `examples/penguins/` reference + test split from the CC0 Palmer Penguins CSV. |
| `split_cox2.py` | Scaffold-aware split for the COX-2 dataset. |

### Evaluation runners

| Script | Purpose |
|---|---|
| `run_loocv_metrics.py` | The general LOOCV evaluator (classification + regression) used to produce the project's benchmark numbers. The new `ma.tl.loocv` and `ma.tl.compare_baselines` cover the same ground for in-process use; this script remains useful for batch jobs that produce JSON / TXT artefacts in `data/`. |
| `run_loocv_evidence.py` | LOOCV with the per-tuple evidence dump (heavy output). |
| `run_loocv_BSLT_regression.py` | BSLT-specific regression LOOCV. |
| `run_ablation_antiTB.py` | Ablation study on the antiTB classification dataset. |
| `aggregate_loocv_results.py` | Merge multiple LOOCV JSON outputs into one consolidated report. |

### Diagnostics / one-off probes

| Script | Purpose |
|---|---|
| `ccr_diagnose_and_recompute.py` | Probe CCR's per-phase behaviour on a fitted run. |
| `test_explanation.py` | Smoke-test the explanation prompt against a real LLM. |
| `run_demo.py` | Legacy end-to-end demo (kept for back-compat). |

### TabPFN / external-baseline comparisons

| Script | Purpose |
|---|---|
| `run_tabpfn_BSLT_loocv.py`, `run_tabpfn_penguins_loocv.py` | Compare TabPFN against MolActAgent on the project's benchmarks. |
| `test_tabpfn_memorization_penguins.py`, `test_tabpfn_prior_adaptation_penguins.py`, `test_prior_adaptation_full.py` | Targeted experiments examining whether TabPFN's small-data wins reflect genuine in-context learning vs implicit memorisation of common benchmarks. |

## Running these scripts

Each script declares its own CLI flags; run with `--help`. They all
expect:

- `molactagent` installed editably (`pip install -e .`) so they can
  `from molactagent import ...`.
- An `LLM_API.txt` or `llm_config.jsonc` at the project root if
  they hit the LLM (most of them do; ML-baseline-only scripts
  don't).

Outputs land in `data/` by convention. Those output files are
gitignored — they are too large and too volatile to keep in version
control. The source CSVs they read (`data/BSLT_huigui.csv`,
`data/antiTB_evidence_table.csv`, etc.) are the durable inputs.
