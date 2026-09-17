# BSLT — Brine-Shrimp Lethality regression example

A small (16-row) dataset of natural-product / drug-lead compounds
with a continuous bioactivity target. Used as the regression
counterpart to the iris (classification) and penguins (regression
on numeric morphometrics) examples — but here the input is SMILES
strings, which exercise the chemistry preprocessing pipeline.

## Files

- `bslt.csv` — 15 compounds with two columns:
  - `smiles` — RDKit-parseable SMILES strings.
  - `standard_value` — the regression target (bioactivity value;
    higher = more toxic in the brine-shrimp lethality assay).

## Use it

```python
import molactagent as ca

# One-call: SMILES → 26 RDKit descriptors + 167 MACCS keys + Murcko
# scaffolds, ready for the estimator.
features = ca.datasets.load_bslt_regression(
    descriptors=True, maccs=True, morgan=False,
)

reg = ca.MolActAgentRegressor(
    llm_config=ca.load_llm_config("llm_config.jsonc"),
    target_name="standard_value",
    feature_names=features.feature_names,
    feature_descriptions=features.feature_descriptions,
    background=ca.pp.chemistry.BSLT_BACKGROUND,
    k=3,
).fit(features.X[:11], features.y[:11])

predictions, lo, hi = reg.predict_with_interval(features.X[11:])
```

## Provenance

The compounds are a curated subset compiled by the project authors
for benchmarking the regression backend on a small-data,
chemistry-driven scenario. The target unit and exact origin of
each measurement are deliberately not surfaced in the example —
this is a *stress test for small-n LLM-driven regression*, not a
QSAR study. For a full QSAR benchmark, see `data/` in the project
repository (research artefacts, not part of the package
distribution).
