# Experimental regression support

**The regression backend remains under development and is not evaluated
in the accompanying manuscript. We share it to support community
exploration and collaboration. Its reliability and applicability require
further validation.**

The manuscript focuses on molecular activity classification and its
traceable evidence. The regression code is an exploratory extension;
the manuscript's classification results do not establish its regression
performance. Its methods and examples are retained so others can inspect
the implementation, reproduce exploratory runs and help improve it.

## What remains available

- `MolActAgentRegressor`, the scikit-learn-style wrapper.
- `CrossClassificationRegressor` (CCR), the underlying backend.
- `run_regression` and the CLI's numeric-target workflow.
- The penguins and BSLT examples, regression report rendering and tests.

The existing interfaces remain available. Their presence in the package
does not imply the same level of evaluation as the manuscript's
classification workflow.

## How to interpret the current outputs

The backend combines reference target values selected through two
classification views. Its behavior still needs broader evaluation across
datasets, target distributions and model configurations.

- Repository example results are exploratory observations, not evidence
  of general superiority or a validated molecular regression method.
- Reported intervals describe selected reference targets; they are not
  statistically calibrated prediction intervals.
- Confidence scores and generated explanations require independent
  assessment. An explanation does not validate a numerical prediction.

For research use, compare with suitable conventional regressors and
report the data split, settings and failed or refused predictions.

## Explore and contribute

The [tutorial](tutorial.md#11-predicting-continuous-values-regression)
retains runnable examples. The [design notes](design.md#19-the-ccr-regression-backend)
describe the current implementation. These sections document the
experimental code; they do not present a regression contribution from
the manuscript.

We welcome reproducible evaluations, failure cases and method improvements
through [issues](https://github.com/studentiz/MolActAgent/issues) and
[pull requests](https://github.com/studentiz/MolActAgent/pulls). These
contributions can help establish where the approach is useful and which
parts need further development.
