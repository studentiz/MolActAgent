# MolActAgent tutorial

- [Start with your own assay](#use-your-own-data)
- [Run the published cell assay](#run-the-published-cell-assay)
- [Understand an abstention in the CDK2 example](#optional-worked-example)
- [Classify numeric tables with the three-class Iris example](#iris-tutorial-for-numeric-tables)
- [Look up commands, configuration, and Python interfaces](tutorial-reference.md)

<a name="3-the-big-idea-in-three-sentences"></a>

## Predict from SMILES

Start with one decision your next set of experiments needs to inform. For example, you may have several new analogs and enough material to test only a few. Use compounds you have already measured under the same conditions to examine each candidate, identify gaps in the evidence, and decide which measurement would address one of those gaps.

<a name="bring-your-own-smiles"></a>

### Use your own data

Save the files below as plain text in the project folder after installing MolActAgent. A CSV is a table your spreadsheet can export. Each row represents one compound. SMILES is a text representation of its structure.

| File | What to put in it |
| --- | --- |
| `reference.csv` | Use the columns `id,smiles,label`. Give each measured reference a unique ID and label it `active` or `inactive` according to your assay. Include compounds from both classes. |
| `query.csv` | Use the columns `id,smiles` for the compounds you want to review. Give each candidate a unique ID. |
| `assay.txt` | State what you measured, the relevant assay conditions and units, and the rule you used to assign each label. Record both rules if active and inactive have separate cutoffs. |

Use isomeric SMILES when you know the stereochemistry. Add a `name` column if it helps you recognize compounds in the report. [Molecular input options](tutorial-reference.md#molecular-cli)

<a name="1-what-you-need-before-you-start"></a>
<a name="2-getting-molactagent-onto-your-computer"></a>
<a name="2-installing-molactagent"></a>

### Install the example environment

Install Git and Python 3.12. Open a terminal in the folder where you want to keep the project, then copy these commands. The tag supplies the example files and the version pins for the chemistry environment.

```bash
git clone --branch v0.6.0 https://github.com/studentiz/MolActAgent.git
cd MolActAgent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[chem]" -c examples/molecular/constraints.txt
molactagent --version
```

In Windows PowerShell, run the same clone and `cd` commands, then use `py -3.12 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip` in place of `python -m pip` and `.\.venv\Scripts\molactagent.exe` in place of `molactagent`. Paste each multiline command as one line, removing the trailing backslashes.

Run the examples from the `MolActAgent` folder. The package supports Python 3.10–3.12. For numeric tables alone, you can use `python -m pip install -e .` in an active environment. [Optional readers and development setup](tutorial-reference.md#development-and-tests)

<a name="5-setting-up-your-llm-your-ai-service"></a>

### Use a live model

The model reads the assay description and selected compound data. Use a service appropriate for your lab data that supports OpenAI-compatible Chat Completions. Create a local settings file.

```bash
molactagent init-config --path llm_config.jsonc
```

Open `llm_config.jsonc` in a text editor and fill in the `api_key`, API root `base_url`, and `model_name` from your service. Keep your API key private. [Credentials, profiles, and local services](tutorial-reference.md#model-and-api-configuration)

### Run the published cell assay

The seven measured references and four candidate structures are ready to use. In this cell reporter assay, EC₅₀ is the concentration needed for half the maximal response. A lower EC₅₀ indicates higher potency. The example labels a compound active at 1,000 nM or below. After configuring your live API, run the assessment.

```bash
molactagent molecular \
  --reference examples/cell-assay/reference.csv \
  --test examples/cell-assay/query.csv \
  --id-column id --endpoint @examples/cell-assay/endpoint.txt \
  --llm-config llm_config.jsonc \
  --record assay-responses.json --output-dir assay-output
```

Read the four decisions in `assay-output/report.html`. In the published run Q2 and Q4 were predicted active. Open Q2 and follow R2 and R5 to see the measured values and terminal substitutions underlying its call. The independent outcomes file is for checking the predictions afterward. [Published report](https://studentiz.github.io/MolActAgent/cell-assay.html) · [Complete case and individual outcomes](../examples/cell-assay)

### Run your own candidate batch

Set `--reference reference.csv --test query.csv --endpoint @assay.txt` with the files prepared above. Choose a new output directory and recording path for this run. If the service or computer stops midway, reissue that run's command with `--resume`. Keep the two CSVs, original settings, output directory and recording path. Each `compound-*.json` under `.molactagent` holds a completed decision. A compound interrupted before its checkpoint can be requested again. A fresh run goes into a new destination.

### Read a molecular prediction

1. Open `assay-output/report.html` and check the assay definition and the predicted class for a candidate. An `Inconclusive` decision means you should read the reason before deciding what to test.
2. Select a cited reference and check its structure, recorded label, and any assay measurement. Look at the difference the model cites and whether your reference set contains a measurement that addresses it.
3. Open the individual judgments to see where the comparisons disagree. Use scores as evidence within the run, and use your lab measurements to assess reliability.

The CLI writes `predictions.json` with the structured decision, comparisons and call records. The response recording saves the model replies and settings for replay. Keep it with the input tables. [Report guide](reports.md#reports-from-smiles)

<a id="optional-worked-example"></a>

### Understand an abstention in the CDK2 example

This second assay illustrates a candidate for which the agent returned an inconclusive call. It has four references from human CDK2 inhibition assay `CHEMBL897015` and one query, `Q1`, an alias for `CHEMBL392776`. The assay file defines active as IC₅₀ at or below 100 nM (pChEMBL ≥ 7) and inactive as IC₅₀ at or above 10,000 nM (pChEMBL ≤ 5). The query CSV supplies its ID and SMILES. [ChEMBL assay](https://www.ebi.ac.uk/chembl/explore/assay/CHEMBL897015) · [Query compound](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL392776) · [Example files](../examples/cdk2)

```bash
molactagent molecular \
  --reference examples/cdk2/reference.csv \
  --test examples/cdk2/query.csv \
  --id-column id \
  --pchembl-column pchembl \
  --endpoint @examples/cdk2/endpoint.txt \
  --llm-config llm_config.jsonc \
  --temperature 0 --max-tokens 16384 \
  --record cdk2-responses.json \
  --output-dir cdk2-output
```

The release recording returned `Inconclusive`. Four judgments explicitly chose `Uncertain` and one voted `active`, with zero failures. The engine needed at least three usable class votes out of five. Exit code `4` signals an inconclusive prediction after the CLI has written the full HTML and JSON. A fresh live run may return different judgments. [CDK2 report](https://studentiz.github.io/MolActAgent/cdk2-report.html)

The close active reference `CHEMBL232557` measured 87 nM, near the 100 nM cutoff. At the same site in `Q1`, a bromine takes the place of the terminal hydroxyl group. The nearest inactive reference, `CHEMBL50`, measured 40,000 nM. All four references are halogen-free, so the effect of the OH-to-Br substitution remains a question for measurement.

### Replay a complete prediction

Use the saved responses to reconstruct the four cell-assay decisions. Open `cell-assay-replay/report.html` and follow Q2 to R2 and R5 to read the same comparison shown on the project site.

```bash
molactagent molecular \
  --reference examples/cell-assay/reference.csv \
  --test examples/cell-assay/query.csv \
  --replay examples/cell-assay/responses.json \
  --output-dir cell-assay-replay
```

To replay your own work, reuse the same CSVs and pass the response file you saved with `--record`. The recording supplies the assay description, column names, and settings. The CLI checks the input files, RDKit version, and rebuilt model requests. Choose a fresh output path for another copy of the report.

The historical phenol example uses synthetic labels to illustrate a structure comparison. v0.6.0 accepts its v0.5.0 schema 2 recording in the pinned environment.

```bash
molactagent molecular \
  --reference examples/molecular/reference.csv \
  --test examples/molecular/query.csv \
  --replay examples/molecular/responses.json \
  --output-dir molecular-output
```

Open `molecular-output/report.html` to compare 4-fluorophenol with p-cresol and benzyl alcohol. For a schema 1 recording made with v0.4.1, use that release, its dependencies, and the original command options. [Historical teaching report](https://studentiz.github.io/MolActAgent/molecular-example.html)

### Use the Python interface

Use `MolecularClassifier` to access the reference comparisons and judgments in a notebook or script. [Molecular Python example](tutorial-reference.md#molecular-classifier)

## Iris tutorial for numeric tables

<a name="4-a-tour-of-the-iris-flower-example"></a>

Use the feature-table path when you have numeric measurements or descriptors. In this example, `reference.csv` contains 24 flowers labeled as `setosa`, `versicolor`, or `virginica`. The six test flowers have the same four sepal and petal measurements without species labels. The JSON descriptions explain each measurement and the label. The background file gives the model context for the comparison. [Iris example files](../examples/iris_flowers)

<a name="6-step-1--validate-the-data"></a>

### Validate the inputs

Run validation before paying for a prediction. The check reads the tables and reports problems with IDs, labels, features, and descriptions without calling a model.

```bash
molactagent validate \
  --reference examples/iris_flowers/reference.csv \
  --test examples/iris_flowers/test.csv \
  --label-column species --id-column flower_id --k 3 \
  --feature-descriptions examples/iris_flowers/feature_descriptions.json \
  --background @examples/iris_flowers/background.txt
```

<a name="7-step-2--run-the-classification-default-mode"></a>

### Run the classification

Reuse your `llm_config.jsonc` from the live example, or create and fill one in with `molactagent init-config`. This command writes a batch report and the predicted species.

```bash
molactagent run \
  --reference examples/iris_flowers/reference.csv \
  --test examples/iris_flowers/test.csv \
  --label-column species --id-column flower_id --task classification --k 3 \
  --feature-descriptions examples/iris_flowers/feature_descriptions.json \
  --background @examples/iris_flowers/background.txt \
  --llm-config llm_config.jsonc \
  --output-dir iris-output
```

<a name="8-reading-the-default-outputs"></a>

### Read the batch outputs

Open `iris-output/summary.csv` for the labels and `iris-output/interpretability.html` for the vote and feature evidence. `interpretability.md` gives a text version. The labels are predictions for the six queries. A run can emit `Inconclusive` if there are no usable comparisons or the votes are too close. [Output formats and fields](tutorial-reference.md#numeric-table-outputs)

<a name="9-adding-per-sample-reports"></a>

### Adding per-sample reports

Add `--per-sample` to save HTML, Markdown, and JSON for each query. HTML or Markdown requests a separate narrative from the model. For structured evidence only, use `--per-sample --per-sample-format json`. To open a saved classification record in the named-compound viewer, add the report context described in the report guide. [Export a saved record](reports.md#export-your-own-saved-classification)

<a name="10-how-molactagent-handles-multi-class"></a>

### How multi-class predictions work

For each query, the numeric engine selects nearby candidate classes. In the Iris example it may compare two species or all three. Each comparison uses one reference row from each candidate class and the query's feature values. The engine checks the model's response against numerical evidence and combines the weighted votes. Inspect the candidate classes and vote evidence when judging a prediction. [Candidate selection, cost, and scores](tutorial-reference.md#how-numeric-classification-works)

<a name="16-using-molactagent-from-python"></a>
<a name="161--sklearn-style-estimators-molactagentclassifier-molactagentregressor"></a>

### Classify a numeric table in Python

The bundled loader prepares arrays with consistent feature columns. Pass feature names, descriptions, and background from the loader to preserve that context in the comparisons.

```python
import molactagent as ma

iris = ma.datasets.load_iris_flowers()
config = ma.load_llm_config("llm_config.jsonc")
classifier = ma.MolActAgentClassifier(
    llm_config=config,
    feature_names=iris.feature_names,
    feature_descriptions=iris.feature_descriptions,
    background=iris.background,
    k=3,
)
classifier.fit(iris.X_train, iris.y_train)
print(classifier.predict(iris.X_test))
print(classifier.classes_)
print(classifier.predict_proba(iris.X_test))

for sample_id, result in zip(iris.test_index, classifier.predict_full(iris.X_test)):
    print(sample_id, result.label, result.confidence, result.candidate_classes)
```

`predict_proba` returns weighted-vote shares in `classes_` order. Read them as evidence scores. The estimator reuses its most recent predictions when called again on the same batch. For reports and file-oriented processing from Python, use `run_classification`. [Estimator and pipeline examples](tutorial-reference.md#python-interfaces)

## Reference and troubleshooting

- <a name="13-bringing-your-own-data"></a>Prepare your own feature tables, descriptions, and labels. [Numeric table inputs](tutorial-reference.md#numeric-table-inputs)
- <a name="14-the-full-cli-reference"></a>Look up commands and flags. [Command-line reference](tutorial-reference.md#command-line-reference)
- <a name="12-pipe-friendly-mode-using-molactagent-inside-other-tools"></a>Read a table from standard input and send structured results to another tool. [Using the CLI in scripts](tutorial-reference.md#using-the-cli-in-scripts)
- <a name="162--compound-chemistry-smiles--molactagent-in-one-call"></a>Build numeric features from SMILES for the feature-table classifier. [SMILES to numeric features](tutorial-reference.md#smiles-to-numeric-features)
- <a name="163--downstream-analysis-loocv-baselines-calibration-attribution"></a>Inspect batches, calibration, or results on your own measured data. [Downstream analysis](tutorial-reference.md#downstream-analysis)
- <a name="15-common-error-messages"></a><a name="17-faq"></a>Resolve input, model, and replay errors. [Troubleshooting](tutorial-reference.md#troubleshooting)

<a name="11-predicting-continuous-values-regression"></a>

### Experimental regression example

The experimental numeric-table backend can predict continuous values. The Penguins example uses bill and flipper measurements to estimate body mass. Remove the text `species` column from the CSVs before using the CLI.

```python
from pathlib import Path

import pandas as pd

source = Path("examples/penguins")
target = Path("penguins-input")
target.mkdir(exist_ok=True)
for filename in ("reference.csv", "test.csv"):
    table = pd.read_csv(source / filename)
    table.drop(columns="species").to_csv(target / filename, index=False)
```

Validate the resulting files. To run a prediction, use the same arguments with `molactagent run` and add `--task regression --llm-config llm_config.jsonc --output-dir penguins-output`.

```bash
molactagent validate \
  --reference penguins-input/reference.csv \
  --test penguins-input/test.csv \
  --target-col body_mass_g --id-column sample_id \
  --feature-descriptions examples/penguins/feature_descriptions.json \
  --background @examples/penguins/background.txt
```

Inspect the estimated value and evidence range when present. A refused prediction means the engine withheld an estimate. In Python, `ma.datasets.load_penguins()` prepares the tables for you. [Regression API and outputs](tutorial-reference.md#experimental-regression) · [Current regression scope](experimental-regression.md)
