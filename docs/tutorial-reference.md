# MolActAgent tutorial reference

[Start with your own assay or follow the examples](tutorial.md)

- [Model and API configuration](#model-and-api-configuration)
- [Command-line reference](#command-line-reference)
- [Numeric table inputs and outputs](#numeric-table-inputs)
- [Using the CLI in scripts](#using-the-cli-in-scripts)
- [Python interfaces](#python-interfaces)
- [Troubleshooting](#troubleshooting)
- [Development and tests](#development-and-tests)

## Model and API configuration

A live run needs a key, a base URL, and a model name for an OpenAI-compatible Chat Completions service. Use the API root your provider specifies, including a version path such as `/v1` if required. For a locally hosted service, use its local API root and the model name served there.

Run `molactagent init-config --path llm_config.jsonc` to create a starter. Fill in the three fields in the active object. Here is the shape of a minimal file with placeholder values.

```jsonc
{
  "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
  "base_url": "https://provider.example/v1",
  "model_name": "your-model-name"
}
```

Keep API keys out of committed files, command history, and reports. Configuration resolves **per field** in this order, from highest precedence to lowest.

1. Explicit CLI options or keywords passed to `ma.load_llm_config`. The CLI uses `--api-key`, `--base-url`, and `--model` (alias `--model-name`).
2. Environment variables `MOLACTAGENT_API_KEY`, `MOLACTAGENT_BASE_URL`, `MOLACTAGENT_MODEL`, `MOLACTAGENT_TEMPERATURE`, and `MOLACTAGENT_TIMEOUT`.
3. The config file selected by `--llm-config PATH` or `ma.load_llm_config(PATH)`. By default, the loader searches the current folder and then its parents, looking in each for `llm_config.jsonc`, `llm_config.json`, `llm_config.yaml`, `llm_config.yml`, `LLM_API.txt`, and `llm_api.txt` in that order.

JSONC supports line and block comments and requires valid JSON after comments are removed. YAML needs the `yaml` extra. Legacy text files use key-value lines with a colon and can separate profiles with a blank line. To keep multiple profiles in JSON, JSONC, or YAML, use a list of objects or an object with a `profiles` list. The loader selects the last entry by default. Use `--profile first`, `--profile last`, or a substring of a model name to select an entry. The separate `--molecular-profile` flag changes the structure comparison.

| Settings file field | CLI option | Behavior |
| --- | --- | --- |
| `temperature` | `--temperature` | Sampling setting. The default is 0. |
| `max_tokens` | `--max-tokens` | Output budget. The client default for numeric runs is 2,000 and the molecular CLI defaults to 16,384 when the config omits the field. The generated starter sets 2,000 explicitly. |
| `timeout` | `--timeout` | Timeout per HTTP attempt in seconds. The default is 120. |
| `max_retries` | `--max-retries` | Additional HTTP attempts for transient errors. The default is 10. Set it to 0 for a single transport attempt. The molecular engine can still request one format correction after a malformed or truncated reply. |

The molecular CLI passes its resolved temperature and token limit to comparison calls. The numeric classification engine sets 2,000 tokens per tuple and 3,500 for the optional narrative. It passes its own temperature, which defaults to 0 and can be set in Python through `llm_temperature` on `MolActAgent` or `MolActAgentClassifier`. The standard numeric pipeline first probes the service with a short request.

## Command-line reference

Use the command from the active Python environment. `python -m molactagent` invokes the same entry point.

```bash
molactagent --version
molactagent --help
molactagent molecular --help
molactagent validate --help
molactagent run --help
molactagent init-config --help
```

Place global options before the subcommand. `--quiet` (`-q`) suppresses informational log lines. `--verbose` (`-v`) includes debug detail, and `--log-dir PATH` writes brief and full logs. Full logs can contain model responses. `init-config` takes `--path`, `--format jsonc|json|txt`, and `--overwrite` for replacing an existing settings file. Both JSON template options generate JSON with comments.

### Molecular CLI

| Option | What it controls |
| --- | --- |
| `--reference PATH`, `--test PATH` | Tables of labeled compounds and query structures. The command checks both files. |
| `--smiles-column`, `--label-column` | Input column names. Defaults are `smiles` and `label`. Reference labels must include `active` and `inactive`. |
| `--id-column`, `--name-column` | Custom ID or display-name columns present in both tables. With default settings, IDs are generated and a `name` column is used wherever present. The CLI preserves text such as `0007` and `NA` in molecular CSVs. |
| `--endpoint "@assay.txt"`, `--series TEXT` | Assay and class definitions, with optional series context. `--endpoint` also accepts literal text. |
| `--pchembl-column pchembl` | Measured numeric potency on the reference rows. The model uses it as reference evidence. Blank entries are allowed when labels are supplied independently. |
| `--pchembl-threshold VALUE`, `--labels-from-pchembl` | An optional single cutoff, with an optional validation rule that requires every reference to have a potency and expects `active` at or above the cutoff and `inactive` below. Supply the potency column and cutoff when enabling the rule. |
| `--molecular-profile panel\|focused` | The default panel compares both classes. The focused profile can add eligible halogen or stereochemical comparisons. |
| `--neighbors`, `--repeats`, `--workers` | Reference cap per class, planned main judgments per query, and parallel calls. Defaults are 8, 5, and 4 respectively. Fewer available references give smaller panels. |
| `--plain-json` | Omit the API's JSON-mode parameter while still asking for a JSON reply. |
| `--output-dir PATH` (`-o`) | Write `predictions.json` and `report.html`. Without this option, the JSON goes to standard output. |
| `--record PATH`, `--replay PATH` | Save responses and effective settings from a live run, or replay an existing recording without connecting to the provider. Choose one. |

Reference CSVs can also include `measurement` (original free-text value with units) and `source_url` (HTTP or HTTPS). These are display fields for people inspecting reference records in the HTML. The model receives assay text and its selected reference evidence, including values supplied through `--pchembl-column`.

The panel excludes exact canonical matches to the query and requires both classes to remain. It requires usable class judgments on more than half of the planned main comparisons. If the usable counts tie, the engine compares their model score totals. A tie after that or too few usable class votes can return `Inconclusive`. The focused profile can change the decision for eligible comparisons. [Molecular design](design.md#molecular-binary-classification) · [Output fields](reports.md#reports-from-smiles)

### Numeric CLI

`validate` checks tables without an API call. `run` performs validation and runs the model. Both require `--reference/-r`, `--test/-t`, `--id-column`, and `--label-column/-l` (also `--label-col`). For a continuous target, `--target-col` and `--target-column` are aliases of the label option.

- Set the reference cap per class with `--k` (default 3), and supply context through `--feature-descriptions PATH` or `--background @FILE`. Background can also be literal text.
- In `run`, set the distance metric with `--metric` (default `euclidean`), scaling with `--scaler` (`standard`, `minmax`, or `none`), and concurrent model calls with `--max-workers` (default 8).
- The validator infers the task from the reference labels. In `run`, an explicit `--task classification` or `--task regression` checks that the requested task matches detection. Read `summary.task` in `validate --format json` to check detection first.
- Use `--output-dir` to save reports. Choose the standard output format with `--format` or `-F`. See the output guide below. The legacy `--positive-class` option is ignored in classification.

### Exit status

| Command | Common exit codes |
| --- | --- |
| `molecular` | `0` for class decisions on all queries, `4` when one or more queries are inconclusive, `2` for caught input, configuration, replay, or output errors. A completed run with an inconclusive result still writes its report. |
| `validate` | `0` for valid input, `1` for validation errors, `2` for loading or parsing errors, `3` for two inputs from stdin. |
| `run` | `0` for a completed batch with no sample errors, `1` for failed validation, `2` for file errors, `3` for configuration or other pipeline setup errors, `4` when a sample failed. Check the verdict field for `Inconclusive` even with exit `0`. |

Argument parsing errors return `2`. A keyboard interrupt during the command returns `130`.

## Numeric table inputs

- Give each reference row a class in the label column. Use at least two distinct classes with at least one reference each. The query label column is optional and supplies known labels for output when present.
- Keep the ID column in both files with nonempty values unique across both sets. Treat it as row metadata. Keep the same numeric feature columns and units across both tables. Remove free-text metadata from the reference table or encode it as numeric features before validation.
- When you supply descriptions, cover every reference feature and the label column. Use JSON or YAML with column names as keys and short definitions as values. The ID column description is optional. Use the background file for method context and the meaning of the classes.
- CSV and TSV work with the base install. The loader also reads delimited TXT, JSON, JSONL, Excel, and Parquet. Use the `excel` extra for `.xlsx` or the `parquet` extra for Parquet. Older `.xls` files may need a separate Excel reader such as `xlrd`.

Text and boolean labels, or up to ten distinct integer-valued labels, are detected as classes. Numeric columns with fractional values or more than ten distinct values are generally detected as regression targets. For integer codes with many categories, use textual labels such as `class_12` in the CSV. String labels work for two classes or multiple classes.

### How numeric classification works

The engine looks for nearby references in each class. By default it retains two to four candidate classes per query when enough classes have usable neighbors. The selection compares each class's nearest distance with the nearest class overall, using a default 1.5× distance threshold and a minimum of two candidates.

Each tuple contains the query and one reference from each candidate class. The model evaluates the discriminating features. The engine checks the answer and can resolve a conflicting tuple label using the numerical comparison. With `M` candidate classes and up to `k` references per class, there are up to `k^M` tuples before exclusions and retries. Reduce `k` to make fewer comparisons when checking a new workflow.

The engine weights usable votes and chooses a class if the winner clears the margin over the runner-up. A tied or weak vote margin, or a lack of usable votes, yields `Inconclusive`. Inspect which classes were considered, which features favored each class, and which references were compared. The engine's confidence score incorporates voting margin, reference pool size, and comparison coverage. [Feature-table design](design.md#feature-table-engine)

## Numeric table outputs

With `--output-dir`, a normal `run` writes these batch files.

| File | Contents |
| --- | --- |
| `summary.csv` | One row per query with ID, verdict, score, elapsed time, error field, and any known label. |
| `interpretability.md`, `interpretability.html` | Batch review with vote summaries and the cited feature evidence. Reading this evidence costs no further model calls. |
| `summary.json` | Opt in with `--summary-json` for summary rows, validation summary and warnings, and LLM statistics. |

Add `--per-sample` for HTML, Markdown, and JSON per query. The HTML and Markdown use an extra model narrative. Use `--per-sample --per-sample-format json` for structured decision records without the narrative call. `--no-interpretability` skips batch explanations and `--no-summary-csv` skips the CSV. With no output directory, use standard output or the Python return object for results.

The included saved evidence example can be rendered offline with `python examples/evidence_report/render.py`. Open `examples/evidence_report/outputs/evidence_report.html`. It uses illustrative data. To use your own numeric classification record with the interactive viewer, populate `report_context` and pass the record to `ma.render_html_report`. [Report context schema](reports.md#export-your-own-saved-classification)

## Using the CLI in scripts

`validate` and `run` accept CSV from standard input when one of `--reference` or `--test` is `-`. Add `--stdin-format tsv` for TSV input.

```bash
cat examples/iris_flowers/test.csv | molactagent validate \
  --reference examples/iris_flowers/reference.csv --test - \
  --label-column species --id-column flower_id --format json
```

In a terminal the commands show a human-readable summary. With stdout redirected or piped, `validate` defaults to JSON and `run` to JSON Lines. Explicit `--format` wins. For `run`, choose JSON for one object containing validation and outcomes, JSONL for a record per query, or CSV/TSV for the verdict table. Logs and progress use stderr in these modes. In a script, check the process exit status and the individual outcomes.

This example runs the three-class Iris prediction and sends just the IDs and labels to `jq`. It needs the API configuration and `jq` installed.

```bash
molactagent --quiet run \
  --reference examples/iris_flowers/reference.csv \
  --test examples/iris_flowers/test.csv \
  --label-column species --id-column flower_id \
  --llm-config llm_config.jsonc --format jsonl \
  | jq -r '[.sample_id, .predicted_label] | @tsv'
```

## Python interfaces

Import the package as `import molactagent as ma`. Use a fitted classifier in a notebook when you want arrays or records. Use the file-oriented pipeline for validation and reports.

| Interface | What it returns |
| --- | --- |
| `molactagent.molecular.MolecularClassifier` | A `MolecularResult` per query with the class, compared references, judgments, and call records. |
| `ma.MolActAgentClassifier` | Labels, vote shares, confidence scores, or wrapped `ClassificationResult` records. |
| `ma.run_classification` or `ma.run_prediction` | `PipelineOutput` with validation, outcomes, report paths when requested, and LLM statistics. `run_prediction` detects the task by default. |
| `ma.MolActAgent` | The lower-level numeric engine. Build it with an LLM client and feature context, then `fit(X, y, task="classification")`. Its `predict(X)` returns dictionaries. Use `ma.wrap_result` for a typed wrapper. |

### Configure the Python client

Pass `ma.load_llm_config("llm_config.jsonc")` to `ma.build_llm_client` or to an estimator as `llm_config`. You can also construct an `ma.LLMClient` or use `ma.build_llm_client(api_key=..., base_url=..., model_name=...)` with credentials supplied by your program.

The numeric estimators resolve the service at `fit` in this order.

1. An existing `llm_client` wins and carries its own settings.
2. A resolved `llm_config` supplies credentials. Explicit estimator overrides for temperature, token budget, timeout, and retries are passed as client settings.
3. All three inline credentials (`api_key`, `base_url`, `model_name`) supply the complete configuration directly.
4. Partial inline credentials override the corresponding fields from the environment and discovered config file. With no inline credentials, the estimator loads the environment and discovered file.

The numeric pipeline takes a resolved config or a prebuilt client. It probes the service by default when the client implements `ping`. Use `verify_llm=False` to skip the probe. Prediction still uses the configured client.

### Molecular classifier

The same API accepts DataFrames. This example uses the included files as inputs. Provide your own files and assay text the same way.

```python
from pathlib import Path

import pandas as pd

import molactagent as ma
from molactagent.molecular import MolecularClassifier

case = Path("examples/cdk2")
reference = pd.read_csv(case / "reference.csv", dtype=str, keep_default_na=False)
query = pd.read_csv(case / "query.csv", dtype=str, keep_default_na=False)
config = ma.load_llm_config("llm_config.jsonc", temperature=0, max_tokens=16384)
classifier = MolecularClassifier(
    ma.build_llm_client(config),
    endpoint=(case / "endpoint.txt").read_text(encoding="utf-8").strip(),
    temperature=config.temperature,
    max_tokens=config.max_tokens,
)
classifier.fit(reference, id_column="id", pchembl_column="pchembl")
for result in classifier.predict(query, id_column="id"):
    print(result.identifier, result.label, result.decision)
```

Pass `response_format=False` for providers that need plain JSON text. `predict(..., record=True)` fills the classifier's `recording` with responses and settings. The CLI adds file fingerprints and column choices to its recordings and produces HTML with the reference annotations.

### Numeric pipeline and estimators

```python
import molactagent as ma

output = ma.run_classification(
    reference="examples/iris_flowers/reference.csv",
    test="examples/iris_flowers/test.csv",
    label_column="species",
    id_column="flower_id",
    feature_descriptions="examples/iris_flowers/feature_descriptions.json",
    background="@examples/iris_flowers/background.txt",
    llm_config=ma.load_llm_config("llm_config.jsonc"),
    output_dir="iris-python-output",
)
print(output.validation.format_report())
for outcome in output.outcomes:
    print(outcome.sample_id, outcome.predicted_label, outcome.error)
```

The pipeline accepts a path or a pandas DataFrame for each table. `outcome.raw_result` exposes the underlying decision record. To do your own loading, use `ma.load_table`, `ma.validate_inputs`, and `ma.prepare_for_agent`. `ma.pp.load_dataset` combines these steps and returns a prepared bundle like `ma.datasets.load_iris_flowers()`. The example dataset loaders use files from the source checkout. [Estimator walkthrough](tutorial.md#classify-a-numeric-table-in-python)

With `MolActAgentClassifier`, pass a two-dimensional numeric matrix to `fit` and `predict`, keeping feature order consistent. If `fit` receives a DataFrame it can infer feature names, while prediction uses column order. `predict_full` adds `candidate_classes`, `vote_summary`, `reference_info`, `driver_attribution`, `tuple_results`, and `to_dict()` for each query. The estimator also exposes `driver_attribution_` for the last batch and `llm_stats_` for the underlying client.

`predict_with_confidence` returns labels and engine confidence scores. `predict_proba` returns normalized weighted-vote shares in `classes_` order. Classes outside the candidate set get zero, and a candidate set with no decisive votes gets uniform shares. Calibrating probabilities requires separate data. `score` measures mean accuracy on labeled cases. The estimator accepts `sample_weight` for compatibility and fits all references with equal weight.

### SMILES to numeric features

Use the chemistry extra if you want descriptors or fingerprints for the numeric engine. Apply the same setup to both tables. With `drop_invariant=False`, the same fingerprint columns are kept even when rows in one file share a bit value. Set `skip_invalid=False` to surface invalid structures and keep your row IDs aligned.

```python
import pandas as pd

import molactagent as ma

reference = pd.read_csv("examples/cdk2/reference.csv")
query = pd.read_csv("examples/cdk2/query.csv")

reference_features = ma.pp.smiles_to_features(
    reference["smiles"], y=reference["label"], drop_invariant=False, skip_invalid=False
)
query_features = ma.pp.smiles_to_features(
    query["smiles"], drop_invariant=False, skip_invalid=False
)
assert reference_features.feature_names == query_features.feature_names
```

Use the reference bundle's `X`, `y`, feature names, and descriptions with the numeric estimator, then pass the query bundle's `X` to prediction. The helper computes RDKit descriptors and MACCS keys by default. Set `morgan=True` to add Morgan fingerprints. `ma.pp.scaffold_split` can group structures by scaffold when you prepare a split for evaluation. Check class coverage in each split.

### Downstream analysis

Use `ma.metrics.classification_metrics` with measured labels and predictions. `ma.tl.loocv` and `ma.tl.kfold_cv` run cross-validation. Each fold may call the model when you pass a model-backed estimator. `ma.tl.compare_baselines` fits conventional models to supplied data. `ma.tl.confidence_calibration` compares scores with observed correctness, and `ma.tl.aggregate_drivers` summarizes which features were cited across predictions. Use these tools with your own measured outcomes. [Analysis API](../molactagent/tl/__init__.py)

### Experimental regression

The experimental numeric backend groups references by measured target ranges and by feature similarity. It uses the selected references to estimate a continuous value, and can withhold an estimate when evidence is weak. The interval describes the range of selected reference targets. Inspect refusals alongside successful estimates.

```python
import molactagent as ma

penguins = ma.datasets.load_penguins()
regressor = ma.MolActAgentRegressor(
    llm_config=ma.load_llm_config("llm_config.jsonc"),
    feature_names=penguins.feature_names,
    feature_descriptions=penguins.feature_descriptions,
    background=penguins.background,
    target_name="body_mass_g",
)
regressor.fit(penguins.X_train, penguins.y_train)
values, lower, upper = regressor.predict_with_interval(penguins.X_test)
```

`predict_full` returns `RegressionResult` records with the selected evidence and refusal status. Refusals appear as `NaN` in numeric arrays and `REFUSED` in pipeline verdicts. For reports use `ma.run_regression`. For backend work the package exposes `ma.CrossClassificationRegressor`. [Penguins walkthrough](tutorial.md#experimental-regression-example) · [Regression scope](experimental-regression.md)

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `molactagent` is missing or the version looks wrong | Activate the environment used for installation or run `python -m molactagent --version` from that environment. |
| Invalid SMILES or conflicting labels for identical structures | Find the named compound in the reference or query table. Correct the structure and resolve contradictory labels before retrying. |
| Missing or overlapping IDs in a numeric run | Check the column header and use a distinct, nonempty ID for every row across both files. |
| A feature is missing or non-numeric | Match feature headers across files and remove text columns from the reference features. Describe every feature and the label if you supply descriptions. |
| A category column is detected as a continuous target | Check the JSON validation summary and use text codes for discrete labels that numeric inference treats as continuous. |
| Model, authentication, or connection error | Verify the provider's API root, API key, and served model name. Check which file and environment variables supplied them. |
| Truncation, invalid JSON, or HTTP 429 | Check the call records. For molecular runs, review `--max-tokens`, try `--plain-json` for a rejected API JSON-mode parameter, or reduce `--workers` for rate limits. For numeric runs reduce `--max-workers` when the service throttles. |
| Replay mismatch or existing output | Use the original file contents, matching RDKit version, and a compatible release. For schema 1 recordings use v0.4.1. For another molecular output choose a fresh report directory and recording filename. |
| Inconclusive result | Read the reason and compare the usable votes with the planned calls. Check whether both classes have comparable references and which measurement would address the uncertainty. |

## Development and tests

From the checkout, `python -m pip install -e ".[chem,test]"` installs the chemistry and testing extras. Add `yaml`, `excel`, or `parquet` to the extras list for those readers. A Conda setup is also available.

```bash
conda env create -f environment.yml
conda activate molactagent_env
python -m pip install -e .
```

Run the offline tests from the repository root.

```bash
python -m pytest -q -m "not real_llm"
```

Tests marked `real_llm` are opt-in through `MOLACTAGENT_REAL_LLM=1` and use a configured service. Include the command, settings with the key removed, and a small shareable example when reporting an issue. [Contributing](../CONTRIBUTING.md) · [Issue tracker](https://github.com/studentiz/MolActAgent/issues)
