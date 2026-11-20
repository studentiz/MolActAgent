# MolActAgent tutorial

## Predict from SMILES

MolActAgent helps you inspect a prediction for an untested molecule. You supply
its SMILES, examples from both classes, and the experimental definition of the
classes. You get a label or an inconclusive result, with the structures and
recorded judgments that led to it.

The example has six labeled references and one query, 4-fluorophenol. Its
synthetic labels define `active` as having a hydroxyl bonded directly to an
aromatic carbon. Watch for the difference between 4-fluorophenol and benzyl
alcohol, whose hydroxyl is one carbon away from the ring.

### Install the example environment

Install Git and Python 3.12, then open a terminal. The tagged source includes
the CSVs and the recording.

```bash
git clone --branch v0.5.0 https://github.com/studentiz/MolActAgent.git
cd MolActAgent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[chem]" -c examples/molecular/constraints.txt
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv`
and run the install command with `.venv\Scripts\python.exe -m pip` in place
of `python -m pip`. Then use `.venv\Scripts\molactagent.exe` in place of
`molactagent`. PowerShell commands can be pasted on one line after removing
the trailing backslashes.

### Replay a complete prediction

```bash
molactagent molecular \
  --reference examples/molecular/reference.csv \
  --test examples/molecular/query.csv \
  --replay examples/molecular/responses.json \
  --output-dir molecular-output
```

The terminal lists `Q1` as `active`. Open `molecular-output/report.html` in a
browser. The comparison shows p-cresol and benzyl alcohol alongside the
query. The highlighted bonds show where each hydroxyl attaches. Follow a
citation to a reference record and open an individual LLM judgment to read
its interpretation and counter-evidence. Five recorded judgments support
the active label. The closest para-substituted phenol in the panel has a
methyl in place of fluorine, leaving the fluorination comparison unresolved.
[View the published report](https://studentiz.github.io/MolActAgent/molecular-example.html)

The run rebuilds the comparisons, checks each request against the recording,
parses the saved model responses and recomputes the vote. It writes one
entry per query to `predictions.json` with the judgment and call records.
Repeat the command with a fresh output directory to retain both reports.

### Use a live model

Create a local configuration with `molactagent init-config --path llm_config.jsonc`.
Open it and set `api_key`, `base_url` and `model_name` for an OpenAI-compatible
Chat Completions service. A live service receives the query and selected
reference evidence. Use a service approved for your data and keep the local
configuration private.

```bash
molactagent molecular \
  --reference examples/molecular/reference.csv \
  --test examples/molecular/query.csv \
  --id-column id \
  --endpoint @examples/molecular/endpoint.txt \
  --llm-config llm_config.jsonc \
  --temperature 0 --max-tokens 16384 \
  --record teaching-live.json \
  --output-dir teaching-live
```

The main panel asks for five LLM judgments per query by default, with extra
attempts for transient API failures or incomplete responses. Concurrent
judgments can use different backends when your service routes requests.
Use `--workers 1` to send one model request at a time, or `--plain-json` when
your provider supports JSON text and rejects the JSON-mode API option.
`--max-retries 0` sends the initial HTTP request and disables transport
retries. The comparison can still make one response-format correction for a
truncated or invalid model reply.

Replay a CLI recording with the same two input files and its `--replay` path.
The command loads the saved assay text, column names and effective settings.
It checks the file fingerprints before rebuilding the model requests. The
files may be moved without changing their content. The report shows the
requested model, and the response log identifies the model names returned
by the service when present. Keep `--record` files with the input data
because they contain model responses. For recordings made by v0.4.1, use that
tag and the original command options and dependencies.

### Use your own data

Create a reference file with `smiles,label` and a query file with `smiles`.
Supply the `active` and `inactive` reference classes from the same
experimental endpoint. Write the assay definition in a text file and pass
`--endpoint @assay.txt`. Include the assay conditions and what qualifies a
compound for each class.

| Column or option | What it supplies |
| --- | --- |
| `smiles` | One structure per row. Use isomeric SMILES when the configuration is known. |
| `label` | The input activity class on reference rows. |
| `id` | Optional ID for matching outputs to your records. Pass `--id-column id` when both tables have that column. CSV values such as `0007` and `NA` retain their text. |
| `name` | Optional name for display. The CLI reads it wherever available. |
| `--pchembl-column VALUE` | Measured reference potency where your assay uses pChEMBL. Leave unavailable measurements blank. |
| `--pchembl-threshold 6.5` | Optional relevant cutoff to show in the report and give to the LLM. |
| `--labels-from-pchembl` | Declare that reference labels are defined by the cutoff. Requires a potency column and cutoff. Every reference must have a value, with `active` at or above the cutoff. |

Exact canonical matches to the query are excluded per prediction.
Resolve contradictory labels for the same canonical structure before running.
The CLI reports an invalid SMILES with its compound ID before any model
requests. For a new set of tables, start with the default `panel` profile.
`--molecular-profile focused` adds eligible local halogen or stereochemical
comparisons and may make more calls.

With `--output-dir`, a completed batch writes `predictions.json` and
`report.html`, including any inconclusive queries. Choose paths without
existing files of the same name. Without an output directory the JSON goes
to standard output. The exit code is `0` when every query receives a class,
`4` when the files contain one or more inconclusive predictions, or `2` for
an input, configuration, replay or output error. [Report guide](reports.md#reports-from-smiles)

### Use the Python interface

The same classifier accepts pandas DataFrames. Load a CSV with
`dtype=str, keep_default_na=False` to preserve text IDs.

```python
import pandas as pd

from molactagent import build_llm_client, load_llm_config
from molactagent.molecular import MolecularClassifier

reference = pd.read_csv("examples/molecular/reference.csv", dtype=str, keep_default_na=False)
query = pd.read_csv("examples/molecular/query.csv", dtype=str, keep_default_na=False)
config = load_llm_config("llm_config.jsonc", default_max_tokens=16384)
agent = MolecularClassifier(
    build_llm_client(config),
    endpoint="Teaching assay with synthetic labels. Active means an aromatic carbon bonded directly to a hydroxyl.",
    temperature=config.temperature,
    max_tokens=config.max_tokens,
)
agent.fit(reference, id_column="id")
for result in agent.predict(query, id_column="id"):
    print(result.identifier, result.label, result.decision)
```

## Iris tutorial for numeric tables

Use `molactagent run` when you already have numeric features for each row.
The bundled example asks the agent to classify six flowers into three
species using sepal and petal measurements. It contains 24 labeled
references. Validate the files first without contacting the model.

```bash
molactagent validate \
  -r examples/iris_flowers/reference.csv \
  -t examples/iris_flowers/test.csv \
  --label-col species --id-column flower_id --k 3 \
  --feature-descriptions examples/iris_flowers/feature_descriptions.json \
  --background @examples/iris_flowers/background.txt
```

Use the same inputs with `molactagent run` to classify the queries. Pass
`--llm-config llm_config.jsonc` to select the service and `--output-dir
iris-output` for the output directory. Open `interpretability.html` to
review the batch, and `summary.csv` to retrieve the labels. The numeric
engine compares feature values in reference tuples and can apply its
numeric decision rules. See the design document for that policy.
[Feature-table design](design.md#feature-table-engine)

### Adding per-sample reports

Add `--per-sample` to `molactagent run` to save each query's HTML, Markdown
and JSON report. This can ask the model to write a narrative. Saved
classification records with the relevant presentation metadata can also
be opened in the named-compound viewer. [Numeric report guide](reports.md#reports-from-numeric-tables)

### Experimental regression example

The experimental numeric-table backend can use a continuous target. The
penguins example includes a text `species` column. To predict body mass
using only the three numeric measurements, run this Python from the
repository root to prepare two files.

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

Validate the prepared files. Replace `validate` with `run` and supply your
LLM configuration and output directory to generate a prediction.

```bash
molactagent validate \
  -r penguins-input/reference.csv \
  -t penguins-input/test.csv \
  --target-col body_mass_g --id-column sample_id --k 5 \
  --feature-descriptions examples/penguins/feature_descriptions.json \
  --background @examples/penguins/background.txt
```

[Regression status and interpretation](experimental-regression.md)
