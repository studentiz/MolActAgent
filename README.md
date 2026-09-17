<div align="center">

<img src="docs/assets/logo.svg" alt="" width="120" />

# MolActAgent

### An evidence-grounded molecular activity agent.

Small-data molecular activity prediction from SMILES or ordinary feature tables — every prediction comes back with cited evidence, calibrated confidence, and a paragraph you can hand to a domain expert.

[![Python](https://img.shields.io/badge/python-3.10--3.12-1d1d1f.svg?style=flat-square)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-1d1d1f.svg?style=flat-square)](#license)
[![Version](https://img.shields.io/badge/version-0.3.0-0066ff.svg?style=flat-square)](#)
[![Status](https://img.shields.io/badge/status-beta-1d1d1f.svg?style=flat-square)](#)

**[Installation](#installation) · [Quick Start (60 seconds)](#quick-start-60-seconds) · [How it works](#how-it-works) · [Why MolActAgent](#why-molactagent) · [Documentation](#documentation)**

</div>

---

## You have a handful of compounds. Now what?

When molecular data is small and the stakes are high, every traditional answer has a blind spot.

| Approach | Asks for | Tells you why? |
| :--- | :--- | :--- |
| **K-NN** | a distance metric | "Closest is class X." That's it. |
| **Random Forest** | thousands of rows | Can't even start to train. |
| **An LLM directly** | a good prompt | Plausible answers — unaudited. |
| **MolActAgent** | twelve rows + a sentence of context | Verdict, **cited evidence**, and a calibrated confidence you can defend. |

MolActAgent works because **numerical similarity picks the references and the LLM argues from them.** Python decides who the LLM gets to look at and acts as a witness that keeps the LLM honest. The LLM reads only those picked references, reasons over molecular feature evidence, and returns a verdict it can defend in writing.

The result is a small, transparent classifier you can trust *because* you can read its argument.

---

## Installation

MolActAgent isn't on PyPI yet — install it from source. The full sequence:

```bash
# 1. Clone the repository
git clone https://github.com/studentiz/MolActAgent.git

# 2. Move into the project directory
cd MolActAgent

# 3. (Optional but recommended) create an isolated environment
python -m venv .venv && source .venv/bin/activate
# — or —
conda create -n molactagent_env python=3.11 -y && conda activate molactagent_env

# 4. Install in editable mode
pip install -e .
```

Optional extras (chemistry tooling, baseline comparisons, YAML configs, etc.):

```bash
pip install -e ".[chem]"   # RDKit + descriptor pipelines
pip install -e ".[all]"    # everything (recommended for trying things out)
```

Verify the install:

```bash
python -c "import molactagent as ma; print(ma.__version__)"
# → 0.3.0
```

---

## Quick Start (60 seconds)

Three lines of code, the bundled iris dataset (already in the repo), and you're classifying with cited evidence.

```python
import molactagent as ma

# 1. The bundled iris example — 24 reference flowers, 6 unknowns to classify.
#    No data preparation needed. Ships with the source repository.
iris = ma.datasets.load_iris_flowers()

# 2. Build a classifier. The credentials are the only thing you supply.
clf = ma.MolActAgentClassifier(
    api_key="sk-...",                          # your key
    base_url="https://api.deepseek.com",       # any OpenAI-compatible endpoint
    model_name="deepseek-chat",                # whatever model you have access to
)

# 3. Fit and predict.
clf.fit(iris.X_train, iris.y_train)
print(clf.predict(iris.X_test))
# → ['setosa' 'versicolor' 'setosa' 'versicolor' 'virginica' 'virginica']
```

**That's it.** No fine-tuning, no embedding store, no vector database — just a structured contrastive prompt around your reference set.

MolActAgent works with any **OpenAI-compatible** endpoint: DeepSeek, OpenAI, Anthropic-via-OpenRouter, Qwen, Mistral, Gemini, or local Ollama / vLLM / LM Studio. Pick whichever fits your pricing and privacy needs.

### What you get back

`predict` returned just the labels. `predict_full` returns the receipts:

```python
results = clf.predict_full(iris.X_test)

print(results[0].label)               # → "setosa"
print(results[0].confidence)          # → 0.89
print(results[0].driver_attribution)
# → [{'feature': 'petal_length', 'description': 'length in cm',
#     'rationale': 'short — characteristic of setosa', ...},
#    {'feature': 'petal_width', ...}]
```

Each result also carries `picked_references`, `counter_evidence`, the raw tuple-by-tuple `vote_summary`, and the original `test_feature_values`. When you go through the high-level pipeline (`ma.run_classification(...)`), the same evidence is auto-rendered to a polished `interpretability.html` report on disk.

A worked sample verdict for the first iris row:

| Field | Value |
| :--- | :--- |
| **Verdict** | `setosa` |
| **Calibrated confidence** | `0.89` |
| **Drivers** | `petal_length = 1.4 cm` (short, characteristic of setosa); `petal_width = 0.2 cm` (narrow, characteristic of setosa) |
| **References cited** | rows 3, 7, 12 — all setosa, mean petal_length 1.5 ± 0.2 cm |
| **Counter-evidence considered** | `sepal_length = 5.1 cm` overlaps the versicolor range; ruled out by the unambiguous petal evidence |

## Molecular activity in one call

For compound data, MolActAgent can turn SMILES into a feature matrix before fitting. The chemistry extra provides RDKit descriptors, MACCS keys, optional Morgan fingerprints, and Murcko scaffolds for scaffold-aware evaluation.

```python
import molactagent as ma

features = ma.datasets.load_bslt_regression(
    descriptors=True,
    maccs=True,
    morgan=False,
)

reg = ma.MolActAgentRegressor(
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    model_name="deepseek-flash",
    feature_names=features.feature_names,
    feature_descriptions=features.feature_descriptions,
    background=ma.pp.chemistry.BSLT_BACKGROUND,
    k=3,
).fit(features.X[:11], features.y[:11])

predictions, lower, upper = reg.predict_with_interval(features.X[11:])
```

Install RDKit with `pip install -e ".[chem]"` or use the supplied Conda environment for the most reproducible chemistry setup.

---

## How it works

Inputs flow through **four stages** — three deterministic Python stages and one LLM stage — and an aggregator that may make one optional extra LLM call to render the report.

| Stage | Actor | What happens |
| :---: | :--- | :--- |
| **1** | Python *(deterministic)* | **Witness layer.** Standardize features; compute contextual similarity (feature distance blended with neighbourhood-of-neighbourhoods overlap); for each candidate class take the top-K contextually closest references; pick the M classes in contention (typically M = 2). |
| **2** | Python *(deterministic)* | **Prompt assembler.** Build one structured prompt per (M+1)-tuple of (test, ref₁, …, ref_M): discriminative-feature table, deterministic alignment tally as ground truth, per-reference texture blocks, strict JSON output schema. |
| **3** | LLM *(one call per tuple, parallel)* | **Reasoner.** Returns structured JSON: predicted label, 3–6 named drivers with per-driver justification, supporting/opposing features, summary, counter-evidence. |
| **4** | Python *(plus one optional LLM call)* | **Aggregator.** Weighted vote tally across tuples → final label, calibrated confidence (margin × coverage × sample-size), weighted driver attribution. One optional extra LLM call writes the Markdown / HTML interpretability report. |

Read the full architectural breakdown in [`docs/design.md`](docs/design.md) — every stage, every hyperparameter, and the reasoning behind every design choice.

---

## Why MolActAgent

<table>
<tr>
<td width="33%" valign="top">

### Built for small data

Production-quality verdicts on as few as twelve labelled samples — the regime where deep models can't even start to train and a single curated reference matters more than another GPU.

</td>
<td width="33%" valign="top">

### Cited evidence, every time

Every prediction returns the references it leaned on, the features that drove the call, the counter-evidence considered, and a calibrated confidence — not a single opaque score.

</td>
<td width="33%" valign="top">

### scikit-learn ergonomics

`fit`, `predict`, `predict_proba`, `score`. Drops into `cross_val_score`, `Pipeline`, `GridSearchCV`. If you've ever called `.fit(X, y)`, you already know how to use it.

</td>
</tr>
<tr>
<td width="33%" valign="top">

### Real-LLM validated

End-to-end runs at **91.67% iris LOOCV accuracy** on a real DeepSeek endpoint. Not "passes a mock" — passes a real model.

</td>
<td width="33%" valign="top">

### Vendor-neutral

OpenAI, DeepSeek, Anthropic-via-OpenRouter, Qwen, Mistral, Gemini, or local Ollama / vLLM / LM Studio. Anything OpenAI-compatible works.

</td>
<td width="33%" valign="top">

### Polished CLI too

`molactagent run`, `molactagent validate`, `molactagent init-config` — pipe-friendly, ready for batch jobs and shell scripting.

</td>
</tr>
</table>

---

## Configuration

MolActAgent accepts LLM credentials in **four** equivalent ways. The Quick Start above used **path 1**; here are the rest.

### 1. In-code — best for notebooks and ad-hoc Python

```python
clf = ma.MolActAgentClassifier(
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    model_name="deepseek-chat",
)
```

Optional context — omit for sensible defaults, supply inline when it matters:

```python
clf = ma.MolActAgentClassifier(
    api_key="sk-...", base_url="...", model_name="...",
    feature_descriptions={
        "petal_length": "length of the petal in centimetres",
        "petal_width":  "width of the petal in centimetres",
    },
    background="Botanical morphometric measurements of iris flowers.",
    k=3,                # per-class neighbours retained for the LLM
    temperature=0.0,    # LLM determinism knob
)
```

Pandas DataFrames are first-class — column names become feature names automatically:

```python
import pandas as pd

df = pd.read_csv("my_data.csv")
X = df[["sensor_a", "sensor_b", "sensor_c"]]
y = df["category"]

clf = ma.MolActAgentClassifier(api_key="sk-...", base_url="...", model_name="...")
clf.fit(X, y)              # feature names auto-inferred from X.columns
```

### 2. Pre-built config file — best for batch jobs and team-shared projects

```python
cfg = ma.load_llm_config("llm_config.jsonc")
clf = ma.MolActAgentClassifier(llm_config=cfg)
```

Generate a starter file with `molactagent init-config`. It supports comments, multiple vendor templates, and the same field names every example uses.

### 3. Pre-built client — best for tests and shared sessions

```python
client = ma.build_llm_client(
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    model_name="deepseek-chat",
)
clf = ma.MolActAgentClassifier(llm_client=client)
```

Useful when you want one `LLMClient` shared across many estimators, or when injecting a test fake.

### 4. Auto-discovery — zero ceremony

```bash
export MOLACTAGENT_API_KEY="sk-..."
export MOLACTAGENT_BASE_URL="https://api.deepseek.com"
export MOLACTAGENT_MODEL="deepseek-chat"
```

```python
clf = ma.MolActAgentClassifier()  # credentials auto-discovered
```

MolActAgent walks up from the current directory looking for `llm_config.jsonc` / `LLM_API.txt`, then falls back to the `MOLACTAGENT_*` environment variables.

---

## What you get

- **A scikit-learn estimator that reasons.** `MolActAgentClassifier` implements the standard estimator contract — slots into `cross_val_score`, `GridSearchCV`, and `Pipeline` with zero friction.
- **Probabilities and confidence as separate axes.** `predict_proba` answers *"how much weighted evidence backs the winner?"*. `predict_with_confidence` is the engine's calibrated trust score (margin × coverage × sample-size). Both are useful; both are exposed.
- **Per-sample interpretability.** `predict_full` returns the picked references, the cited drivers, counter-evidence the model considered, and the raw tuple-by-tuple vote tally — auto-rendered to Markdown / HTML by the high-level pipeline.
- **Polished CLI.** `molactagent run`, `molactagent validate`, `molactagent init-config` — pipe-friendly and ready for batch jobs.

> **About regression.** MolActAgent's primary mission is classification. Regression is supported as a derivative capability — the **Cross-Classification Regressor** decomposes a continuous target into two binned classification problems and recombines them. It's competitive on regression tasks that *can be reframed as classification*; for general regression, use a tool built for it.

---

## Documentation

| Resource | What it is |
| :--- | :--- |
| [`docs/tutorial.md`](docs/tutorial.md) | The canonical step-by-step tutorial. Start here — it covers both the CLI and the Python API end-to-end. |
| [`docs/design.md`](docs/design.md) | The architectural design document — every stage of the pipeline, every hyperparameter, every design decision. |
| `molactagent --help` | Full CLI reference. |

---

## License

MIT. See [LICENSE](LICENSE) for details.

<div align="center">
<sub>Built with care for researchers, scientists, and engineers working at the small-data frontier.</sub>
</div>
