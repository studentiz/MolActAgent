# MolActAgent — A Friendly, Step-by-Step Tutorial

Welcome! This tutorial is written for **complete beginners** — even if
you have never used Python, the command line, or an AI service, you
should be able to follow along. We will go slowly and explain every
step.

We will do **one example from start to finish**: tell the computer
about a few flowers we already know (`reference.csv`), show it four
new flowers we do NOT know (`test.csv`), and ask it to decide whether
each new flower is an **Iris setosa** or an **Iris versicolor** — and
to explain *why* in plain English. Everything you need is already in
the `examples/iris_flowers/` folder.

> **The big promise of MolActAgent**: you show it a few labelled
> examples of two kinds of things, it asks a Large Language Model
> (LLM) to compare each new thing against those examples, and it
> gives you a verdict with a human-readable explanation you can
> double-check.

---

## Table of contents

1. [What you need before you start](#1-what-you-need-before-you-start)
2. [Installing MolActAgent](#2-installing-molactagent)
3. [The big idea, in three sentences](#3-the-big-idea-in-three-sentences)
4. [A tour of the iris flower example](#4-a-tour-of-the-iris-flower-example)
5. [Setting up your LLM (your AI service)](#5-setting-up-your-llm-your-ai-service)
6. [Step 1 — Validate the data](#6-step-1--validate-the-data)
7. [Step 2 — Run the classification (default mode)](#7-step-2--run-the-classification-default-mode)
8. [Reading the default outputs](#8-reading-the-default-outputs)
9. [Adding per-sample reports](#9-adding-per-sample-reports)
10. [How MolActAgent handles multi-class](#10-how-molactagent-handles-multi-class)
11. [Predicting continuous values (regression)](#11-predicting-continuous-values-regression)
12. [Pipe-friendly mode (using MolActAgent inside other tools)](#12-pipe-friendly-mode-using-molactagent-inside-other-tools)
13. [Bringing your own data](#13-bringing-your-own-data)
14. [The full CLI reference](#14-the-full-cli-reference)
15. [Common error messages](#15-common-error-messages)
16. [Using MolActAgent from Python](#16-using-molactagent-from-python)
    - 16.1 [sklearn-style estimators](#161--sklearn-style-estimators-molactagentclassifier-molactagentregressor)
    - 16.2 [Compound chemistry: SMILES → MolActAgent](#162--compound-chemistry-smiles--molactagent-in-one-call)
    - 16.3 [Downstream analysis: LOOCV, baselines, calibration](#163--downstream-analysis-loocv-baselines-calibration-attribution)
17. [FAQ](#17-faq)

---

## 1. What you need before you start

You need four things on your computer. Don't worry if you've never
heard of some of them — each row tells you what to do if you don't.

| What | Why | If you don't have it |
| :--- | :--- | :--- |
| **A computer** running macOS, Linux, or Windows. | To run the program. | — |
| **A terminal app** (the black/white text window where you type commands). | All the commands below are typed into a terminal. | **macOS**: open the **Terminal** app (in *Applications → Utilities*). **Windows**: open **PowerShell** (search "PowerShell" in the Start menu). **Linux**: open any "Terminal" app. |
| **Python 3.10–3.12**. | MolActAgent is tested on these Python versions. Check by typing `python3 --version` in your terminal. | Install from [python.org](https://www.python.org/downloads/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). |
| **Git** (a tool for downloading code from the internet). | We'll use it to copy the MolActAgent project onto your computer. Check by typing `git --version` — if it prints something like `git version 2.39.x`, you're good. | **macOS**: Git comes built-in; if it's missing, the first `git --version` you type will offer to install it for you. **Windows**: install from [git-scm.com/download/win](https://git-scm.com/download/win). **Linux**: `sudo apt install git` (Ubuntu/Debian) or `sudo yum install git` (Fedora/CentOS). |
| **An API key for an OpenAI-compatible LLM**. | MolActAgent asks an AI to compare your data and explain the result. | Sign up for one of: DeepSeek ([platform.deepseek.com](https://platform.deepseek.com)) — cheap, recommended for first try; Alibaba Qwen ([dashscope.console.aliyun.com](https://dashscope.console.aliyun.com)); OpenAI ([platform.openai.com](https://platform.openai.com)); or Anthropic, Google Gemini, Mistral, or a local Ollama model. Any provider that speaks the OpenAI API format works. |

---

## 2. Getting MolActAgent onto your computer

We'll do this in **five small steps**. Open your terminal and type
each command exactly. Press <kbd>Enter</kbd> after each one and wait
for it to finish before typing the next one.

> **Copy-paste tip.** Click and drag to highlight a command below,
> then <kbd>Cmd</kbd>+<kbd>C</kbd> (macOS) or <kbd>Ctrl</kbd>+<kbd>C</kbd>
> (Windows / Linux) to copy. In the terminal, paste with
> <kbd>Cmd</kbd>+<kbd>V</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>V</kbd>.

### Step 2.1 — Download the project

This downloads a copy of the MolActAgent project from GitHub onto your
computer:

```bash
git clone https://github.com/studentiz/MolActAgent.git
```

You should see a few lines like `Cloning into 'MolActAgent'...` and
`Receiving objects: 100%`. When it's done, a new folder called
`MolActAgent` has appeared in whatever folder your terminal was sitting
in (usually your home folder).

### Step 2.2 — Enter the project folder

From now on, every command you type needs to be run *inside* the
MolActAgent folder. Move into it with:

```bash
cd MolActAgent
```

`cd` stands for "change directory". Now your terminal is "inside" the
project.

### Step 2.3 — (Recommended) create an isolated Python environment

This keeps MolActAgent's Python packages separate from anything else
on your computer, so nothing breaks. Pick **one** of the two
commands below — whichever matches the tools you have:

```bash
# If you installed Python from python.org:
python3 -m venv .venv && source .venv/bin/activate
```

```bash
# If you installed Miniconda / Anaconda:
conda create -n molactagent_env python=3.11 -y && conda activate molactagent_env
```

You only need to do this *once*. Every time you open a new terminal
in the future, re-run just the activation part:

```bash
source .venv/bin/activate          # for venv
conda activate molactagent_env      # for conda
```

> **Skipping this step is OK** for a quick try-it-out, but the
> isolated environment saves you from "package version conflict"
> headaches if you use Python for other projects.

### Step 2.4 — Install MolActAgent

```bash
pip install -e .
```

This reads the project's package list and installs everything
MolActAgent needs. It usually takes 1–2 minutes. The `-e` means
"editable install" — if the code is updated later, your installation
picks up the new version automatically.

If you also want optional extras (Excel / Parquet / YAML support,
chemistry tooling), run this instead:

```bash
pip install -e ".[all]"
```

### Step 2.5 — Check that it worked

```bash
molactagent --version
```

You should see:

```text
molactagent 0.3.0
```

If you see that line, MolActAgent is installed correctly. 

> **Tip.** If `molactagent` says "command not found", try
> `python3 -m molactagent --version` instead. Both ways work.

> **Reminder.** From this point on, every command in this tutorial
> assumes you're inside the `MolActAgent` folder with your environment
> activated. If you closed your terminal, re-open one and run:
> `cd MolActAgent && source .venv/bin/activate` (or
> `conda activate molactagent_env`).

---

## 3. The big idea, in three sentences

1. You give MolActAgent a table of **known** examples (the *reference
   dataset*, with labels for classification or with continuous target
   values for regression) and a table of **unknown** examples to
   predict (the *test dataset*).
2. For every unknown example, MolActAgent picks the **k** most similar
   known examples from each candidate class, decides which classes
   are genuinely "in contention" for that sample (typically 2;
   sometimes 3), forms small **tuples** of one neighbour per
   candidate class, and asks the LLM — "given these reference
   examples, which class does the test example look most like, and
   why?"
3. MolActAgent collects all those per-tuple votes, picks the winner
   (or reports `Inconclusive` if no class is meaningfully ahead),
   and writes one **interpretability report** (Markdown + HTML)
   showing the per-class vote tally, the top features that drove
   every decision, and how the final confidence was calibrated.

That's it for classification. **Regression works exactly the same
way** — under the hood, MolActAgent reformulates regression as
*two* classifications (one over equal-frequency target bins, one
over unsupervised feature clusters), runs the same contrastive
engine on each, picks the nearest references in each predicted
class, and combines their actual target values into a confidence-
weighted numeric prediction. See §11.

No training, no machine learning model, no GPU — just smart
prompting. Binary classification (two classes) is the natural M=2
special case of this design and runs through exactly the same
code path.

---

## 4. A tour of the iris flower example

The folder `examples/iris_flowers/` has everything the tutorial
needs:

```text
examples/iris_flowers/
├── reference.csv             # 24 flowers we already know the species of
├── test.csv                  # 6 flowers to classify
├── feature_descriptions.json # plain-English meaning of each column
├── background.txt            # a short note telling the AI what we're doing
├── llm_config.jsonc          # where you paste your API key
└── outputs/                  # created automatically when you run (Step 2)
```

### 4.1 `reference.csv` — the "study guide" for the AI

This file has 24 rows, 4 features (petal / sepal measurements in
centimetres), one **label column** (`species`), and a unique row
**ID** (`flower_id`):

```csv
flower_id,sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,species
S01,5.1,3.5,1.4,0.2,setosa
S02,4.9,3.0,1.4,0.2,setosa
...
V01,7.0,3.2,4.7,1.4,versicolor
V02,6.4,3.2,4.5,1.5,versicolor
...
G01,6.3,3.3,6.0,2.5,virginica
G02,5.8,2.7,5.1,1.9,virginica
...
```

Each `species` label is one of `setosa`, `versicolor`, or
`virginica`. MolActAgent **requires at least two classes** with at
least one row per class — there is no upper bound, but the LLM
reasons most reliably with up to ~5 classes.

### 4.2 `test.csv` — the unknowns

The four new flowers we want to classify. Same columns as
`reference.csv` **minus the label** (the label is optional in
`test.csv`; if you include it, MolActAgent will report accuracy):

```csv
flower_id,sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm
T1,5.0,3.3,1.4,0.2
T2,6.6,3.0,4.4,1.4
T3,4.8,3.1,1.6,0.2
T4,6.1,2.9,4.7,1.4
```

> **Important: `flower_id` must be unique across BOTH files
> combined.** MolActAgent uses that ID to label every line in the
> interpretability report and (if you opt in) to name each per-sample
> file. The validator enforces this for you.

### 4.3 `feature_descriptions.json` — what the columns mean

This is **optional but strongly recommended**. If you include it,
it must describe *every* column (including the label):

```json
{
  "sepal_length_cm": "How long the green leaf-like part under the petals is, measured in centimetres.",
  "sepal_width_cm":  "How wide that same green leaf-like part is, measured in centimetres.",
  "petal_length_cm": "How long the colourful petal of the flower is, measured in centimetres. Setosa flowers have very short petals; versicolor petals are much longer.",
  "petal_width_cm":  "How wide the colourful petal is, measured in centimetres.",
  "species":         "Which kind of iris flower this is — either 'setosa' or 'versicolor'. This is the column we want to predict for new flowers."
}
```

The AI reads these descriptions and uses them in its explanations.

### 4.4 `background.txt` — research context

Also optional. One or two paragraphs telling the AI what you are
studying. Example:

```text
We are studying two kinds of iris flowers: Iris setosa and Iris versicolor.
Botanists tell them apart mostly by looking at the petals: setosa flowers have
short, narrow petals, while versicolor flowers have much longer and wider
petals. The sepals overlap a bit more between the two species, so they are
weaker clues. Our goal is to look at a flower's measurements and decide
whether it is more likely to be a setosa or a versicolor, and to explain
*why* using the petal/sepal numbers.
```

Why it helps: without it, the AI treats the columns as anonymous
numbers. With it, the AI knows that "petal_length_cm = 4.4 cm" is
much larger than typical setosa petals, and it says so in the
report.

---

## 5. Setting up your LLM (your AI service)

**Run this once** in your project directory to create a starter
LLM config file:

```bash
molactagent init-config
```

That writes a file called `llm_config.jsonc` next to you. Open it
in any text editor. The extension `.jsonc` means
**JSON-with-comments** — a JSON file where `//` and `/* ... */`
comments are allowed (so the file can document itself).

```jsonc
// ─────────────────────────────────────────────────────────────────
// MolActAgent LLM API configuration (JSONC — JSON with comments)
//   * Keep exactly ONE provider block active (uncommented).
//   * Required fields: api_key, base_url, model_name.
//   * Replace REPLACE-WITH-YOUR-OWN-KEY with your real API key.
// ─────────────────────────────────────────────────────────────────

{
  // Active provider — DeepSeek (cheap and reliable for most experiments)
  "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
  "base_url": "https://api.deepseek.com",
  "model_name": "deepseek-chat",
  "temperature": 0.0,
  "max_tokens": 2000,
  "timeout": 120,
  "max_retries": 10
}

/* ──────── Other vendor templates (copy ONE into the block above) ────────
  // ── Alibaba Cloud DashScope (Qwen) ─────────────────────────────
  { "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model_name": "qwen-plus" }

  // ── OpenAI ─────────────────────────────────────────────────────
  { "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini" }

  // ... Claude (via OpenRouter), Gemini, Mistral, local Ollama ...
──────────────────────────────────────────────────────────────────────── */
```

**What to do:**

1. Get an API key from your LLM provider of choice.
2. Replace `REPLACE-WITH-YOUR-OWN-KEY` in the active block with your
   real key (the quotes around it are required).
3. If you want to use a different vendor, copy one of the block
   templates from inside the `/* ... */` section, paste it over the
   active DeepSeek block, and put your real key in.
4. Save the file.

> **Why JSONC and not plain JSON?** Because the file is meant to be
> edited by humans, and humans need comments to remember what each
> field does and to keep alternative vendor blocks at hand. JSON
> doesn't support comments; JSONC does. MolActAgent strips the
> comments before parsing, so the file behaves like normal JSON to
> the loader.

**Where MolActAgent looks for the file.** When you don't pass
`--llm-config`, MolActAgent walks up from your current directory
looking for any of: `llm_config.jsonc`, `llm_config.json`,
`llm_config.yaml`, `llm_config.yml`, `LLM_API.txt`. The first
match wins. So if you keep the file at your project root, every
`molactagent ...` command "just works" without flags.

**Alternative: environment variables.** If you prefer not to have a
file at all, you can also set:

```bash
export MOLACTAGENT_API_KEY="sk-..."
export MOLACTAGENT_BASE_URL="https://api.deepseek.com"
export MOLACTAGENT_MODEL="deepseek-chat"
```

The CLI reads these as a fallback when no `--llm-config` is given
and no config file is found by the walk-up search.

---

## 6. Step 1 — Validate the data

Before spending a single LLM token, always run `validate` first. It
checks every rule MolActAgent cares about (at least two classes, no
duplicate IDs, feature descriptions complete, numeric feature
values, etc.) and tells you exactly what is wrong if anything is
off.

From the project root, run:

```bash
molactagent validate \
  -r examples/iris_flowers/reference.csv \
  -t examples/iris_flowers/test.csv \
  --label-col species \
  --id-column flower_id \
  --k 3 \
  --feature-descriptions examples/iris_flowers/feature_descriptions.json \
  --background "@examples/iris_flowers/background.txt"
```

> **What these flags mean:**
> - `-r` / `-t` — path to the reference and test files.
> - `--label-col` — the column name that holds the class label (must
>   exist in `reference.csv`; optional in `test.csv`).
> - `--id-column` — **required.** The column that holds a unique
>   identifier for every row. It must exist in **both** files, and
>   the IDs must be unique across the two files combined.
> - `--k` — how many nearest neighbours per class to pull from the
>   reference set for each test sample (3 is a sensible default).
> - `--feature-descriptions` — optional JSON/YAML file with one
>   human-readable description per column (covers *all* columns,
>   including the label).
> - `--background "@path"` — the leading `@` tells MolActAgent to read
>   the text from a file. Without `@`, the argument is treated as the
>   literal background string.

You should see:

```text
2026-04-18 03:08:07 [INFO] molactagent.io: Loaded reference.csv: 24 rows × 6 columns
2026-04-18 03:08:07 [INFO] molactagent.io: Loaded test.csv: 6 rows × 5 columns
Validation: OK
  - reference_classes: ['setosa', 'versicolor', 'virginica']
  - reference_class_counts: {'setosa': 8, 'versicolor': 8, 'virginica': 8}
  - class_labels: ['setosa', 'versicolor', 'virginica']
  - n_features: 4
  - feature_columns: ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
  - n_reference_samples: 24
  - n_test_samples: 6
  - k: 3
  - id_column: flower_id
```

MolActAgent treats every class equally — there is no longer a
"positive" or "negative" axis. The `class_labels` line shows the
canonical sorted order of the classes, which the reports use for
column ordering and colour assignment. The legacy
`--positive-class` flag is still accepted for backward compatibility
but has no effect; you can safely omit it.

Want machine-readable output for a script? Add `--json`:

```bash
molactagent validate ... --json
```

returns a JSON object like
`{"is_valid": true, "summary": {...}, "warnings": [...], "errors": []}`.

---

## 7. Step 2 — Run the classification (default mode)

Everything is ready. Ask MolActAgent to classify the four unknown
flowers:

```bash
molactagent run \
  -r examples/iris_flowers/reference.csv \
  -t examples/iris_flowers/test.csv \
  --label-col species \
  --id-column flower_id \
  --k 3 \
  --feature-descriptions examples/iris_flowers/feature_descriptions.json \
  --background "@examples/iris_flowers/background.txt" \
  -o examples/iris_flowers/outputs
```

New flags compared to `validate`:

- `--llm-config PATH` — the JSONC/JSON/YAML/TXT file you edited in Step 5.
- `-o` / `--output-dir PATH` — where to save all reports.

> The `--positive-class` flag from older binary versions is still
> accepted but is now ignored (the multi-class pipeline treats every
> class equally). You will see a deprecation warning if you pass it.

While it runs you'll see a line per test sample ("T1 → setosa …") and
periodic `[LLM #xx]` log lines that show how long each LLM call took.
Expect **~30 seconds per test sample** for a DeepSeek-sized model.

### What gets saved by default

The default mode is intentionally **token-light**:

```text
examples/iris_flowers/outputs/
├── summary.csv              # one row per test sample
├── interpretability.md      # aggregate "why?" report (Markdown)
└── interpretability.html    # same content, styled HTML
```

That's it — no per-sample HTML, no per-sample JSON, no `summary.json`.
The reasoning is that those extras cost roughly **one extra LLM call
per test sample** (the explanation pass), which can easily double
your token bill. Whenever you actually want them, add `--per-sample`
(see [Section 9](#9-adding-per-sample-reports)).

### What you see in the terminal

At the end of the run you get a rich summary plus the
**interpretability panel** rendered live:

```text
============================================================
CLASSIFICATION SUMMARY
============================================================
Sample Predicted     Confidence  True           Time(s)
------------------------------------------------------
T1  setosa         95.0%      —                 34.2
T2  versicolor     95.0%      —                 36.8
T3  setosa         95.0%      —                 32.7
T4  versicolor     95.0%      —                 32.3
------------------------------------------------------
4 succeeded, 0 failed.
Summary CSV: examples/iris_flowers/outputs/summary.csv
LLM stats: 36 calls, 75260 tokens, 665.4s wall.

============================================================
INTERPRETABILITY — WHY MOLACTAGENT SAID WHAT IT SAID
============================================================
...
── T2  →  versicolor  (95% confidence)
   Verdict rationale: Vote tally 9→Pos / 0→Neg / 0→Unknown (weighted
       8.55 vs 0.00); weighted margin 1.00; small reference pool
       (min class size 8) discounts confidence
   Triplet votes: Pos=9 Neg=0 Unknown=0  (weighted 8.55 vs 0.00)
   Confidence basis: margin=1.00  sample_size=0.80  coverage=1.00
   Top drivers (feature · direction · strength · citations):
     • petal_length_cm  →versicolor  weighted=+8.65  cited=9 — ...
     • petal_width_cm   →versicolor  weighted=+8.20  cited=9 — ...
     • sepal_length_cm  →versicolor  weighted=+6.30  cited=9 — ...
     • sepal_width_cm   →versicolor  weighted=+3.05  cited=8 — ...
   Reference context: 3 versicolor + 3 setosa neighbors from 16 ref. samples
       (9 pair comparisons)
...

Interpretability (Markdown) saved to: examples/iris_flowers/outputs/interpretability.md
Interpretability (HTML) saved to: examples/iris_flowers/outputs/interpretability.html
```

> **How many tokens did we save?** With the new defaults the iris
> example used **75260 tokens (36 LLM calls)**. Adding `--per-sample`
> bumps that to **~97000 tokens (40 calls)** — about **+30%** more
> tokens per run, just for the polished per-flower narratives. Skip
> them when you only need the verdicts.

### How to read the interpretability panel

- **Verdict rationale** — one-line summary of how the final label was
  decided.
- **Tuple votes** — for each test sample, MolActAgent runs `k^M`
  tuple comparisons where M is the number of candidate classes that
  were "in contention" (typically 2; sometimes 3). The
  per-class-name counts show the raw vote tally; the "weighted"
  numbers factor in each vote's confidence.
- **Confidence basis** — three sub-scores that combine into the final
  confidence: how big the margin is between the winning class and
  the runner-up, how many reference samples there are, and how
  consistently features were cited.
- **Top drivers** — the features cited most often and most strongly
  across the 9 comparisons. The arrow shows which class the driver
  pointed toward (e.g. `→versicolor`); `weighted` is the summed
  evidence strength; `cited=` counts how many of the 9 comparisons
  mentioned it. The short excerpt after the dash is the first
  per-pair rationale the LLM gave for that feature.
- **Opposing evidence** — features (if any) that worked *against*
  the verdict; decisive classifications usually leave this empty.
- **Reference context** — how many neighbours were used and how
  many pair comparisons were run.

---

## 8. Reading the default outputs

### 8.1 `summary.csv` — the verdicts table

```csv
test_index,sample_id,predicted_label,confidence,true_label,elapsed_s,html_report,error
0,T1,setosa,0.95,,34.21,,
1,T2,versicolor,0.95,,36.76,,
2,T3,setosa,0.95,,32.7,,
3,T4,versicolor,0.95,,32.33,,
```

Open this in Excel / Numbers / Google Sheets to get a quick
overview. The `html_report` column is empty in default mode (no
per-sample HTMLs are generated); switch to `--per-sample` and it
fills in automatically.

### 8.2 `interpretability.md` — Markdown reasoning report

A complete, human-readable Markdown file. Excerpt:

```markdown
# MolActAgent — Interpretability Report

Class labels: **versicolor** (positive) vs **setosa** (negative).

## Verdicts at a glance

| Sample | Verdict | Confidence | Pos votes | Neg votes | Unknown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T1` | **setosa**     | 95% | 0 (0.00) | 9 (8.55) | 0 |
| `T2` | **versicolor** | 95% | 9 (8.55) | 0 (0.00) | 0 |
| `T3` | **setosa**     | 95% | 0 (0.00) | 9 (8.55) | 0 |
| `T4` | **versicolor** | 95% | 9 (8.55) | 0 (0.00) | 0 |

## T2 → versicolor (95% confidence)

**Verdict rationale.** Vote tally 9→Pos / 0→Neg / 0→Unknown ...

**Top drivers.**

| # | Feature           | Direction    | Weighted | Citations | First rationale |
| 1 | petal_length_cm   | →versicolor  | +8.65    | 9         | This is the most decisive feature; ... |
| 2 | petal_width_cm    | →versicolor  | +8.20    | 9         | Domain context states setosa petals ... |
...
```

Open it in any Markdown viewer (or just read it as text).

### 8.3 `interpretability.html` — the same content, styled

Open `examples/iris_flowers/outputs/interpretability.html` in your
browser for a polished, card-based version with colour-coded verdicts
and tables that look great when shared with a non-technical
collaborator.

### 8.4 Optional: `summary.json`

If you need a single machine-readable file with the full validation
block, label mapping, per-row results, and LLM stats, add
`--summary-json`:

```bash
molactagent run ... --summary-json
```

Produces `summary.json` next to `summary.csv`. Useful when chaining
MolActAgent into a bigger pipeline.

---

## 9. Adding per-sample reports

When you want a **full per-flower narrative** (the LLM-written
multi-paragraph explanation, with key-driver tables, robustness
analysis, counter-evidence discussion, and confidence caveats), add
`--per-sample`:

```bash
molactagent run \
  -r examples/iris_flowers/reference.csv \
  -t examples/iris_flowers/test.csv \
  --label-col species --id-column flower_id --k 3 \
  --feature-descriptions examples/iris_flowers/feature_descriptions.json \
  --background "@examples/iris_flowers/background.txt" \
  --positive-class versicolor \
  -o examples/iris_flowers/outputs \
  --per-sample
```

Now your output directory looks like this:

```text
examples/iris_flowers/outputs/
├── summary.csv
├── interpretability.md
├── interpretability.html
├── T1.html / T1.md / T1.json   ← per-sample reports (NEW)
├── T2.html / T2.md / T2.json
├── T3.html / T3.md / T3.json
└── T4.html / T4.md / T4.json
```

Each `T{n}.html` is a beautifully formatted standalone report:
hero verdict, key-driver table, multi-section narrative, robustness
card, raw-evidence appendix. Open one in a browser and you'll see
sections like "Why These Drivers Matter", "Counter-evidence
Considered", and "Recommended Next Steps" — all generated by the
LLM from the per-pair evidence.

### Just want one format?

If you only want the per-sample JSON (and no HTML or Markdown narrative):

```bash
molactagent run ... --per-sample --per-sample-format json
```

Accepted values: any comma-separated subset of `html,md,json`. When
you exclude `html` AND `md`, MolActAgent **skips the explanation LLM
call entirely** — saving the same tokens as default mode while still
giving you per-sample JSON.

### Quiet mode: skip even the interpretability files

```bash
molactagent run ... --no-interpretability
```

You'll only get `summary.csv` (and `summary.json` if you asked for
it). Useful for super-minimal scripted runs.

---

## 10. How MolActAgent handles multi-class

MolActAgent treats binary classification as the M=2 special case of a
single multi-class pipeline. Two design choices keep the cost
low and the explanations honest as the number of classes grows:

### Adaptive top-M selection

For each test sample, MolActAgent computes the closest reference
neighbour in *every* class, then asks: which classes are genuinely
"in contention" for this sample? A class joins the candidate set
when its closest neighbour is within roughly 1.5x the closest
class's closest-neighbour distance. The result is capped at M=4 and
floored at M=2 (so there is always something to compare against).

In practice this gives:

| Test sample lives in… | Typical M | Cost compared to binary |
| :--- | :--- | :--- |
| One cluster, far from the rest | M=2 | unchanged (`k^2` LLM calls) |
| Boundary between two clusters | M=2 | unchanged |
| Boundary among three clusters | M=3 | ~3x (`k^3` LLM calls) |
| Genuinely between four | M=4 | ~9x (`k^4` LLM calls) |

Empirically M=2 covers around 80% of test samples, M=3 covers most
of the rest, and M=4 is rare. So multi-class adds only a moderate
average overhead — you only pay for M>2 when the data actually
warrants it.

### Per-tuple multi-way comparison

For a candidate set of M classes MolActAgent forms (M+1)-tuples:
one neighbour per candidate class plus the test sample. For each
tuple the LLM sees a wide table comparing the test sample side-by-
side against M reference samples, an alignment tally telling it
which class's reference is numerically closest on each feature, and
a baseline-class verdict from that tally. Its job is to pick 3-6
features as drivers, name the favoured class for each one, and emit
a final label that should match the baseline unless decisively
contradicted.

The aggregator then collects per-class votes from every tuple,
weights them by stated confidence, and either picks a winner or
reports `Inconclusive` if no class is meaningfully ahead.

### When you'll see `Inconclusive`

The verdict is `Inconclusive` whenever no class is meaningfully
ahead of the runner-up — for example a clean three-way tie among
classes A, B and C. The system refuses to invent a winner when the
evidence is split. If you see `Inconclusive` often, your reference
set may be too small per class or the test samples may be sitting
in a region the reference set does not really cover; both are
fixable by enlarging or rebalancing the reference set.

---

## 11. Predicting continuous values (regression)

MolActAgent's regression flow shares the same intuition as the
classification flow — **find similar references, let the LLM
interpret what they imply** — but instead of a class label the
output is a *number* with a calibrated empirical interval. You
don't have to do anything special to switch tasks: MolActAgent
inspects your label column and auto-detects regression when the
column is continuous numeric.

### How regression actually works (CCR — Cross-Classification Regressor)

Under the hood, regression is reformulated as **two multi-class
classification problems** that share the same contrastive engine
classification mode uses:

1. **Phase 1 — target view.** MolActAgent splits the training
   targets into `N = ⌊√n⌋ + 1` equal-frequency bins (e.g. for
   n=24 reference samples, that's 5 bins of ≈5 samples each), then
   runs a multi-class classification over those bins to predict
   *which slice of the target distribution* the test sample lies
   in. Top-M=2 means even with many bins, only the two closest
   bins are contrasted in the LLM call.

2. **Phase 2 — feature view.** MolActAgent runs hierarchical /
   average-linkage clustering on the standardised features under
   cosine distance to produce `M = ⌊√n⌋ + 1` unsupervised feature
   clusters, then classifies the test sample against those. This
   asks "which chemistry / shape / behavioural cluster does this
   sample belong to?" — completely independent of target value.

3. **Selection.** From each phase's predicted class, MolActAgent
   picks the **top-2 nearest references** by the same contextual
   distance (feature similarity + neighbourhood structure) the
   classification path uses.

4. **Aggregation.** The picked references' actual target values
   are combined via a **confidence-weighted mean** — each
   reference's weight equals the source phase's classification
   confidence, so a high-confidence Phase 1 verdict counts more
   than a low-confidence Phase 2 verdict. References picked by
   *both* phases get summed weights (a vote of confidence from
   both views).

The key benefit: every per-test-sample prediction comes with both
a numeric verdict AND the full per-phase classification reasoning
(which bin/cluster, with what confidence, citing which features),
so the explanation report has rich evidence to work with.

### A small bundled example: penguin body mass

The repository ships a reproducible builder for a tiny Palmer
Penguins regression demo. Generate it once:

```bash
python scripts/prepare_penguins_subsample.py
```

This writes `examples/penguins/reference.csv` (24 penguins with
known body masses) and `examples/penguins/test.csv` (6 penguins
whose body mass we want to predict). The body mass values span
roughly 3300 g to 5800 g.

Validate the inputs:

```bash
molactagent validate \
  -r examples/penguins/reference.csv \
  -t examples/penguins/test.csv \
  --target-col body_mass_g --id-column sample_id --k 5 \
  --feature-descriptions examples/penguins/feature_descriptions.json \
  --background @examples/penguins/background.txt
```

You should see `Detected task: regression` plus the observed target
range. The flag is named `--target-col` for clarity, but `-l` /
`--label-col` work the same way; both refer to the same column
pointer mechanism.

Run the prediction:

```bash
molactagent run \
  -r examples/penguins/reference.csv \
  -t examples/penguins/test.csv \
  --target-col body_mass_g --id-column sample_id --k 5 \
  --feature-descriptions examples/penguins/feature_descriptions.json \
  --background @examples/penguins/background.txt \
  -o reports/penguins_first_run
```

### What the regression report shows

The `interpretability.html` for a regression run delivers:

- A coloured numeric badge (cool blue for low values through warm
  red for high) showing the predicted body mass, plus the explicit
  empirical interval (the min/max of the picked references' actual
  targets — e.g. `[3850 g, 4350 g]`).
- A **Reference Picks Across the Target Range** dot plot — one dot
  per picked reference's actual target value, plotted along the
  observed target-value axis with the final prediction marked.
  Tightly clustered dots mean the two phases converged on a
  coherent neighbourhood; widely-spread dots mean the picked
  references straddle different regions of target space (a useful
  uncertainty signal).
- A **Top Drivers** card where each cited feature carries a
  direction (`↑ increases` / `↓ decreases` / `mixed`) derived from
  the per-feature Pearson correlation with the target across the
  reference set — so you can read at a glance which features push
  predictions up vs down on this dataset.
- A "Where the Sample Sits" narrative section in the explanation
  that places the test sample inside the assembled reference
  landscape, citing the Phase 1 winning bin and the Phase 2
  winning cluster by name.

### Refuse-to-predict

If neither Phase 1 nor Phase 2 nominates a usable winning class
(both came back inconclusive with no per-class weighted vote),
MolActAgent returns `predicted_value: null` and the report displays
a `REFUSED` badge instead of fabricating a number from an empty
selection set. This is rare on real datasets — it usually means
the test sample is so far from any reference cluster that even
the contrastive classifier couldn't pick a side.

### Tuning regression cost

Two knobs control how many LLM calls a regression run makes per
test sample:

| Parameter | Meaning | Default |
| :--- | :--- | :--- |
| `ccr_k_per_class` (Python API only) | Top-k references picked from each phase's predicted class for the aggregation step. | 2 |
| `ccr_agent_k` (Python API only) | The inner `MolActAgent.k` used by each phase's contrastive classifier. With top-M=2 each phase makes `k × k` tuple LLM calls. | 2 |
| `ccr_n_mode` (Python API only) | How `N` (target bins) and `M` (feature clusters) scale with `n`. Default `'sqrt'` = `⌊√n⌋ + 1`; `'sqrt-floor'` = `⌊√n⌋`; `'sqrt-capped-10'` = old default; `'fixed'` honours `ccr_n_target_bins` / `ccr_n_feature_clusters` literally. | `'sqrt'` |
| `ccr_aggregator` (Python API only) | How the picked references' targets are combined. Default `'conf'` (confidence-weighted mean — best in our ablations); also `'mean'`, `'median'`, `'weighted'` (1/distance). | `'conf'` |

So a regression run costs roughly `2 × k × k = 8` LLM calls per
test sample at the defaults. Crucially, the cost grows with
`ccr_agent_k`, **not** with the number of bins/clusters — top-M=2
means even 32 classes only contrast two at a time.

### Performance

LOOCV against the bundled penguins reference set (n=24, body mass
in grams):

| Method | MAE | R² | Pearson r | Median AE |
|---|---:|---:|---:|---:|
| KNN3 | 369.10 | 0.523 | 0.736 | 300.00 |
| Ridge | 355.17 | 0.542 | 0.738 | 311.76 |
| Linear | 350.36 | 0.539 | 0.739 | 257.71 |
| RF | 360.01 | 0.540 | 0.746 | 334.50 |
| **MolActAgent (CCR)** | **353.89** | **0.532** | **0.755** | **242.70** |

MolActAgent matches the best ML baselines on aggregate error and
holds the **lowest median absolute error of any method tested** —
i.e. it's the most *typically* accurate regressor on this dataset
in addition to producing a per-sample explanation report.

---

## 12. Pipe-friendly mode (using MolActAgent inside other tools)

MolActAgent is a well-behaved Unix command. When you pipe its output to
another program (or redirect to a file), it automatically switches to
a structured format so downstream tools like `jq`, `awk`, `cut`, or
your own scripts can consume it directly. The aggregate
`interpretability.{md,html}` files are still written to disk if you
pass `-o`, but stdout becomes a pure data stream.

### What "pipe-friendly" gets you

| In a terminal (`stdout` is a TTY) | When piped (`\| jq`, `> file`, …) |
| :--- | :--- |
| **Pretty banner** with classification table, interpretability panel, "saved to" notes. | **Compact JSONL** — one JSON object per test sample, ready for `jq`. |
| Progress bar / log lines on stderr. | Same: progress + logs on stderr, **stdout is data only**. |

You can always force the format with `--format / -F`:

```
--format human   # pretty multi-section terminal view
--format json    # one big JSON object (everything: validation, outcomes, paths, llm_stats)
--format jsonl   # one compact JSON object per test sample (great for streaming)
--format tsv     # tab-separated verdicts table (sample_id, predicted_label, ...)
--format csv     # same but comma-separated
```

For `validate`, only `human` and `json` are meaningful (the legacy
`--json` flag is kept as an alias).

### Recipe 1 — pipe verdicts into `jq`

Print only the sample IDs and confidence values:

```bash
molactagent run \
  -r examples/iris_flowers/reference.csv \
  -t examples/iris_flowers/test.csv \
  --label-col species --id-column flower_id --k 3 \
  --feature-descriptions examples/iris_flowers/feature_descriptions.json \
  --background "@examples/iris_flowers/background.txt" \
  | jq -c '{id: .sample_id, label: .predicted_label, conf: .confidence, top: .key_drivers[0].feature}'
```

Output (one JSON object per line, one per test flower):

```json
{"id":"T1","label":"setosa","conf":0.95,"top":"petal_length_cm"}
{"id":"T2","label":"versicolor","conf":0.89,"top":"petal_length_cm"}
{"id":"T3","label":"setosa","conf":0.95,"top":"petal_width_cm"}
{"id":"T4","label":"versicolor","conf":0.84,"top":"petal_length_cm"}
{"id":"T5","label":"virginica","conf":0.95,"top":"petal_length_cm"}
{"id":"T6","label":"versicolor","conf":0.48,"top":"petal_length_cm"}
```

(T6 sits near the versicolor / virginica boundary, so its confidence is the lowest in the batch.)

> **No `-o` needed.** When you only want the data on stdout (and don't
> care about disk artefacts), drop the `-o` flag entirely. MolActAgent
> still computes everything in memory and emits it on stdout.

### Recipe 2 — pipe a CSV INTO MolActAgent (the `-` sentinel)

Use `-r -` or `-t -` to read CSV from stdin. This lets MolActAgent sit
in the middle of a chain:

```bash
# Take any CSV from a previous command and validate it
cat data/cleaned_reference.csv \
  | molactagent validate -r - -t test.csv \
      --label-col species --id-column flower_id \
      --k 3 \
  | jq '.is_valid, .summary.reference_class_counts'
```

> Only **one** of `-r` / `-t` may be `-` per invocation (they would
> otherwise consume the same byte stream). Use `--stdin-format tsv`
> if your stdin is tab-separated.

### Recipe 3 — TSV verdicts → `cut` / `awk`

```bash
molactagent run ... -F tsv \
  | cut -f1,2 \
  | column -t -s $'\t'
```

Output:

```
sample_id  predicted_label
T1         setosa
T2         versicolor
T3         setosa
T4         versicolor
T5         virginica
T6         versicolor
```

### Recipe 4 — silence the chrome with `--quiet` and `2>/dev/null`

`--quiet / -q` keeps the LLM's `INFO` log lines off stderr. Combine
with `2>/dev/null` for an absolutely silent stderr:

```bash
molactagent --quiet run ... 2>/dev/null | jq -c '.predicted_label' | sort | uniq -c
#  2 "setosa"
#  3 "versicolor"
#  1 "virginica"
```

### How auto-detection works

When `--format` is not given:

- For `validate`: stdout TTY → `human`, otherwise → `json`.
- For `run`: stdout TTY → `human`, otherwise → `jsonl`.

The detection looks at `sys.stdout.isatty()`. Any time you redirect
or pipe, you get structured output; in an interactive shell you get
the pretty banner. You never have to think about it.

### What ends up where

| Stream | Format = `human` | Format = `json/jsonl/tsv/csv` |
| :--- | :--- | :--- |
| **stdout** | Verdicts table + interpretability banner + "saved to" notes | Just the structured data (JSON / JSONL / TSV / CSV) |
| **stderr** | LLM status (`[LLM #N]`) + progress (`[1/4] T1 -> setosa`) | Same plus `wrote summary CSV: …` log lines |

So you can always do `… 2>/tmp/molactagent.log | downstream-tool` and
keep your data clean.

---

## 13. Bringing your own data

Use the iris folder as a **template**. Your own project folder needs
exactly these pieces:

```text
my_project/
├── reference.csv              # your labelled data
├── test.csv                   # your unlabelled data
├── feature_descriptions.json  # optional but recommended
├── background.txt             # optional but recommended
└── llm_config.jsonc           # your LLM API settings
```

Checklist before running:

1. Both CSVs have the **same feature columns** (order doesn't matter).
2. The `--label-col` appears in `reference.csv`. (It's fine to also
   include it in `test.csv`; MolActAgent will compute accuracy
   against it.)
3. The `--id-column` appears in both files; every ID is unique both
   within a file and across the two files combined.
4. `reference.csv` has **at least two** distinct labels with at
   least one row each (more rows per class is better — the
   confidence calibration's `sample_size` component caps out around
   10 per class). The system supports any number of classes; it
   warns above ~10 classes that the LLM may strain.
5. All feature columns are numeric. Anything that can't be parsed
   as a number will be flagged by `validate`.
6. If you provide `feature_descriptions.json`, it must describe
   every column (features + label).

Then swap in your paths:

```bash
molactagent validate -r my_project/reference.csv -t my_project/test.csv \
  --label-col my_label --id-column my_id --k 3 \
  --feature-descriptions my_project/feature_descriptions.json \
  --background "@my_project/background.txt"

molactagent run -r my_project/reference.csv -t my_project/test.csv \
  --label-col my_label --id-column my_id --k 3 \
  --feature-descriptions my_project/feature_descriptions.json \
  --background "@my_project/background.txt" \
  --llm-config my_project/llm_config.jsonc \
  -o my_project/outputs
```

Add `--per-sample` if you want individual reports as well.

Other useful knobs:

- `--positive-class LABEL` — *deprecated.* Accepted for backward
  compatibility but ignored under the multi-class pipeline; the
  system treats every class equally now.
- `--summary-json` — also write `summary.json`.
- `--no-interpretability` — skip the aggregate Markdown/HTML.
- `--max-workers N` — raise concurrent LLM calls (default 8). Keep
  this low for cheap providers.
- `--progress/--no-progress` — show a live progress bar per sample.

---

## 14. The full CLI reference

Run `molactagent --help` for a list of subcommands and
`molactagent <cmd> --help` for per-command detail.

### `molactagent validate`

Checks the inputs without calling the LLM.

| Flag | Required? | Meaning |
| :--- | :--- | :--- |
| `-r, --reference PATH \| -` | yes | reference CSV/TSV/Excel/Parquet/JSON file. Use `-` to read CSV from stdin. |
| `-t, --test PATH \| -` | yes | test file, same accepted formats. `-` for stdin (only ONE of -r / -t may be `-`). |
| `--label-col NAME` | yes | label column in the reference file |
| `--id-column NAME` | yes | unique-ID column present in BOTH files |
| `--k INT` | no (default 3) | neighbours per class |
| `--feature-descriptions PATH` | no | JSON/YAML dict, must cover every column |
| `--background STR or @PATH` | no | research context text (or `@path` to a file) |
| `--positive-class LABEL` | no | *deprecated;* accepted for back-compat but ignored under the multi-class pipeline |
| `--format / -F human \| json` | no | stdout format. Auto: human if TTY, json if piped. |
| `--json` | no | deprecated alias for `--format json`. |
| `--stdin-format csv \| tsv` | no (default csv) | format used when `-r` / `-t` is `-` |

Exit codes: `0` = valid, `1` = validation failure, `2` = I/O failure
(file missing, unreadable), `3` = misuse.

### `molactagent run`

Does a full classification.

Everything `validate` accepts, **plus**:

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `-o, --output-dir PATH` | (optional) | where to save reports. Omit for "stdout-only" middleware mode. |
| `--llm-config PATH` | auto-discover | path to your `.jsonc/.json/.yaml/.txt` config |
| `--profile NAME` | last block | pick a named profile when several are in the file |
| `--api-key / --base-url / --model-name` | from config | override individual settings |
| `--temperature F`, `--max-tokens N`, `--timeout S`, `--max-retries N` | from config | fine-tune the LLM |
| `--max-workers N` | 8 | parallel per-tuple LLM calls per sample |
| `--format / -F human\|json\|jsonl\|tsv\|csv` | auto | stdout format. Auto: human if TTY, jsonl if piped. |
| `--per-sample` | OFF | generate per-sample HTML+MD+JSON (one extra LLM call per sample) |
| `--per-sample-format LIST` | `html,md,json` | when `--per-sample` is on, which formats to write |
| `--summary-json` | OFF | also write `summary.json` |
| `--no-summary-csv` | (CSV is on) | skip `summary.csv` |
| `--no-interpretability` | (interp is on) | skip `interpretability.{md,html}` |
| `--no-progress` | (progress is on) | silence the progress bar |
| `--raise-on-validation-error` | OFF | fail hard instead of writing summary + non-zero exit |

### `molactagent init-config`

Drops a fresh `llm_config.jsonc` template next to you:

```bash
molactagent init-config --path my_project/llm_config.jsonc
```

Pass `--format txt` for the legacy `LLM_API.txt` format, or
`--overwrite` to replace an existing file.

---

## 15. Common error messages

### "id_column `X` is missing from the reference dataset"

Make sure every row (in BOTH files) has that exact column name.
Column names are case-sensitive. The validator prints the columns it
*did* find to help.

### "id_column `X` value(s) [...] appear in BOTH the reference and test datasets"

IDs must be globally unique so per-sample reports don't overwrite
each other. Rename the duplicates (e.g. prefix reference rows with
`R_` and test rows with `T_`).

### Multi-class warnings ("11 distinct classes" etc.)

MolActAgent supports any number of classes (≥ 2). When the reference
set contains more than ~10 distinct classes the validator emits a
non-blocking warning that the LLM may struggle to weigh many
candidates per call; consider grouping rare classes if the warning
matters for your use case.

### "Feature descriptions are missing for columns: [...]"

If you provide `--feature-descriptions`, every column (features +
label) must have an entry. Either complete the dictionary or omit
the flag entirely.

### "LLM CONFIGURATION ERROR" banner

MolActAgent runs a **preflight ping** against your LLM before spending
tokens. If that fails you get a clearly formatted banner:

```text
====================================================================
LLM CONFIGURATION ERROR
====================================================================
LLM endpoint is unreachable.
Config file checked: /path/to/llm_config.jsonc
Action items:
  1. Open llm_config.jsonc (or run `molactagent init-config` to create one).
  2. Make sure exactly ONE provider block is uncommented and that
     api_key, base_url, and model_name are set correctly.
  3. Verify the API key has not expired and that you have internet access.
  * Could not reach https://api.example.com/v1. Check your internet
    connection and any VPN / firewall rules.
====================================================================
```

The banner classifies the failure (auth, model, network, or unknown)
and gives actionable next steps. Exit code is `3` for any LLM-side
problem.

### "non-numeric value"

A feature column has text where there should be numbers. Look at the
offending column and clean / drop the bad rows. Remember: strings
are fine for the `id_column`, but every other column must be numeric
(or at least coercible with `pd.to_numeric`).

---

## 16. Using MolActAgent from Python

MolActAgent's Python API is **layered**. Pick the layer that matches
your familiarity:

| If you... | use |
|---|---|
| Want one all-in-one call that handles validation, prediction, and report writing | `ca.run_classification` / `ca.run_regression` (this section) |
| Want sklearn-style estimators that expose `predict_proba`, calibrated confidence, and slot into `cross_val_score`, GridSearchCV, etc. | `ca.MolActAgentClassifier` / `ca.MolActAgentRegressor` (§16.A below) |
| Want Scanpy-style discoverable namespaces — chemistry preprocessing, LOOCV, baseline comparison, calibration | `ca.pp` / `ca.tl` / `ca.metrics` / `ca.datasets` (§16.B and §16.C below) |
| Want maximum control: assemble prompts, drive the LLM client yourself, render reports | `ca.MolActAgent`, `ca.CrossClassificationRegressor`, `ca.LLMClient`, `ca.render_html_report` |

The high-level all-in-one path:

```python
from molactagent import run_classification, load_llm_config, build_llm_client

cfg = load_llm_config()  # auto-discovers llm_config.jsonc at project root
client = build_llm_client(cfg)

result = run_classification(
    reference="examples/iris_flowers/reference.csv",
    test="examples/iris_flowers/test.csv",
    label_column="species",
    id_column="flower_id",
    k=3,
    feature_descriptions="examples/iris_flowers/feature_descriptions.json",
    background="@examples/iris_flowers/background.txt",
    output_dir="examples/iris_flowers/outputs",
    llm_client=client,
    # Same defaults as the CLI:
    per_sample_reports=False,         # set True for per-flower HTML+MD+JSON
    per_sample_formats=("html", "md", "json"),
    write_summary_csv=True,
    write_summary_json=False,         # set True to also dump summary.json
    write_interpretability=True,      # interpretability.{md,html}
)

print(f"Validation OK? {result.validation.is_valid}")
print(f"Interpretability MD: {result.interpretability_md_path}")
print(f"Interpretability HTML: {result.interpretability_html_path}")
for o in result.outcomes:
    print(o.sample_id, o.predicted_label, f"{o.confidence:.0%}")
```

You can also pass in pandas `DataFrame` objects directly (no files
needed). The returned `PipelineOutput` has:

- `.outcomes` — a list of `SampleOutcome` objects with `sample_id`,
  `predicted_label`, `confidence`, `raw_result`, and (when
  `per_sample_reports=True`) `explanation_md`,
  `explanation_html_path`, `explanation_md_path`, `raw_json_path`.
  For regression runs each `SampleOutcome` additionally carries
  `predicted_value` (float) and `interval` (a `(lo, hi)` tuple).
- `.validation` — the full `ValidationResult`.
- `.summary_csv_path`, `.summary_json_path` — file paths when those
  formats are enabled.
- `.interpretability_text` — the rendered banner text.
- `.interpretability_md_path`, `.interpretability_html_path` — file
  paths when `write_interpretability=True`.
- `.llm_stats` — total LLM calls, tokens, wall time.

### Regression from Python

The same pattern works for regression — just call `run_regression`
and point at a continuous-numeric column:

```python
from molactagent import run_regression, load_llm_config, build_llm_client

cfg = load_llm_config()  # auto-discovers llm_config.jsonc at project root
client = build_llm_client(cfg)

result = run_regression(
    reference="examples/penguins/reference.csv",
    test="examples/penguins/test.csv",
    target_column="body_mass_g",
    id_column="sample_id",
    feature_descriptions="examples/penguins/feature_descriptions.json",
    background="@examples/penguins/background.txt",
    output_dir="examples/penguins/outputs",
    llm_client=client,
    per_sample_reports=True,
)

for o in result.outcomes:
    if o.predicted_value is None:
        print(f"{o.sample_id}: REFUSED")
    else:
        lo, hi = o.interval
        print(
            f"{o.sample_id}: predicted={o.predicted_value:.0f} g  "
            f"interval=[{lo:.0f}, {hi:.0f}]  "
            f"confidence={o.confidence:.0%}"
        )
```

The `raw_result` carries the full CCR diagnostic dict if you want
to inspect which target bin / feature cluster won each phase, what
references were picked, what each phase's classification confidence
was, and which features the per-feature → target correlation
identified as drivers — see the `phase_target`, `phase_feature`,
`selected_indices`, `selected_targets`, and `driver_attribution`
keys.

If you want to use the regression backend directly without the
`MolActAgent` wrapper:

```python
from molactagent import CrossClassificationRegressor, build_llm_client, load_llm_config
import numpy as np

llm = build_llm_client(load_llm_config())  # auto-discovery
X_train = np.array([[34.0, 17.1, 185.0], [44.1, 18.0, 210.0], ...])
y_train = np.array([3400.0, 4000.0, ...])

reg = CrossClassificationRegressor(
    llm_client=llm,
    feature_names=["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
    background="Predict body mass of Palmer penguins from morphology.",
    # Optional knobs (defaults shown):
    n_mode="sqrt",            # N = M = floor(sqrt(n)) + 1
    k_per_class=2,            # top-2 nearest in each predicted class
    agent_k=2,                # MolActAgent k for each phase
    aggregator="conf",        # confidence-weighted mean
).fit(X_train, y_train)

result = reg.predict_one(np.array([42.0, 18.5, 195.0]), test_id=0)
print(result["predicted_value"], result["interval"], result["confidence"])
```

### 16.1 — sklearn-style estimators (`MolActAgentClassifier`, `MolActAgentRegressor`)

If you've used scikit-learn, this is the API you'll feel at home
with. Both estimators implement `fit` / `predict` /
`predict_proba` / `score` — they slot into `cross_val_score`,
`GridSearchCV`, sklearn `Pipeline`s, and any other consumer that
accepts `BaseEstimator`-shaped objects.

```python
import molactagent as ca

cfg = ca.load_llm_config()  # auto-discovers llm_config.jsonc at project root

# One-call dataset loader (handles validation + preparation).
data = ca.datasets.load_iris_flowers()

clf = ca.MolActAgentClassifier(
    llm_config=cfg,
    feature_names=data.feature_names,
    feature_descriptions=data.feature_descriptions,
    background=data.background,
    k=3,
)
clf.fit(data.X_train, data.y_train)

# Three different views of the same predictions:
labels         = clf.predict(data.X_test)                  # ['setosa', ...]
probabilities  = clf.predict_proba(data.X_test)            # (n, n_classes) — sums to 1 per row
labels, conf   = clf.predict_with_confidence(data.X_test)  # calibrated confidence (NOT proba.max!)
full           = clf.predict_full(data.X_test)             # rich ClassificationResult per sample
score          = clf.score(data.X_test, data.y_test)      # accuracy

# `ClassificationResult` exposes everything the engine produced:
res = full[0]
print(res.label, res.confidence)
print(res.proba())                    # {'setosa': 0.0, 'versicolor': 0.78, 'virginica': 0.22}
print(res.driver_attribution)         # cross-tuple per-feature attribution
print(res.tuple_results)              # raw per-tuple records
print(res.test_feature_values)        # the test row's feature values by name

# After predict, the estimator carries cohort-level introspection:
print(clf.driver_attribution_)        # cross-batch cited-driver summary
print(clf.llm_stats_)                 # tokens / calls / time spent
```

`predict_proba` and `predict_with_confidence` answer **different
questions** and are exposed as separate axes by design:

- `predict_proba` rows are normalised weighted-vote shares per
  candidate class. `proba.max(axis=1)` is "of the candidates
  considered, how much weighted evidence backs the winner?".
- `predict_with_confidence` returns the engine's calibrated
  confidence (margin × sample-size × coverage; see
  `docs/design.md` §18.2). It additionally discounts for small
  reference pools and inconclusive-tuple rates.

The regression counterpart:

```python
data = ca.datasets.load_penguins()
reg = ca.MolActAgentRegressor(
    llm_config=cfg,
    target_name="body_mass_g",
    feature_names=data.feature_names,
    feature_descriptions=data.feature_descriptions,
    background=data.background,
    k=5,
)
reg.fit(data.X_train, data.y_train)

predictions = reg.predict(data.X_test)                 # NaN for refusals
preds, lo, hi = reg.predict_with_interval(data.X_test) # honest [min, max] bracket
preds, conf = reg.predict_with_confidence(data.X_test)
r2 = reg.score(data.X_test, data.y_test)               # R²; refusals masked out
```

### 16.2 — Compound chemistry: SMILES → MolActAgent in one call

If your data is a SMILES column plus a label column, `ca.pp` does
the upstream work that used to require running
`scripts/prepare_features.py` first.

```python
import pandas as pd
import molactagent as ca

df = pd.read_csv("my_compounds.csv")  # columns: smiles, activity

features = ca.pp.smiles_to_features(
    df["smiles"], y=df["activity"],
    descriptors=True,    # 26 RDKit descriptors (MolWt, MolLogP, TPSA, ...)
    maccs=True,          # 167 MACCS keys with full plain-English descriptions
    morgan=False,        # set True to also append a Morgan / ECFP block
)
print(f"X shape: {features.X.shape}")
print(f"feature_descriptions has {len(features.feature_descriptions)} entries")

# Bundled domain backgrounds (verbatim from the project's benchmarks):
bg = ca.pp.chemistry.EGFR_BACKGROUND      # for EGFR inhibitor classification
# Other options: ANTITB_BACKGROUND, BSLT_BACKGROUND

clf = ca.MolActAgentClassifier(
    llm_config=cfg,
    feature_names=features.feature_names,
    feature_descriptions=features.feature_descriptions,
    background=bg,
    k=3,
).fit(features.X, features.y)
```

For chemistry-correct held-out evaluation, use scaffold splitting:

```python
split = ca.pp.scaffold_split(
    features.smiles, test_size=0.2, random_state=42,
)
clf.fit(features.X[split.train_idx], features.y[split.train_idx])
y_pred = clf.predict(features.X[split.test_idx])
```

The `scaffold_ids` field plugs into sklearn's `GroupKFold`:

```python
from sklearn.model_selection import GroupKFold
groups = ca.pp.scaffold_split(features.smiles).scaffold_ids
for train_idx, test_idx in GroupKFold(n_splits=5).split(features.X, features.y, groups):
    ...
```

### 16.3 — Downstream analysis: LOOCV, baselines, calibration, attribution

Everything you need for benchmarking and diagnostics lives under
`ca.tl`. None of these functions touches the engine internals —
they orchestrate predictions and post-process them.

```python
# Leave-one-out CV against the LLM-backed estimator.
result = ca.tl.loocv(clf, features.X, features.y, fold_workers=4)
print(result.summary())                # full metric table
print(result.metrics.to_table())       # just the metrics
result.to_dataframe()                  # tidy per-fold table

# Head-to-head against ML baselines (KNN, RF, LR, SVM, ...). The
# default baseline list is curated for the task; pass `baselines='all'`
# to run every baseline in the catalogue.
comparison = ca.tl.compare_baselines(
    molactagent_estimator=clf,
    X=features.X, y=features.y,
    baselines=["KNN3", "RF", "LR"],
    cv="loocv",
    fold_workers=4,
)
print(comparison.metrics_table())      # exact same fixed-width text the legacy script produced
print(comparison.best("mcc"))          # method name with highest MCC

# Confidence calibration (Brier, ECE, reliability curve).
cal = ca.tl.confidence_calibration(fold_records=result.fold_records)
print(cal.summary())

# Cohort-level driver attribution: which features mattered ACROSS
# the test set?
agg = ca.tl.aggregate_drivers(result.fold_records)
print(agg.to_dataframe().head(10))
print(agg.per_class_top_features(n=6))
```

Every result object has `to_dict()` for JSON serialisation and
`to_dataframe()` for notebook use.

---

## 17. FAQ

**Q: Why does MolActAgent take ~30 seconds per sample?**
For **classification**, each test sample runs `k^M` tuple
comparisons, where M is how many classes the adaptive top-M
selection picked for that sample (typically 2 — so 9 calls at
`k=3`; 27 calls if M=3). For **regression**, each test sample runs
the contrastive engine *twice* (once for the target-bin Phase 1,
once for the feature-cluster Phase 2), so the cost is roughly
`2 × ccr_agent_k × ccr_agent_k = 8` calls at the defaults. Each
tuple is a separate LLM call. With `--per-sample` there's also one
explanation call per sample. `--max-workers` parallelises them; a
larger `max_tokens` setting slows the provider down. DeepSeek is
slower than OpenAI but cheaper.

**Q: How many reference rows do I need?**
The minimum is 2 per class; in practice 8+ per class gives a
noticeably more stable verdict (the `sample_size` confidence score
in the interpretability panel caps out around 20 per class).

**Q: How do I choose `k`?**
`k=3` is a good default. Larger `k` = more comparisons =
slower and more expensive, but better coverage of feature space.
Smaller `k` = cheaper but less robust.

**Q: Can I get probabilities instead of a hard label?**
The `confidence` field in every output file is calibrated from
margin, sample size, and coverage — treat it as a soft probability.
With `--per-sample` the `T<n>.json` files additionally contain the
raw per-tuple confidences and the per-class `vote_summary.per_class`
breakdown, so you can derive any aggregate score you want.

**Q: How do I use a local LLM (Ollama / LM Studio)?**
Uncomment the "Local self-hosted" block in `llm_config.jsonc`,
typically with `base_url: http://localhost:11434/v1` and
`model_name: llama3.1:8b` (or whatever you're serving).
MolActAgent talks to anything that speaks the OpenAI chat
completions API.

**Q: Does MolActAgent send my data to the internet?**
Yes — every feature value of every row is embedded in a prompt and
sent to the LLM you configured. If your data is sensitive, point
MolActAgent at a locally-hosted model, or review your LLM
provider's data retention policy.

**Q: Can I stop and resume?**
Not directly, but each per-sample artefact (when `--per-sample` is
on) is written to disk as soon as it finishes. If you crash
mid-run you still have the per-sample files that completed.
Re-running the command starts over.

**Q: Where did `interpretability.txt` go?**
It's now `interpretability.md` (Markdown) — and we added a sister
`interpretability.html` so you can hand the results to a
non-technical reader. Both replace the old plain-text file.

**Q: Can I use MolActAgent inside a shell pipeline (`… | molactagent | …`)?**
Yes. MolActAgent is a well-behaved Unix tool: when stdout is piped or
redirected it auto-switches to JSONL (or JSON for `validate`) so
`jq`, `awk`, `cut`, and friends can consume it directly. Force the
format with `--format / -F`. Use `-r -` or `-t -` to read CSV from
stdin. See [Section 12 — Pipe-friendly mode](#12-pipe-friendly-mode-using-molactagent-inside-other-tools)
for ready-to-copy recipes.

**Q: Where did `--no-explanations` go?**
It's still accepted (with a deprecation note) but maps to
"don't generate per-sample reports" — which is now the default. The
new flag name is `--per-sample` (off by default; opt in to enable).

---

Happy classifying! If you find an edge case the tutorial doesn't
cover, please open an issue — we'd love to hear about it.
