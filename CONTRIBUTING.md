# Contributing to MolActAgent

Issues and pull requests are welcome. Please keep changes focused, explain the scientific or user-facing motivation, and add or update tests for behavioural changes.

## Development setup

The reproducible environment is:

```bash
conda env create -f environment.yml
conda activate molactagent_env
pip install -e .
```

Run the offline test suite before opening a pull request:

```bash
python -m pytest -q
```

Tests that call a real LLM are opt-in. Never commit API keys, local credential files, generated reports, or private datasets.
