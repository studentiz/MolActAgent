"""Regression tests for the public MolActAgent naming contract."""

from __future__ import annotations

import re
from pathlib import Path


_TEXT_SUFFIXES = {
    ".cff",
    ".html",
    ".json",
    ".jsonc",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}


def _project_text_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _TEXT_SUFFIXES
        and ".git" not in path.parts
        and "data" not in path.relative_to(root).parts
    )


def test_no_legacy_brand_aliases_remain():
    """The renamed project must not publish the old brand or its aliases."""
    root = Path(__file__).resolve().parent.parent
    legacy_brand = "Class" + "Agent"
    legacy_lower = legacy_brand.lower()
    legacy_snake = "class" + "_agent"
    legacy_dash = "class" + "-agent"
    forbidden = (
        legacy_brand,
        legacy_lower,
        legacy_snake,
        legacy_dash,
        r"import\s+molactagent\s+as\s+ca\b",
        r"\bca\.(?:pp|tl|metrics|datasets|run_|MolActAgent|CrossClassificationRegressor|LLMClient)\b",
        r"\bca_(?:metrics|details|preds|records)\b",
        r"\bMolActAgent\s+as\s+CA\b",
        r"[\"']CA\s+\(",
    )

    failures: list[str] = []
    for path in _project_text_files():
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in {legacy_brand, legacy_lower, legacy_snake, legacy_dash}:
                matched = pattern in text
            else:
                matched = re.search(pattern, text) is not None
            if matched:
                failures.append(f"{path.relative_to(root)}: {pattern}")
    assert not failures, "Legacy MolActAgent naming found:\n" + "\n".join(failures)


def test_package_aliases_are_consistently_ma():
    """Any documented package alias must match MolActAgent's ``ma`` convention."""
    alias_pattern = re.compile(r"import\s+molactagent\s+as\s+([A-Za-z_]\w*)")
    aliases = {
        alias
        for path in _project_text_files()
        for alias in alias_pattern.findall(path.read_text(encoding="utf-8"))
    }
    assert aliases <= {"ma"}, f"Unexpected molactagent aliases: {sorted(aliases)}"
