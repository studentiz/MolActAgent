"""Tests for ``molactagent.config`` — LLM API configuration loading."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from molactagent.config import (
    DEFAULT_CONFIG_FILENAMES,
    LLMConfig,
    LLMConfigError,
    build_llm_client,
    load_llm_config,
    write_example_config,
)
from molactagent.config import _strip_jsonc_comments  # type: ignore


# ── load_llm_config ──────────────────────────────────────────────────


class TestLoadFromText:
    def _write_legacy(self, path: pathlib.Path, body: str) -> pathlib.Path:
        path.write_text(body, encoding="utf-8")
        return path

    def test_single_block_full_width_colon(self, tmp_path):
        p = self._write_legacy(
            tmp_path / "LLM_API.txt",
            "apiKey：sk-A\nbase_url：https://x.example/v1\nModelname：m1\n",
        )
        cfg = load_llm_config(p)
        assert cfg.api_key == "sk-A"
        assert cfg.base_url == "https://x.example/v1"
        assert cfg.model_name == "m1"
        assert cfg.source == str(p)

    def test_single_block_ascii_colon(self, tmp_path):
        p = self._write_legacy(
            tmp_path / "LLM_API.txt",
            "apiKey: sk-A\nbase_url: https://x.example/v1\nModelname: m1\n",
        )
        cfg = load_llm_config(p)
        assert cfg.api_key == "sk-A"

    def test_two_blocks_picks_last_by_default(self, tmp_path):
        p = self._write_legacy(
            tmp_path / "LLM_API.txt",
            "apiKey: sk-A\nbase_url: https://a/v1\nModelname: deepseek-chat\n"
            "\n"
            "apiKey: sk-B\nbase_url: https://b/v1\nModelname: qwen-max\n",
        )
        cfg = load_llm_config(p)
        assert cfg.api_key == "sk-B"
        assert cfg.model_name == "qwen-max"

    def test_two_blocks_pick_deepseek(self, tmp_path):
        p = self._write_legacy(
            tmp_path / "LLM_API.txt",
            "apiKey: sk-A\nbase_url: https://api.deepseek.com\nModelname: deepseek-chat\n"
            "\n"
            "apiKey: sk-B\nbase_url: https://b/v1\nModelname: qwen-max\n",
        )
        cfg = load_llm_config(p, profile="deepseek")
        assert cfg.api_key == "sk-A"

    def test_two_blocks_pick_qwen(self, tmp_path):
        p = self._write_legacy(
            tmp_path / "LLM_API.txt",
            "apiKey: sk-A\nbase_url: https://api.deepseek.com\nModelname: deepseek-chat\n"
            "\n"
            "apiKey: sk-B\nbase_url: https://dashscope.aliyuncs.com\nModelname: qwen-max\n",
        )
        cfg = load_llm_config(p, profile="qwen")
        assert cfg.api_key == "sk-B"

    def test_unknown_profile_falls_back(self, tmp_path):
        p = self._write_legacy(
            tmp_path / "LLM_API.txt",
            "apiKey: sk-A\nbase_url: https://x/v1\nModelname: deepseek-chat\n",
        )
        with pytest.raises(ValueError, match="Unknown profile"):
            load_llm_config(p, profile="weirdname")

    def test_file_missing_when_explicit(self, tmp_path):
        """Missing config path now raises the user-friendly LLMConfigError,
        which inherits from ValueError and carries actionable hints."""
        with pytest.raises(LLMConfigError) as e:
            load_llm_config(tmp_path / "nope.txt")
        assert "not found" in str(e.value)
        msg = e.value.user_facing_message()
        assert "llm_config.jsonc" in msg  # actionable hint

    def test_missing_required_fields(self, tmp_path):
        p = tmp_path / "LLM_API.txt"
        p.write_text("apiKey: sk-A\n")
        with pytest.raises(LLMConfigError, match="required field"):
            load_llm_config(p)

    def test_comments_and_blank_lines(self, tmp_path):
        p = tmp_path / "LLM_API.txt"
        p.write_text(
            "# comment\n"
            "apiKey: sk-A\n"
            "# another\n"
            "base_url: https://x/v1\n"
            "Modelname: m1\n",
        )
        cfg = load_llm_config(p)
        assert cfg.api_key == "sk-A"


class TestLoadFromJson:
    def test_single_object(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "api_key": "sk-A", "base_url": "https://x", "model_name": "m1",
            "temperature": 0.2, "max_tokens": 1500,
        }))
        cfg = load_llm_config(p)
        assert cfg.api_key == "sk-A"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 1500

    def test_profiles_array(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "profiles": [
                {"api_key": "k1", "base_url": "u1", "model_name": "deepseek-chat"},
                {"api_key": "k2", "base_url": "u2", "model_name": "qwen-max"},
            ]
        }))
        cfg = load_llm_config(p, profile="qwen")
        assert cfg.api_key == "k2"

    def test_alias_keys(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "apiKey": "sk-A", "baseUrl": "https://x", "Modelname": "m1",
        }))
        cfg = load_llm_config(p)
        assert cfg.api_key == "sk-A"
        assert cfg.base_url == "https://x"
        assert cfg.model_name == "m1"

    def test_bad_json(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text("{not json")
        with pytest.raises(ValueError):
            load_llm_config(p)

    def test_bad_temperature_value(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "api_key": "k", "base_url": "u", "model_name": "m",
            "temperature": "very hot",
        }))
        with pytest.raises(ValueError, match="temperature"):
            load_llm_config(p)


class TestEnvAndOverrides:
    def test_env_provides_required(self, tmp_path, monkeypatch):
        # No file, must use env
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "envkey")
        monkeypatch.setenv("MOLACTAGENT_BASE_URL", "envurl")
        monkeypatch.setenv("MOLACTAGENT_MODEL", "envmodel")
        cfg = load_llm_config(None)
        assert cfg.api_key == "envkey"
        assert cfg.base_url == "envurl"
        assert cfg.model_name == "envmodel"

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        p = tmp_path / "LLM_API.txt"
        p.write_text("apiKey: filekey\nbase_url: fileurl\nModelname: filemodel\n")
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "envkey")
        cfg = load_llm_config(p)
        assert cfg.api_key == "envkey"
        assert cfg.base_url == "fileurl"

    def test_kwargs_override_everything(self, tmp_path, monkeypatch):
        p = tmp_path / "LLM_API.txt"
        p.write_text("apiKey: filekey\nbase_url: fileurl\nModelname: filemodel\n")
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "envkey")
        cfg = load_llm_config(p, api_key="cli-key", model_name="cli-model")
        assert cfg.api_key == "cli-key"
        assert cfg.model_name == "cli-model"
        assert cfg.base_url == "fileurl"

    def test_no_source_at_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MOLACTAGENT_API_KEY", raising=False)
        monkeypatch.delenv("MOLACTAGENT_BASE_URL", raising=False)
        monkeypatch.delenv("MOLACTAGENT_MODEL", raising=False)
        with pytest.raises(LLMConfigError, match="required field"):
            load_llm_config(None)

    def test_auto_discovery_walks_up(self, tmp_path, monkeypatch):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        p = tmp_path / "LLM_API.txt"
        p.write_text("apiKey: K\nbase_url: U\nModelname: M\n")
        monkeypatch.chdir(deep)
        cfg = load_llm_config(None)
        assert cfg.api_key == "K"


class TestSourceLabel:
    """The ``source`` field should honestly list every layer (file,
    env vars, kwargs) that fed the final config."""

    def _block_other_envs(self, monkeypatch):
        for k in (
            "MOLACTAGENT_API_KEY", "MOLACTAGENT_BASE_URL",
            "MOLACTAGENT_MODEL", "MOLACTAGENT_TEMPERATURE",
            "MOLACTAGENT_TIMEOUT",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_file_only(self, tmp_path, monkeypatch):
        self._block_other_envs(monkeypatch)
        p = tmp_path / "LLM_API.txt"
        p.write_text("apiKey: k\nbase_url: u\nModelname: m\n")
        cfg = load_llm_config(p)
        assert cfg.source == str(p)

    def test_env_only(self, tmp_path, monkeypatch):
        self._block_other_envs(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "ek")
        monkeypatch.setenv("MOLACTAGENT_BASE_URL", "eu")
        monkeypatch.setenv("MOLACTAGENT_MODEL", "em")
        cfg = load_llm_config(None)
        assert cfg.source == "env"

    def test_kwargs_only(self, tmp_path, monkeypatch):
        self._block_other_envs(monkeypatch)
        monkeypatch.chdir(tmp_path)
        cfg = load_llm_config(
            None,
            api_key="ck", base_url="cu", model_name="cm",
        )
        assert cfg.source == "kwargs"

    def test_file_plus_env(self, tmp_path, monkeypatch):
        self._block_other_envs(monkeypatch)
        p = tmp_path / "LLM_API.txt"
        p.write_text("apiKey: k\nbase_url: u\nModelname: m\n")
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "ek")
        cfg = load_llm_config(p)
        # File path is preserved, " + env" appended so debugging is easy.
        assert cfg.source == f"{p} + env"

    def test_file_plus_env_plus_kwargs(self, tmp_path, monkeypatch):
        self._block_other_envs(monkeypatch)
        p = tmp_path / "LLM_API.txt"
        p.write_text("apiKey: k\nbase_url: u\nModelname: m\n")
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "ek")
        cfg = load_llm_config(p, model_name="cm")
        assert cfg.source == f"{p} + env + kwargs"


# ── write_example_config ─────────────────────────────────────────────


class TestExampleConfig:
    def test_write_txt(self, tmp_path):
        p = write_example_config(tmp_path / "LLM_API.txt", fmt="txt")
        assert p.exists()
        assert "apiKey" in p.read_text()

    def test_write_jsonc_default_extension(self, tmp_path):
        """The starter JSONC file has exactly ONE active provider block and
        several commented-out vendor templates inside ``/* ... */``.

        ``write_example_config`` defaults to ``fmt='jsonc'`` so callers
        only have to specify a path."""
        p = write_example_config(tmp_path / "llm_config.jsonc")
        text = p.read_text()
        assert "//" in text
        assert "REPLACE-WITH-YOUR-OWN-KEY" in text

    def test_write_jsonc_placeholder_rejected(self, tmp_path):
        """A freshly-written starter config refuses to load as-is because its
        api_key is a placeholder — that's what surfaces the LLM-config banner
        to the user."""
        p = write_example_config(tmp_path / "llm_config.jsonc")
        with pytest.raises(LLMConfigError):
            load_llm_config(p)

    def test_write_jsonc_commented_templates(self, tmp_path):
        """The commented-out vendor block enumerates the supported providers."""
        p = write_example_config(tmp_path / "llm_config.jsonc")
        text = p.read_text()
        for vendor in ("DeepSeek", "Qwen", "OpenAI", "Gemini", "Mistral"):
            assert vendor in text

    def test_json_fmt_is_alias_for_jsonc(self, tmp_path):
        """``fmt='json'`` writes the same JSONC body — kept as an alias."""
        p = write_example_config(tmp_path / "llm_config.json", fmt="json")
        text = p.read_text()
        assert "//" in text
        assert "REPLACE-WITH-YOUR-OWN-KEY" in text

    def test_no_overwrite(self, tmp_path):
        p = tmp_path / "LLM_API.txt"
        p.write_text("existing")
        with pytest.raises(FileExistsError):
            write_example_config(p, fmt="txt")
        # overwrite=True works
        write_example_config(p, fmt="txt", overwrite=True)
        assert "apiKey" in p.read_text()

    def test_unknown_format(self, tmp_path):
        with pytest.raises(ValueError):
            write_example_config(tmp_path / "x.xml", fmt="xml")


# ── build_llm_client ────────────────────────────────────────────────


class TestJSONCParsing:
    """Unit tests for ``_strip_jsonc_comments`` + end-to-end JSONC load."""

    def test_strip_line_comments(self):
        text = '{\n  "a": 1, // trailing comment\n  "b": 2 // eol\n}'
        cleaned = _strip_jsonc_comments(text)
        data = json.loads(cleaned)
        assert data == {"a": 1, "b": 2}

    def test_strip_block_comments(self):
        text = '{\n  /* block before */\n  "a": 1,\n  "b": /* inline */ 2\n}'
        cleaned = _strip_jsonc_comments(text)
        data = json.loads(cleaned)
        assert data == {"a": 1, "b": 2}

    def test_preserves_string_slashes(self):
        """URL forward slashes inside a string MUST not be treated as comments."""
        text = '{"base_url": "https://api.example.com/v1"}'
        cleaned = _strip_jsonc_comments(text)
        data = json.loads(cleaned)
        assert data["base_url"] == "https://api.example.com/v1"

    def test_preserves_escape_sequences_in_strings(self):
        text = r'{"msg": "he said \"hi\" // not a comment"}'
        cleaned = _strip_jsonc_comments(text)
        data = json.loads(cleaned)
        assert data["msg"] == 'he said "hi" // not a comment'

    def test_unterminated_block_is_treated_as_end_of_file(self):
        text = '{"a": 1} /* never closed'
        cleaned = _strip_jsonc_comments(text)
        data = json.loads(cleaned)
        assert data == {"a": 1}

    def test_loads_jsonc_file_end_to_end(self, tmp_path):
        """A .jsonc config file with every flavor of comment must load."""
        p = tmp_path / "llm_config.jsonc"
        p.write_text(
            "// ── top-level banner ──\n"
            "{\n"
            '  "api_key": "sk-A",  // inline\n'
            "  /* block comment */\n"
            '  "base_url": "https://x/v1",\n'
            '  "model_name": "m1"\n'
            "}\n"
            "/* ignored trailing block */\n",
        )
        cfg = load_llm_config(p)
        assert cfg.api_key == "sk-A"
        assert cfg.base_url == "https://x/v1"
        assert cfg.model_name == "m1"

    def test_jsonc_extension_recognised(self, tmp_path):
        """``.jsonc`` is a first-class extension in the loader — not just
        a renamed ``.json``."""
        p = tmp_path / "creds.jsonc"
        p.write_text(
            "// must allow comments here\n"
            '{"api_key": "k", "base_url": "u", "model_name": "m"}\n'
        )
        cfg = load_llm_config(p)
        assert cfg.api_key == "k"

    def test_jsonc_listed_in_default_filenames(self):
        """Auto-discovery looks for ``llm_config.jsonc`` first."""
        from molactagent.config import DEFAULT_CONFIG_FILENAMES
        assert "llm_config.jsonc" in DEFAULT_CONFIG_FILENAMES
        assert DEFAULT_CONFIG_FILENAMES.index("llm_config.jsonc") < (
            DEFAULT_CONFIG_FILENAMES.index("llm_config.json")
        )

    def test_auto_discovery_finds_jsonc_first(self, tmp_path, monkeypatch):
        """When BOTH llm_config.jsonc and llm_config.json exist, the
        loader prefers the .jsonc one."""
        (tmp_path / "llm_config.jsonc").write_text(
            '{"api_key": "JSONC", "base_url": "u", "model_name": "m"}'
        )
        (tmp_path / "llm_config.json").write_text(
            '{"api_key": "JSON", "base_url": "u", "model_name": "m"}'
        )
        monkeypatch.delenv("MOLACTAGENT_API_KEY", raising=False)
        monkeypatch.delenv("MOLACTAGENT_BASE_URL", raising=False)
        monkeypatch.delenv("MOLACTAGENT_MODEL", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_llm_config(None)
        assert cfg.api_key == "JSONC"

    def test_placeholder_value_triggers_config_error(self, tmp_path):
        """A key left as ``REPLACE-WITH-…`` counts as missing, not present."""
        p = tmp_path / "llm_config.jsonc"
        p.write_text(
            '{"api_key": "REPLACE-WITH-YOUR-OWN-KEY",\n'
            ' "base_url": "https://x/v1",\n'
            ' "model_name": "m1"}\n',
        )
        with pytest.raises(LLMConfigError) as e:
            load_llm_config(p)
        assert "api_key" in str(e.value)


class TestUserFacingMessage:
    def test_user_facing_message_formats_hints(self):
        err = LLMConfigError(
            "Thing broke.", config_path="/tmp/llm_config.jsonc",
            hints=["do X", "check Y"],
        )
        msg = err.user_facing_message()
        assert "Thing broke." in msg
        assert "/tmp/llm_config.jsonc" in msg
        assert "do X" in msg
        assert "check Y" in msg
        assert "Open llm_config.jsonc" in msg


class TestBuildLLMClient:
    def test_returns_llmclient(self):
        cfg = LLMConfig(
            api_key="sk-x",
            base_url="https://example.com",
            model_name="m",
        )
        client = build_llm_client(cfg)
        from molactagent.llm import LLMClient
        assert isinstance(client, LLMClient)
        assert client.model_name == "m"

    def test_to_client_kwargs_complete(self):
        cfg = LLMConfig(
            api_key="k", base_url="u", model_name="m",
            temperature=0.3, max_tokens=1500, timeout=60.0, max_retries=5,
        )
        kw = cfg.to_client_kwargs()
        assert kw["api_key"] == "k"
        assert kw["temperature"] == 0.3
        assert kw["max_tokens"] == 1500
        assert kw["timeout"] == 60.0
        assert kw["max_retries"] == 5
