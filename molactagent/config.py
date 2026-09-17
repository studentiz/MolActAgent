"""LLM API configuration loading.

Reads OpenAI-compatible API credentials from a variety of sources so the
CLI can be friendly to both first-time users and power users.

Supported sources, in order of precedence (the first to provide a value
wins for each field):
  1. Explicit keyword arguments to :func:`load_llm_config`.
  2. Environment variables: ``MOLACTAGENT_API_KEY``, ``MOLACTAGENT_BASE_URL``,
     ``MOLACTAGENT_MODEL``.
  3. The configuration file pointed to by the ``--llm-config`` CLI flag
     (defaults to ``llm_config.jsonc`` in the project root).

Supported file formats:
  - ``.jsonc``: the **recommended** format. JSON-with-comments — ``//``
    line comments and ``/* … */`` block comments are stripped before
    parsing. Lets users keep ready-made vendor templates commented out
    in the same file.
  - ``.json``: pure JSON. MolActAgent still strips ``//`` and ``/* */``
    comments out of legacy ``.json`` files for back-compat (so old
    ``llm_config.json`` files keep working) but the canonical extension
    for commented configs is ``.jsonc``.
  - ``.yaml`` / ``.yml``: same keys as JSON; comments use ``#``.
  - ``.txt`` (legacy): ``key：value`` (full-width OR ASCII colon) lines,
    optional blank-line separated profiles. Kept for backward compat.

Required keys are ``api_key`` (or ``apiKey``), ``base_url`` (or
``baseUrl``), and ``model_name`` (or ``Modelname`` / ``model``).
Optional keys: ``temperature``, ``max_tokens``, ``timeout``,
``max_retries``.

MolActAgent expects **exactly one active configuration** per file. To
switch vendors, edit the file and uncomment the block you want.

The loader returns a :class:`LLMConfig` dataclass that
:func:`build_llm_client` can turn into a ready-to-use
:class:`molactagent.llm.LLMClient`.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


__all__ = [
    "LLMConfig",
    "LLMConfigError",
    "load_llm_config",
    "write_example_config",
    "build_llm_client",
    "DEFAULT_CONFIG_FILENAMES",
]

# llm_config.jsonc is the recommended file (JSONC = JSON with comments;
# users want the file documented inline). The remaining names are
# back-compat fallbacks so existing checkouts keep working.
DEFAULT_CONFIG_FILENAMES = (
    "llm_config.jsonc",
    "llm_config.json",
    "llm_config.yaml",
    "llm_config.yml",
    "LLM_API.txt",
    "llm_api.txt",
)


class LLMConfigError(ValueError):
    """Raised when the LLM configuration is missing, malformed, or
    when a runtime LLM call exposes a credential / endpoint problem.

    The CLI catches this and prints a clear "edit your llm_config.jsonc"
    pointer instead of dumping a stack trace.
    """

    def __init__(
        self, message: str, *, config_path: str | None = None,
        hints: list[str] | None = None,
    ):
        super().__init__(message)
        self.config_path = config_path
        self.hints = list(hints or [])

    def user_facing_message(self) -> str:
        bits = [str(self)]
        if self.config_path:
            bits.append(f"Config file checked: {self.config_path}")
        else:
            bits.append("No config file was found.")
        bits.append("Action items:")
        bits.append(
            "  1. Open llm_config.jsonc (or run `molactagent init-config` "
            "to create one)."
        )
        bits.append(
            "  2. Make sure exactly ONE provider block is uncommented "
            "and that api_key, base_url, and model_name are set "
            "correctly."
        )
        bits.append(
            "  3. Verify the API key has not expired and that you have "
            "internet access."
        )
        for h in self.hints:
            bits.append(f"  * {h}")
        return "\n".join(bits)

_KEY_ALIASES = {
    "api_key": ("api_key", "apiKey", "apikey", "openai_api_key", "key"),
    "base_url": ("base_url", "baseUrl", "baseurl", "endpoint", "openai_base_url"),
    "model_name": ("model_name", "Modelname", "ModelName", "modelname", "model"),
    "temperature": ("temperature",),
    "max_tokens": ("max_tokens", "maxTokens"),
    "timeout": ("timeout",),
    "max_retries": ("max_retries", "maxRetries"),
}


@dataclass
class LLMConfig:
    """Resolved LLM connection settings.

    Only ``api_key``, ``base_url`` and ``model_name`` are required. The
    rest carry the same defaults as :class:`molactagent.llm.LLMClient`.
    """

    api_key: str
    base_url: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int | None = 2000
    timeout: float = 120.0
    max_retries: int = 10
    profile: str | None = None
    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": float(self.temperature),
            "max_tokens": self.max_tokens,
            "timeout": float(self.timeout),
            "max_retries": int(self.max_retries),
        }
        return kwargs


# ── parsing helpers ──────────────────────────────────────────────────

def _split_kv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    for sep in ("：", ":", "="):
        if sep in line:
            k, v = line.split(sep, 1)
            return k.strip(), v.strip().strip('"').strip("'")
    return None


def _parse_text_chunk(chunk: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in chunk.splitlines():
        kv = _split_kv_line(line)
        if kv:
            out[kv[0]] = kv[1]
    return out


def _parse_text_file(text: str) -> list[dict[str, str]]:
    """Parse legacy LLM_API.txt content into one dict per blank-line block."""
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text.strip()) if c.strip()]
    if not chunks:
        return []
    parsed = [_parse_text_chunk(c) for c in chunks]
    return [d for d in parsed if d]


def _strip_jsonc_comments(text: str) -> str:
    """Strip ``//`` line and ``/* … */`` block comments from a JSONC string.

    Skips characters that fall inside string literals so a URL like
    ``"https://api.example.com"`` is preserved intact. Trailing commas
    that may appear after stripping are left alone — the spec requires
    only that the underlying JSON is otherwise valid; users get a
    clear message from :func:`json.loads` if they include trailing
    commas.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    string_char = ""
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            j = text.find("\n", i + 2)
            if j == -1:
                break
            i = j  # leave the newline in place so error line numbers stay accurate
            continue
        if ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                break
            # Preserve newlines inside the block comment so error
            # messages keep line numbers reasonable.
            block = text[i + 2:j]
            out.extend("\n" * block.count("\n"))
            i = j + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_structured_file(path: pathlib.Path) -> list[dict[str, Any]] | dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf in (".json", ".jsonc"):
        cleaned = _strip_jsonc_comments(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            ext = "JSONC" if suf == ".jsonc" else "JSON / JSONC"
            raise ValueError(
                f"{path.name} is not valid {ext}: {e.msg} "
                f"(line {e.lineno}, column {e.colno}). Hint: MolActAgent "
                "supports `// line` and `/* block */` comments, but not "
                "trailing commas."
            ) from e
    if suf in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - optional dep
            raise ValueError(
                "PyYAML is required to load .yaml/.yml configs; install "
                "it with `pip install pyyaml`."
            ) from e
        try:
            return yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise ValueError(
                f"{path.name} is not valid YAML: {e}"
            ) from e
    raise ValueError(f"Unsupported config format: {suf!r}")


def _normalise_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Map any aliased key to its canonical name."""
    out: dict[str, Any] = {}
    for canonical, aliases in _KEY_ALIASES.items():
        for a in aliases:
            if a in d and d[a] not in (None, ""):
                out[canonical] = d[a]
                break
    return out


# ── profile selection (mirrors scripts/run_demo.py behaviour) ─────────

def _select_profile(
    profiles: list[dict[str, Any]], profile: str | None,
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("No usable profile found in the config file.")

    # Normalise each profile so we can search across alias names.
    norm = [_normalise_dict(p) for p in profiles]

    if profile is None or profile.lower() in ("last", "auto", "default"):
        return norm[-1]
    p = profile.lower()
    if p in ("first",):
        return norm[0]
    if p in ("deepseek", "ds"):
        for entry in norm:
            blob = (entry.get("base_url", "") + " " + entry.get("model_name", "")).lower()
            if "deepseek" in blob:
                return entry
        return norm[0]
    if p in ("qwen", "dashscope"):
        for entry in norm:
            blob = (entry.get("base_url", "") + " " + entry.get("model_name", "")).lower()
            if "dashscope" in blob or "qwen" in blob:
                return entry
        return norm[-1]
    # Fallback: name-match against the model_name
    for entry in norm:
        if profile.lower() in entry.get("model_name", "").lower():
            return entry
    raise ValueError(
        f"Unknown profile {profile!r}; expected one of: last, first, "
        f"deepseek, qwen, or part of a model name."
    )


# ── main entry point ─────────────────────────────────────────────────

def _find_default_config(start: pathlib.Path | None = None) -> pathlib.Path | None:
    base = pathlib.Path(start) if start else pathlib.Path.cwd()
    for parent in (base, *base.parents):
        for name in DEFAULT_CONFIG_FILENAMES:
            candidate = parent / name
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def load_llm_config(
    config_path: str | pathlib.Path | None = None,
    *,
    profile: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    project_root: str | pathlib.Path | None = None,
) -> LLMConfig:
    """Resolve LLM connection settings from CLI args, env, or a file.

    The order of precedence is (highest to lowest):

    1. Explicit keyword arguments to this function.
    2. ``MOLACTAGENT_API_KEY`` / ``MOLACTAGENT_BASE_URL`` / ``MOLACTAGENT_MODEL``
       environment variables.
    3. The supplied ``config_path`` (or, if not given, the first file
       matching :data:`DEFAULT_CONFIG_FILENAMES` found by walking up from
       ``project_root`` or the current directory).

    Raises :class:`FileNotFoundError` only when ``config_path`` is given
    explicitly and does not exist. Raises :class:`ValueError` when no
    source provides the required fields.
    """
    file_data: dict[str, Any] = {}
    source: str | None = None

    if config_path is not None:
        path = pathlib.Path(str(config_path)).expanduser()
        if not path.exists():
            raise LLMConfigError(
                f"LLM config file not found: {path}.",
                config_path=str(path),
                hints=[
                    "Pass --llm-config with a valid path, OR",
                    "Set the MOLACTAGENT_API_KEY / MOLACTAGENT_BASE_URL / "
                    "MOLACTAGENT_MODEL environment variables, OR",
                    "Run `molactagent init-config` to create a starter "
                    "llm_config.jsonc next to your data.",
                ],
            )
    else:
        path = _find_default_config(project_root if project_root else None)

    if path is not None and path.exists():
        suf = path.suffix.lower()
        try:
            if suf in (".txt",):
                profiles = _parse_text_file(path.read_text(encoding="utf-8"))
                file_data = _select_profile(profiles, profile) if profiles else {}
            else:
                parsed = _parse_structured_file(path)
                if isinstance(parsed, list):
                    file_data = _select_profile(parsed, profile)
                elif isinstance(parsed, dict) and "profiles" in parsed:
                    file_data = _select_profile(parsed["profiles"], profile)
                else:
                    file_data = _normalise_dict(parsed if isinstance(parsed, dict) else {})
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Could not parse LLM config {path}: {type(e).__name__}: {e}"
            ) from e
        source = str(path)

    env_data: dict[str, Any] = {}
    for env_key, target in (
        ("MOLACTAGENT_API_KEY", "api_key"),
        ("MOLACTAGENT_BASE_URL", "base_url"),
        ("MOLACTAGENT_MODEL", "model_name"),
        ("MOLACTAGENT_TEMPERATURE", "temperature"),
        ("MOLACTAGENT_TIMEOUT", "timeout"),
    ):
        if os.environ.get(env_key):
            env_data[target] = os.environ[env_key]

    explicit: dict[str, Any] = {}
    if api_key is not None:
        explicit["api_key"] = api_key
    if base_url is not None:
        explicit["base_url"] = base_url
    if model_name is not None:
        explicit["model_name"] = model_name
    if temperature is not None:
        explicit["temperature"] = temperature
    if max_tokens is not None:
        explicit["max_tokens"] = max_tokens
    if timeout is not None:
        explicit["timeout"] = timeout
    if max_retries is not None:
        explicit["max_retries"] = max_retries

    merged: dict[str, Any] = {**file_data, **env_data, **explicit}

    missing = [
        k for k in ("api_key", "base_url", "model_name")
        if not merged.get(k) or str(merged.get(k, "")).strip().upper().startswith("REPLACE")
    ]
    if missing:
        sources_tried = []
        if source:
            sources_tried.append(f"file {source}")
        if env_data:
            sources_tried.append(f"env vars {sorted(env_data)}")
        if explicit:
            sources_tried.append("explicit kwargs")
        tried = ", ".join(sources_tried) if sources_tried else "no sources"
        raise LLMConfigError(
            f"LLM config is missing or has placeholder value(s) for "
            f"required field(s) {missing}. Sources consulted: {tried}.",
            config_path=source,
            hints=[
                "Open llm_config.jsonc and replace any 'REPLACE-WITH-…' "
                "placeholder values with your real API key.",
                "Make sure exactly one provider block is uncommented.",
                "Required fields: api_key, base_url, model_name.",
            ],
        )

    # ``source`` reflects every layer that actually fed values into the
    # final config. This makes it easy to debug "why is my api_key
    # 'sk-FAKE' when my file says 'sk-real'?" — the answer shows up
    # right in the dataclass.
    source_label = source
    if env_data and explicit:
        source_label = (source_label + " + env + kwargs") if source_label else "env + kwargs"
    elif env_data:
        source_label = (source_label + " + env") if source_label else "env"
    elif explicit:
        source_label = (source_label + " + kwargs") if source_label else "kwargs"

    cfg_kwargs: dict[str, Any] = {
        "api_key": str(merged["api_key"]),
        "base_url": str(merged["base_url"]),
        "model_name": str(merged["model_name"]),
        "profile": profile,
        "source": source_label,
    }
    if "temperature" in merged:
        try:
            cfg_kwargs["temperature"] = float(merged["temperature"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Could not interpret temperature value "
                f"{merged['temperature']!r} as a float."
            )
    if "max_tokens" in merged and merged["max_tokens"] not in (None, "", "null"):
        try:
            cfg_kwargs["max_tokens"] = int(merged["max_tokens"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Could not interpret max_tokens value "
                f"{merged['max_tokens']!r} as an int."
            )
    if "timeout" in merged:
        try:
            cfg_kwargs["timeout"] = float(merged["timeout"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Could not interpret timeout value "
                f"{merged['timeout']!r} as a float."
            )
    if "max_retries" in merged:
        try:
            cfg_kwargs["max_retries"] = int(merged["max_retries"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Could not interpret max_retries value "
                f"{merged['max_retries']!r} as an int."
            )

    return LLMConfig(**cfg_kwargs)


def build_llm_client(
    config: LLMConfig | None = None,
    /,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
):
    """Instantiate a :class:`molactagent.llm.LLMClient`.

    Two equivalent calling styles:

    1. **From a resolved :class:`LLMConfig`** (the original API,
       used by the CLI and any pipeline that already loaded a
       config file)::

           cfg = ma.load_llm_config('llm_config.jsonc')
           client = ma.build_llm_client(cfg)

    2. **From in-line credentials** (the Python-first ergonomic
       path — useful when you want to drive the client purely from
       code without touching a config file)::

           client = ma.build_llm_client(
               api_key='sk-...',
               base_url='https://api.deepseek.com',
               model_name='deepseek-chat',
           )

    The optional ``temperature`` / ``max_tokens`` / ``timeout`` /
    ``max_retries`` kwargs override the corresponding
    :class:`LLMClient` defaults; omit them to accept the engine's
    defaults (temperature 0.0, max_tokens 2000, timeout 120s,
    max_retries 10).
    """
    from .llm import LLMClient

    if config is not None:
        kwargs = config.to_client_kwargs()
        # Allow ad-hoc overrides without rebuilding the LLMConfig:
        if temperature is not None: kwargs["temperature"] = float(temperature)
        if max_tokens is not None:  kwargs["max_tokens"]  = int(max_tokens)
        if timeout is not None:     kwargs["timeout"]     = float(timeout)
        if max_retries is not None: kwargs["max_retries"] = int(max_retries)
        return LLMClient(**kwargs)

    if not (api_key and base_url and model_name):
        raise LLMConfigError(
            "build_llm_client() needs either a resolved LLMConfig as the "
            "first positional argument, OR all three of api_key/base_url/"
            "model_name as keyword arguments. Got: "
            f"api_key={'set' if api_key else 'missing'}, "
            f"base_url={'set' if base_url else 'missing'}, "
            f"model_name={'set' if model_name else 'missing'}.",
            hints=[
                "Direct call:  ma.build_llm_client(api_key='sk-...', "
                "base_url='https://...', model_name='deepseek-chat')",
                "From config:  ma.build_llm_client(ma.load_llm_config('llm_config.jsonc'))",
            ],
        )

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
    }
    if temperature is not None: kwargs["temperature"] = float(temperature)
    if max_tokens is not None:  kwargs["max_tokens"]  = int(max_tokens)
    if timeout is not None:     kwargs["timeout"]     = float(timeout)
    if max_retries is not None: kwargs["max_retries"] = int(max_retries)
    return LLMClient(**kwargs)


# ── starter file ─────────────────────────────────────────────────────

EXAMPLE_TXT = """\
# MolActAgent LLM API config (legacy text format)
#   * one block per provider, separated by a blank line
#   * keys may use ASCII ':' or full-width '：'
#   * lines starting with '#' are comments
#
# Pick ONE block (or have several and pass --profile to choose).
# Strings are stored AS-IS — do not wrap values in quotes.

# DeepSeek (default if you have only one block)
apiKey: sk-REPLACE-ME
base_url: https://api.deepseek.com
Modelname: deepseek-chat

# Alibaba Cloud DashScope (Qwen) — uncomment to enable
# apiKey: sk-REPLACE-ME-TOO
# base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
# Modelname: qwen-max
"""

# JSONC template. Exactly ONE provider object must be active at a time;
# the rest are kept as commented-out templates the user can swap in.
EXAMPLE_JSONC = """\
// ─────────────────────────────────────────────────────────────────────
// MolActAgent LLM API configuration (JSONC — JSON with comments)
//
// Rules:
//   * Keep exactly ONE configuration object active (uncommented) below.
//   * Required fields: api_key, base_url, model_name.
//   * Optional: temperature (default 0.0), max_tokens (default 2000),
//     timeout (default 120), max_retries (default 10).
//   * Replace REPLACE-WITH-YOUR-OWN-KEY with your real API key.
//   * To switch vendors: comment out the active block and uncomment
//     a different one. Then save the file.
// ─────────────────────────────────────────────────────────────────────

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

  // ── Alibaba Cloud DashScope (Qwen) ──────────────────────────────
  {
    "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model_name": "qwen-plus"
  }

  // ── OpenAI ──────────────────────────────────────────────────────
  {
    "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini"
  }

  // ── Anthropic Claude (via OpenAI-compatible proxies, e.g. OpenRouter) ─
  {
    "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
    "base_url": "https://openrouter.ai/api/v1",
    "model_name": "anthropic/claude-3.5-sonnet"
  }

  // ── Google Gemini (via OpenAI-compatible endpoint) ──────────────
  {
    "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "model_name": "gemini-1.5-pro"
  }

  // ── Mistral AI ──────────────────────────────────────────────────
  {
    "api_key": "REPLACE-WITH-YOUR-OWN-KEY",
    "base_url": "https://api.mistral.ai/v1",
    "model_name": "mistral-large-latest"
  }

  // ── Local self-hosted (Ollama / vLLM / LM Studio) ───────────────
  {
    "api_key": "ollama",
    "base_url": "http://localhost:11434/v1",
    "model_name": "llama3.1:8b"
  }

──────────────────────────────────────────────────────────────────────── */
"""


def write_example_config(
    path: str | pathlib.Path,
    *,
    fmt: str = "jsonc",
    overwrite: bool = False,
) -> pathlib.Path:
    """Write a starter LLM config file.

    Parameters
    ----------
    path:
        Destination filepath. Parent directories are created as needed.
    fmt:
        ``"jsonc"`` (recommended; writes JSON-with-comments),
        ``"json"`` (alias for jsonc — the loader strips comments either
        way), or ``"txt"`` (legacy text format).
    overwrite:
        Refuse to overwrite an existing file unless this is ``True``.
    """
    p = pathlib.Path(path).expanduser()
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"{p} already exists. Pass --overwrite to replace it."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("jsonc", "json"):
        p.write_text(EXAMPLE_JSONC, encoding="utf-8")
    elif fmt == "txt":
        p.write_text(EXAMPLE_TXT, encoding="utf-8")
    else:
        raise ValueError(
            f"Unknown format: {fmt!r}; use 'jsonc' (recommended), "
            "'json', or 'txt'."
        )
    logger.info(f"Wrote example LLM config to {p}")
    return p
