"""Record and replay model responses through the molecular engine."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from platform import python_version

from . import __version__


class ReplayError(ValueError):
    """The recording does not match the requested prediction."""


class ModelRequestError(RuntimeError):
    """A request failed before a model response was available."""


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_versions():
    from rdkit import rdBase
    import pandas as pd

    return {"molactagent": __version__, "rdkit": rdBase.rdkitVersion,
            "python": python_version(), "pandas": pd.__version__}


def _request_hash(kwargs):
    return digest({name: value for name, value in kwargs.items() if name != "tag"})


class ReplayClient:
    """Provide stored model responses without credentials or network access."""

    def __init__(self, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("schema") == 1:
            raise ReplayError("This recording uses schema 1. Replay it with MolActAgent v0.4.1 and its original options.")
        if not isinstance(payload, dict) or payload.get("schema") != 2:
            raise ReplayError("Expected a schema 2 molecular recording.")
        for field in ("calls", "settings", "versions", "inputs"):
            if not isinstance(payload.get(field), dict):
                raise ReplayError(f"The recording must contain a {field!r} object.")
        if not isinstance(payload.get("model"), str):
            raise ReplayError("The recording must contain a model name.")
        self.model_name = payload["model"]
        self.settings = payload["settings"]
        self.command = payload.get("command")
        if self.command is not None and (
            not isinstance(self.command, dict)
            or any(not isinstance(self.command.get(field), dict) for field in ("files", "columns", "transport"))
        ):
            raise ReplayError("The recording's CLI manifest is invalid.")
        self.inputs = payload["inputs"]
        versions = runtime_versions()
        for field in ("molactagent", "rdkit"):
            # 0.5.1 changed the report. Its molecular requests and vote rules are
            # identical to 0.5.0, and replay still checks each request digest.
            if (field == "molactagent" and versions[field] == "0.5.1"
                    and payload["versions"].get(field) == "0.5.0"):
                continue
            if payload["versions"].get(field) != versions[field]:
                raise ReplayError(
                    f"Recording version differs for {field}: expected {payload['versions'].get(field)!r}, "
                    f"running {versions[field]!r}. Use the release and dependencies that made this recording."
                )
        self._calls = payload["calls"]
        self._local = threading.local()
        self._lock = threading.Lock()
        self._used = set()

    def begin(self, settings, inputs):
        for name, actual in settings.items():
            if self.settings.get(name) != actual:
                raise ReplayError(f"Recording setting differs for {name!r}.")
        for name, actual in inputs.items():
            if self.inputs.get(name) != actual:
                raise ReplayError(f"Recording input differs for {name!r}.")
        with self._lock:
            self._used.clear()

    @property
    def last_call(self):
        return getattr(self._local, "last_call", None)

    def chat(self, **kwargs):
        tag = kwargs["tag"]
        call = self._calls.get(tag)
        if not isinstance(call, dict) or call.get("request_sha256") != _request_hash(kwargs):
            raise ReplayError(
                f"Recorded request differs at {tag}. The input, prompt, or request options have changed."
            )
        with self._lock:
            if tag in self._used:
                raise ReplayError(f"Duplicate request {tag} in this replay.")
            self._used.add(tag)
        metadata = call.get("metadata")
        if not isinstance(metadata, dict) or (
            not call.get("error") and not isinstance(call.get("response"), str)
        ):
            raise ReplayError(f"Missing response or metadata for {tag}.")
        self._local.last_call = metadata
        if call.get("error"):
            raise ModelRequestError(call["error"])
        return call["response"]

    def verify_complete(self):
        unused = self._calls.keys() - self._used
        if unused:
            raise ReplayError(f"{len(unused)} recorded requests were unused.")


class CallRecorder:
    def __init__(self, upstream, *, keep_responses):
        self.upstream = upstream
        self.keep_responses = keep_responses
        self.calls = {}
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def last_call(self):
        return getattr(self._local, "last_call", None)

    def chat(self, **kwargs):
        tag = kwargs["tag"]
        response = None
        error = None
        try:
            response = self.upstream.chat(**kwargs) or ""
        except ReplayError:
            raise
        except ModelRequestError as exc:
            error = str(exc)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            error = f"HTTP {status}" if isinstance(status, int) else type(exc).__name__

        metadata = {
            **(getattr(self.upstream, "last_call", None) or {}),
            "tag": tag,
            "error": error,
        }
        self._local.last_call = metadata
        with self._lock:
            self.calls[tag] = {
                "request_sha256": _request_hash(kwargs),
                "metadata": metadata,
                "error": error,
                **({"response": response} if self.keep_responses else {}),
            }
        if error:
            raise ModelRequestError(error)
        return response

    def mark_response(self, status, detail=None):
        metadata = self.last_call
        metadata["status"] = status
        if detail:
            metadata["detail"] = detail
