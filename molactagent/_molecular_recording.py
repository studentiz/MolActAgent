"""Record and replay model responses through the molecular engine."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


class ReplayError(ValueError):
    """The recording does not match the requested prediction."""


class ModelRequestError(RuntimeError):
    """A request failed before a model response was available."""


def _request_hash(kwargs):
    request = {name: value for name, value in kwargs.items() if name != "tag"}
    encoded = json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class ReplayClient:
    """Provide stored model responses without credentials or network access."""

    def __init__(self, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != 1:
            raise ReplayError("Expected a schema 1 molecular recording.")
        if not isinstance(payload.get("calls"), dict) or not isinstance(payload.get("model"), str):
            raise ReplayError("The recording must contain model and calls fields.")
        self.model_name = payload["model"]
        self.settings = payload.get("settings", {})
        if not isinstance(self.settings, dict):
            raise ReplayError("The recording settings must be an object.")
        self._calls = payload["calls"]
        self._local = threading.local()
        self._lock = threading.Lock()
        self._used = set()

    @property
    def last_call(self):
        return getattr(self._local, "last_call", None)

    def chat(self, **kwargs):
        tag = kwargs["tag"]
        call = self._calls.get(tag)
        if not isinstance(call, dict) or call.get("request_sha256") != _request_hash(kwargs):
            raise ReplayError(
                f"Recorded request differs at {tag}. Use the recording's inputs, options, "
                "and MolActAgent and RDKit versions."
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
