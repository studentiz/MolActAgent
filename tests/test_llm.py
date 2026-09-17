"""Unit tests for :class:`molactagent.llm.LLMClient` — specifically the
``ping()`` method that classifies connection errors so the preflight
check can surface actionable messages."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from molactagent.llm import LLMClient


# ── Helpers ──────────────────────────────────────────────────────────


def _make_client() -> LLMClient:
    """Build an LLMClient without actually contacting any API.

    We replace the underlying ``self.client`` with a MagicMock so we can
    drive ``chat.completions.create`` to whatever response we want.
    """
    c = LLMClient(
        api_key="sk-FAKE",
        base_url="https://example.invalid/v1",
        model_name="fake-model",
        timeout=5.0,
        max_retries=1,
    )
    c.client = MagicMock()
    return c


# ── ping() success ──────────────────────────────────────────────────


class TestPingSuccess:
    def test_returns_ok_with_response(self):
        c = _make_client()
        # Simulate an OpenAI-style response object.
        choice = SimpleNamespace(
            message=SimpleNamespace(content="pong")
        )
        c.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice]
        )
        out = c.ping(timeout=1.0)
        assert out["ok"] is True
        assert out["response"] == "pong"
        assert "elapsed_s" in out

    def test_empty_content_is_still_ok(self):
        """An endpoint may return None / '' for max_tokens=1; still 'ok'."""
        c = _make_client()
        choice = SimpleNamespace(
            message=SimpleNamespace(content=None)
        )
        c.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice]
        )
        out = c.ping()
        assert out["ok"] is True
        assert out["response"] == ""


# ── ping() error classification ─────────────────────────────────────


class TestPingErrors:
    @pytest.mark.parametrize(
        "raise_msg",
        [
            "AuthenticationError: 401 Unauthorized",
            "Invalid API key",
            "403 Forbidden",
            "Authentication failed",
            "Incorrect API key supplied",
        ],
    )
    def test_auth_error_kinds(self, raise_msg):
        c = _make_client()
        c.client.chat.completions.create.side_effect = Exception(raise_msg)
        out = c.ping()
        assert out["ok"] is False
        assert out["error_kind"] == "auth"
        assert raise_msg in out["error"]

    @pytest.mark.parametrize(
        "raise_msg",
        [
            "NotFoundError: 404 model not found",
            "No such model: fakemodel",
            "model 'x' does not exist",
            "unknown model",
        ],
    )
    def test_model_error_kinds(self, raise_msg):
        c = _make_client()
        c.client.chat.completions.create.side_effect = Exception(raise_msg)
        out = c.ping()
        assert out["ok"] is False
        assert out["error_kind"] == "model"

    @pytest.mark.parametrize(
        "raise_msg",
        [
            "ConnectionError: name resolution failure",
            "Request timed out",
            "Connection refused",
            "SSL handshake failed",
            "Network is unreachable",
        ],
    )
    def test_network_error_kinds(self, raise_msg):
        c = _make_client()
        c.client.chat.completions.create.side_effect = Exception(raise_msg)
        out = c.ping()
        assert out["ok"] is False
        assert out["error_kind"] == "network"

    def test_unknown_error_kind_falls_back(self):
        c = _make_client()
        c.client.chat.completions.create.side_effect = Exception(
            "completely unexpected cosmic ray event"
        )
        out = c.ping()
        assert out["ok"] is False
        assert out["error_kind"] == "unknown"
        assert "cosmic ray" in out["error"]

    def test_auth_wins_over_model_for_mixed_message(self):
        """If both '401' AND 'not found' appear, auth takes precedence
        because that's the more specific diagnosis."""
        c = _make_client()
        c.client.chat.completions.create.side_effect = Exception(
            "401 Unauthorized: model not found"
        )
        out = c.ping()
        assert out["error_kind"] == "auth"


# ── ping() bypasses retries ─────────────────────────────────────────


class TestPingBypassesRetries:
    def test_single_api_call_on_failure(self):
        """``ping`` must not trigger the retry loop — users want fast
        feedback on config errors, not 10 slow retries."""
        c = _make_client()
        c.client.chat.completions.create.side_effect = Exception(
            "401 Unauthorized"
        )
        c.ping()
        assert c.client.chat.completions.create.call_count == 1
