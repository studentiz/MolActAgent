"""Tests for the Python-first in-code LLM configuration path.

The estimator wrappers and ``build_llm_client`` accept LLM credentials
via four equivalent paths:

    1. In-code:   MolActAgentClassifier(api_key=..., base_url=..., model_name=...)
    2. Reuse:     MolActAgentClassifier(llm_client=existing_client)
    3. Reuse cfg: MolActAgentClassifier(llm_config=ma.load_llm_config(...))
    4. Auto:      MolActAgentClassifier()  # walks up looking for a file or env vars

These tests exercise paths 1, 2, 3, and the env-var part of path 4.
The "walk up looking for a file" branch is covered indirectly by the
existing real-LLM smoke suite and is omitted here so we don't depend
on filesystem state.

Every test uses :class:`tests.conftest.FakeLLM` (or an in-memory
``LLMConfig``) so the suite is fully deterministic and offline.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pytest

import molactagent as ma
from molactagent.config import LLMConfig, LLMConfigError, build_llm_client
from molactagent.estimators._base import _ensure_llm_client


@contextmanager
def _clean_env(*keys: str):
    """Temporarily clear the named env vars; restore on exit."""
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


MOLACTAGENT_ENV = (
    "MOLACTAGENT_API_KEY",
    "MOLACTAGENT_BASE_URL",
    "MOLACTAGENT_MODEL",
    "MOLACTAGENT_TEMPERATURE",
    "MOLACTAGENT_TIMEOUT",
)


# ── build_llm_client: both calling styles ───────────────────────────


def test_build_llm_client_accepts_llmconfig():
    cfg = LLMConfig(
        api_key="sk-fake",
        base_url="https://example.com/v1",
        model_name="fake-model",
    )
    client = build_llm_client(cfg)
    assert client.model_name == "fake-model"
    # The OpenAI client wraps the api_key/base_url internally; we
    # assert via attributes the LLMClient itself stores.
    assert getattr(client, "client").api_key == "sk-fake"


def test_build_llm_client_accepts_inline_kwargs():
    client = build_llm_client(
        api_key="sk-inline",
        base_url="https://example.com/v1",
        model_name="inline-model",
    )
    assert client.model_name == "inline-model"
    assert getattr(client, "client").api_key == "sk-inline"


def test_build_llm_client_inline_overrides_optional_kwargs():
    client = build_llm_client(
        api_key="sk-x", base_url="u", model_name="m",
        temperature=0.7, max_tokens=512, timeout=30.0, max_retries=2,
    )
    assert client.temperature == 0.7
    assert client.max_tokens == 512
    assert client.timeout == 30.0
    assert client.max_retries == 2


def test_build_llm_client_rejects_partial_inline_kwargs():
    with pytest.raises(LLMConfigError) as excinfo:
        build_llm_client(api_key="sk-x", base_url="u")  # no model_name
    msg = str(excinfo.value)
    assert "model_name=missing" in msg
    assert "api_key=set" in msg


def test_build_llm_client_config_overrides_optional_kwargs():
    """Optional kwargs override LLMConfig values without rebuilding."""
    cfg = LLMConfig(
        api_key="sk-x", base_url="u", model_name="m",
        temperature=0.0, timeout=120.0,
    )
    client = build_llm_client(cfg, temperature=0.5, timeout=10.0)
    assert client.temperature == 0.5
    assert client.timeout == 10.0


# ── _ensure_llm_client resolution order ─────────────────────────────


def test_ensure_llm_client_pre_built_client_wins(fake_llm):
    """If llm_client is given, every other source is ignored."""
    out = _ensure_llm_client(
        fake_llm, None,
        api_key="ignored", base_url="ignored", model_name="ignored",
    )
    assert out is fake_llm


def test_ensure_llm_client_uses_inline_when_no_client_or_config():
    with _clean_env(*MOLACTAGENT_ENV):
        client = _ensure_llm_client(
            None, None,
            api_key="sk-fake",
            base_url="https://example.com/v1",
            model_name="fake-model",
        )
    assert client.model_name == "fake-model"


def test_ensure_llm_client_uses_env_vars_when_no_other_source(monkeypatch):
    with _clean_env(*MOLACTAGENT_ENV):
        monkeypatch.setenv("MOLACTAGENT_API_KEY", "sk-env")
        monkeypatch.setenv("MOLACTAGENT_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("MOLACTAGENT_MODEL", "env-model")
        client = _ensure_llm_client(None, None)
    assert client.model_name == "env-model"
    assert getattr(client, "client").api_key == "sk-env"


def test_ensure_llm_client_raises_with_helpful_message_when_no_source(tmp_path, monkeypatch):
    """No source available → clear, copy-pastable error message."""
    monkeypatch.chdir(tmp_path)  # walk-up search will find nothing
    with _clean_env(*MOLACTAGENT_ENV):
        with pytest.raises(ValueError) as excinfo:
            _ensure_llm_client(None, None)
    msg = str(excinfo.value)
    # All four documented escape hatches must be in the error message.
    assert "ma.MolActAgentClassifier(api_key=" in msg
    assert "llm_client=" in msg
    assert "llm_config=" in msg
    assert "MOLACTAGENT_API_KEY" in msg
    assert "molactagent init-config" in msg


def test_ensure_llm_client_raises_friendly_error_with_only_overrides(tmp_path, monkeypatch):
    """Optional knobs alone (temperature/max_tokens/etc.) without
    credentials should still surface the friendly multi-option
    error message — NOT the raw LLMConfigError. This is the case
    that used to fall through with an unfriendly error before the
    review fix.
    """
    monkeypatch.chdir(tmp_path)
    with _clean_env(*MOLACTAGENT_ENV):
        with pytest.raises(ValueError) as excinfo:
            _ensure_llm_client(
                None, None,
                temperature=0.5, max_tokens=512,
                timeout=30.0, max_retries=2,
            )
    msg = str(excinfo.value)
    assert "ma.MolActAgentClassifier(api_key=" in msg, (
        "Optional-knobs-only path must surface the friendly error "
        "message, not the raw LLMConfigError. Got: " + msg
    )


def test_ensure_llm_client_raises_friendly_error_with_partial_credentials(tmp_path, monkeypatch):
    """Partial credentials with no env / file fallback also need
    the friendly message."""
    monkeypatch.chdir(tmp_path)
    with _clean_env(*MOLACTAGENT_ENV):
        with pytest.raises(ValueError) as excinfo:
            _ensure_llm_client(None, None, api_key="sk-only")
    msg = str(excinfo.value)
    assert "ma.MolActAgentClassifier(api_key=" in msg


def test_ensure_llm_client_complete_inline_bypasses_file_discovery(tmp_path, monkeypatch):
    """Path 3: when ALL THREE credentials are inline, no file or
    env-var content should leak into the resulting config — even if
    a stale config file is present in the cwd.
    """
    # Plant a stale config file with a different model_name and
    # a non-default temperature that would override silently if
    # the loader still merged file data.
    cfg_file = tmp_path / "llm_config.jsonc"
    cfg_file.write_text(
        '{"api_key": "sk-stale", "base_url": "https://stale.example.com/v1",'
        ' "model_name": "stale-model", "temperature": 0.99}'
    )
    monkeypatch.chdir(tmp_path)
    with _clean_env(*MOLACTAGENT_ENV):
        client = _ensure_llm_client(
            None, None,
            api_key="sk-fresh",
            base_url="https://fresh.example.com/v1",
            model_name="fresh-model",
        )
    # The stale config file MUST NOT have leaked any value into the
    # resulting client.
    assert client.model_name == "fresh-model"
    assert getattr(client, "client").api_key == "sk-fresh"
    # And the temperature MUST NOT be the file's 0.99 — should be the
    # LLMConfig dataclass default (0.0).
    assert client.temperature == 0.0


def test_ensure_llm_client_partial_inline_does_merge_file(tmp_path, monkeypatch):
    """Path 4: partial credentials → file/env fills the missing fields.
    The user explicitly accepted the merge by NOT supplying everything."""
    cfg_file = tmp_path / "llm_config.jsonc"
    cfg_file.write_text(
        '{"api_key": "sk-from-file", "base_url": "https://file.example.com/v1",'
        ' "model_name": "model-from-file"}'
    )
    monkeypatch.chdir(tmp_path)
    with _clean_env(*MOLACTAGENT_ENV):
        client = _ensure_llm_client(
            None, None, api_key="sk-explicit",
        )
    # Explicit api_key wins; base_url and model_name come from the file.
    assert getattr(client, "client").api_key == "sk-explicit"
    assert client.model_name == "model-from-file"


def test_ensure_llm_client_llm_config_path_honors_optional_overrides(fake_llm):
    """Path 2: passing llm_config + optional overrides should apply
    the overrides (consistent with build_llm_client(cfg, **overrides))."""
    cfg = LLMConfig(
        api_key="sk-x", base_url="u", model_name="m",
        temperature=0.0, timeout=120.0,
    )
    client = _ensure_llm_client(
        None, cfg, temperature=0.7, timeout=15.0,
    )
    assert client.temperature == 0.7
    assert client.timeout == 15.0


def test_ensure_llm_client_complete_inline_honors_optional_overrides():
    """Path 3: complete in-code credentials + optional overrides
    should apply the overrides at LLMConfig construction time."""
    with _clean_env(*MOLACTAGENT_ENV):
        client = _ensure_llm_client(
            None, None,
            api_key="sk-x", base_url="u", model_name="m",
            temperature=0.6, max_tokens=4096, timeout=45.0, max_retries=5,
        )
    assert client.temperature == 0.6
    assert client.max_tokens == 4096
    assert client.timeout == 45.0
    assert client.max_retries == 5


# ── MolActAgentClassifier: in-code credentials path ─────────────────


def test_classifier_in_code_credentials_construction():
    """In-code credentials don't trigger LLM calls until fit()."""
    clf = ma.MolActAgentClassifier(
        api_key="sk-fake",
        base_url="https://example.com/v1",
        model_name="fake-model",
        k=3,
    )
    # Hyperparameters must be stored unchanged; no LLM client built yet.
    assert clf.api_key == "sk-fake"
    assert clf.base_url == "https://example.com/v1"
    assert clf.model_name == "fake-model"
    assert clf.k == 3
    assert clf._agent is None  # lazy — no fit yet


def test_classifier_in_code_credentials_get_set_params_roundtrip():
    """sklearn clone-style round-trip preserves the new credential fields."""
    clf = ma.MolActAgentClassifier(
        api_key="sk-x", base_url="u", model_name="m", temperature=0.3, k=4,
    )
    params = clf.get_params()
    assert params["api_key"] == "sk-x"
    assert params["base_url"] == "u"
    assert params["model_name"] == "m"
    assert params["temperature"] == 0.3
    assert params["k"] == 4

    # Reconstruct via set_params (this is what sklearn.clone uses).
    fresh = ma.MolActAgentClassifier()
    fresh.set_params(**params)
    assert fresh.api_key == "sk-x"
    assert fresh.k == 4


def test_classifier_in_code_credentials_fit_predict(
    multiclass_reference, multiclass_test, fake_llm, monkeypatch,
):
    """Full fit→predict using in-code creds. We monkey-patch
    build_llm_client so the inline credentials route to the FakeLLM
    used elsewhere in the suite — no real LLM endpoint needed.
    """
    monkeypatch.setattr(
        "molactagent.estimators._base.build_llm_client",
        lambda cfg, **_: fake_llm,
    )
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_ref = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_ref = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)

    clf = ma.MolActAgentClassifier(
        api_key="sk-fake",
        base_url="https://example.com/v1",
        model_name="fake-model",
        k=3, max_workers=1,
    ).fit(X_ref, y_ref)

    # Engine was built and predict still returns the expected shape.
    assert clf._agent is not None
    labels = clf.predict(X_test)
    assert labels.shape == (X_test.shape[0],)


# ── MolActAgentRegressor: in-code credentials path ──────────────────


def test_regressor_in_code_credentials_construction():
    reg = ma.MolActAgentRegressor(
        api_key="sk-fake",
        base_url="https://example.com/v1",
        model_name="fake-model",
        k=5,
    )
    assert reg.api_key == "sk-fake"
    assert reg.base_url == "https://example.com/v1"
    assert reg.model_name == "fake-model"
    assert reg.k == 5
    assert reg._agent is None


# ── pandas DataFrame auto-inference of feature_names ───────────────


def test_classifier_infers_feature_names_from_dataframe(
    multiclass_reference, fake_llm,
):
    pd = pytest.importorskip("pandas")
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_ref_df = multiclass_reference[feature_cols]
    y_ref = multiclass_reference["bucket"].to_numpy()

    clf = ma.MolActAgentClassifier(
        llm_client=fake_llm, k=3, max_workers=1,
    ).fit(X_ref_df, y_ref)

    # The engine's stored feature_names matches the DataFrame columns,
    # not the generic "feature_0 / feature_1 / ..." it would default to.
    assert list(clf.feature_names_in_) == feature_cols


def test_classifier_explicit_feature_names_override_dataframe_columns(
    multiclass_reference, fake_llm,
):
    pd = pytest.importorskip("pandas")
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    explicit = [f"explicit_{i}" for i in range(len(feature_cols))]
    X_ref_df = multiclass_reference[feature_cols]
    y_ref = multiclass_reference["bucket"].to_numpy()

    clf = ma.MolActAgentClassifier(
        llm_client=fake_llm,
        feature_names=explicit,
        k=3, max_workers=1,
    ).fit(X_ref_df, y_ref)

    # Explicit names win over DataFrame columns.
    assert list(clf.feature_names_in_) == explicit


# ── back-compat: existing patterns must still work ─────────────────


def test_back_compat_llm_client_path_still_works(
    multiclass_reference, multiclass_test, fake_llm,
):
    """Path 2 (pre-built client) — the pattern every existing test uses."""
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_ref = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_ref = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)
    clf = ma.MolActAgentClassifier(
        llm_client=fake_llm, k=3, max_workers=1,
    ).fit(X_ref, y_ref)
    assert clf.predict(X_test).shape == (X_test.shape[0],)


def test_back_compat_llm_config_path_still_works(
    multiclass_reference, multiclass_test, fake_llm, monkeypatch,
):
    """Path 3 (pre-built LLMConfig) — the pattern the CLI uses."""
    monkeypatch.setattr(
        "molactagent.estimators._base.build_llm_client",
        lambda cfg, **_: fake_llm,
    )
    feature_cols = ["alpha", "beta", "gamma", "delta"]
    X_ref = multiclass_reference[feature_cols].to_numpy(dtype=float)
    y_ref = multiclass_reference["bucket"].to_numpy()
    X_test = multiclass_test[feature_cols].to_numpy(dtype=float)
    cfg = LLMConfig(api_key="sk-fake", base_url="u", model_name="m")
    clf = ma.MolActAgentClassifier(
        llm_config=cfg, k=3, max_workers=1,
    ).fit(X_ref, y_ref)
    assert clf.predict(X_test).shape == (X_test.shape[0],)


# ── sklearn integration smoke tests (regression: real-LLM e2e
#    revealed that estimators without BaseEstimator inheritance
#    fail sklearn 1.6+'s tag introspection) ─────────────────────────


def test_classifier_passes_sklearn_is_classifier_check():
    """sklearn 1.6+ introspects ``__sklearn_tags__`` to decide
    whether an estimator is a classifier. Estimators that don't
    inherit from ``BaseEstimator`` raise an AttributeError here,
    which made ``cross_val_score`` / ``GridSearchCV`` fail.
    Inheriting from ``ClassifierMixin + BaseEstimator`` fixes it.
    """
    from sklearn.base import is_classifier
    clf = ma.MolActAgentClassifier(api_key="sk-x", base_url="u", model_name="m")
    assert is_classifier(clf), (
        "is_classifier() must return True after the BaseEstimator + "
        "ClassifierMixin inheritance fix; otherwise sklearn rejects "
        "the estimator before any prediction logic runs."
    )
    # The tags-system call itself must not raise.
    tags = clf.__sklearn_tags__()
    assert tags.estimator_type == "classifier"


def test_regressor_passes_sklearn_is_regressor_check():
    from sklearn.base import is_regressor
    reg = ma.MolActAgentRegressor(api_key="sk-x", base_url="u", model_name="m")
    assert is_regressor(reg)
    tags = reg.__sklearn_tags__()
    assert tags.estimator_type == "regressor"


def test_classifier_sklearn_clone_round_trip(fake_llm):
    """``sklearn.clone()`` must produce a structurally-identical
    estimator. The pre-fix code corrupted ``agent_kwargs`` because
    ``get_params`` returned ``agent_kwargs__foo`` keys but
    ``__init__`` didn't strip the prefix back out.
    """
    from sklearn.base import clone
    original = ma.MolActAgentClassifier(
        api_key="sk-x", base_url="u", model_name="m",
        feature_descriptions={"alpha": "the alpha column"},
        background="some background",
        k=4, metric="cosine", scaler="minmax", max_workers=2,
        # An ad-hoc engine kwarg flows through agent_kwargs:
        random_state=42,
    )
    cloned = clone(original)

    # Top-level fields preserved.
    assert cloned.api_key == "sk-x"
    assert cloned.base_url == "u"
    assert cloned.model_name == "m"
    assert cloned.k == 4
    assert cloned.metric == "cosine"
    assert cloned.scaler == "minmax"
    assert cloned.max_workers == 2
    assert cloned.feature_descriptions == {"alpha": "the alpha column"}
    assert cloned.background == "some background"

    # Critically: agent_kwargs must contain "random_state": 42 with the
    # CORRECT key name — NOT the agent_kwargs__random_state prefixed
    # name that the round-trip would produce without our __init__ fix.
    assert cloned.agent_kwargs == {"random_state": 42}, (
        f"clone() corrupted agent_kwargs: {cloned.agent_kwargs!r}. "
        "Expected {'random_state': 42}. The agent_kwargs__ prefix "
        "must be stripped in __init__ for clone() round-trips to work."
    )


def test_regressor_sklearn_clone_round_trip():
    """Same agent_kwargs round-trip test for the regressor."""
    from sklearn.base import clone
    original = ma.MolActAgentRegressor(
        api_key="sk-x", base_url="u", model_name="m",
        target_name="body_mass_g",
        k=5,
        # Ad-hoc CCR kwarg flows through agent_kwargs:
        ccr_aggregator="mean",
    )
    cloned = clone(original)
    assert cloned.api_key == "sk-x"
    assert cloned.target_name == "body_mass_g"
    assert cloned.k == 5
    assert cloned.agent_kwargs == {"ccr_aggregator": "mean"}


def test_classifier_cross_val_score_doesnt_fail_at_setup(fake_llm, monkeypatch):
    """Smoke-test that ``cross_val_score`` reaches the fit/predict
    stage instead of failing on ``__sklearn_tags__`` lookup. We
    monkey-patch the LLM client so this test runs offline."""
    pytest.importorskip("sklearn.model_selection")
    from sklearn.model_selection import KFold, cross_val_score
    monkeypatch.setattr(
        "molactagent.estimators._base.build_llm_client",
        lambda cfg, **_: fake_llm,
    )
    rng = np.random.default_rng(0)
    n = 6
    X = rng.normal(size=(n, 4))
    y = np.array(["a", "b"] * (n // 2))
    clf = ma.MolActAgentClassifier(
        api_key="sk-x", base_url="u", model_name="m",
        k=2, max_workers=1,
    )
    scores = cross_val_score(
        clf, X, y, cv=KFold(n_splits=2, shuffle=False), n_jobs=1,
    )
    assert scores.shape == (2,)
    assert all(0.0 <= float(s) <= 1.0 for s in scores)
