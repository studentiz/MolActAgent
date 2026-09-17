"""Shared sklearn plumbing for MolActAgent estimators.

The two estimators (:class:`MolActAgentClassifier`,
:class:`MolActAgentRegressor`) need the same boilerplate: input
validation, LLM client resolution, fitted-state checks, a place to
stash diagnostic state. This module factors that out.

Intentionally light on dependencies — we avoid importing
:mod:`sklearn.base` lazily-but-eagerly so that ``import molactagent``
stays cheap. The full sklearn ``BaseEstimator`` machinery is only
imported when an estimator is instantiated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import LLMConfig, build_llm_client


_NO_CREDS_HINT = (
    "No LLM credentials available. Pick one:\n"
    "  • In-code  : ca.MolActAgentClassifier(api_key='sk-...', "
    "base_url='https://api.deepseek.com', model_name='deepseek-chat')\n"
    "  • Reuse    : ca.MolActAgentClassifier(llm_client=existing_client)\n"
    "  • Reuse cfg: ca.MolActAgentClassifier(llm_config=ca.load_llm_config('llm_config.jsonc'))\n"
    "  • Env vars : export MOLACTAGENT_API_KEY / MOLACTAGENT_BASE_URL / MOLACTAGENT_MODEL\n"
    "  • File     : run `molactagent init-config` to create llm_config.jsonc\n"
)


def _ensure_llm_client(
    llm_client: Any | None,
    llm_config: LLMConfig | None,
    *,
    # Required credentials (any subset)
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    # Optional client-behavior overrides
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Resolve an LLM client from the available configuration sources.

    Resolution order (highest priority first; ties resolved within
    each path by precedence rules described below):

    1. **Pre-built client.** If ``llm_client`` is supplied, return it
       verbatim. The optional override kwargs (``temperature`` etc.)
       are intentionally ignored on this path — a pre-built client
       carries its own settings, baked in at construction time.

    2. **Pre-built config.** If ``llm_config`` is supplied, build a
       :class:`molactagent.llm.LLMClient` from it. Any optional
       override kwargs supplied to this call are layered on top
       (mirroring :func:`build_llm_client`).

    3. **Complete in-code credentials.** If all three of ``api_key``,
       ``base_url``, ``model_name`` are supplied, build an
       :class:`LLMConfig` purely from those — no file or env-var
       discovery is consulted. Predictable, no-surprises path:
       what you pass is what you get. Optional overrides apply.

    4. **Partial in-code credentials.** If at least one but not all
       three of the credential kwargs are supplied, route through
       :func:`load_llm_config` so the missing fields can be filled
       from a discovered config file or ``MOLACTAGENT_*`` env vars.
       Explicit kwargs always win for the fields they cover.

    5. **Full auto-discovery.** No explicit source — call
       :func:`load_llm_config` with no required args, which walks
       up from cwd looking for ``llm_config.jsonc`` /
       ``LLM_API.txt`` and consults ``MOLACTAGENT_*`` env vars.
       Optional override kwargs still apply.

    Paths 3, 4, and 5 wrap their config-loading step with one
    common ``try`` block, so any failure to produce required
    credentials (api_key / base_url / model_name) is re-raised as
    a single ``ValueError`` whose message lists every alternative
    path the user could have taken.

    The wrapper deliberately does NOT perform a preflight ping
    (the high-level pipeline does, but estimator users typically
    run many predictions in a row and validate the endpoint
    up-front via their own ``client.ping()`` call).
    """
    # ── Path 1: pre-built client wins, period. ────────────────────
    if llm_client is not None:
        return llm_client

    # Optional client-behavior overrides — applied to every other
    # path so the call site has consistent semantics.
    overrides: dict[str, Any] = {}
    if temperature is not None: overrides["temperature"] = temperature
    if max_tokens is not None:  overrides["max_tokens"]  = max_tokens
    if timeout is not None:     overrides["timeout"]     = timeout
    if max_retries is not None: overrides["max_retries"] = max_retries

    # ── Path 2: pre-built LLMConfig (with optional overrides). ───
    if llm_config is not None:
        return build_llm_client(llm_config, **overrides)

    # Decide which of paths 3/4/5 to take based on which of the
    # three required credential fields the user supplied.
    has_complete_credentials = (
        api_key is not None
        and base_url is not None
        and model_name is not None
    )
    has_partial_credentials = (
        not has_complete_credentials
        and any(v is not None for v in (api_key, base_url, model_name))
    )

    from ..config import LLMConfigError, load_llm_config

    try:
        if has_complete_credentials:
            # Path 3: pure in-code. Bypass any file/env discovery —
            # the user opted into "this and only this" by giving all
            # three required fields. Optional overrides are applied
            # at LLMConfig construction time.
            cfg_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
                "model_name": model_name,
                **overrides,
            }
            cfg = LLMConfig(**cfg_kwargs)
        elif has_partial_credentials:
            # Path 4: partial inline credentials. Allow file/env
            # discovery to fill the missing required fields. The
            # explicit kwargs always win for the fields they cover.
            cfg = load_llm_config(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            # Path 5: full auto-discovery. Optional overrides still
            # take effect, applied on top of whatever auto-discovery
            # finds (file values < env values < these kwargs).
            cfg = load_llm_config(
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
    except LLMConfigError as e:
        raise ValueError(_NO_CREDS_HINT + f"\nUnderlying error: {e}") from e

    # Paths 3, 4, and 5 already folded the optional overrides into
    # ``cfg``. A plain ``build_llm_client(cfg)`` is enough here —
    # passing ``**overrides`` again would be a harmless no-op
    # because the values are already in ``cfg``.
    return build_llm_client(cfg)


def _coerce_X(X) -> np.ndarray:
    """Convert ``X`` to a 2D float64 ndarray, mirroring sklearn semantics.

    Accepts numpy arrays, lists of lists, pandas DataFrames, etc.
    Raises ``ValueError`` on shape problems with a message that
    points at the expected ``(n_samples, n_features)`` convention.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        # A single sample — promote to 2D so downstream code can
        # treat it like a 1-row batch. This matches sklearn's
        # accepted input shapes for ``predict``.
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected X to be 2D (n_samples, n_features); "
            f"got shape {arr.shape}."
        )
    return arr


def _coerce_y_classification(y) -> np.ndarray:
    """Stringify class labels — matches the engine's contract.

    The multi-class engine treats class labels as opaque strings
    throughout (see ``docs/design.md`` §29.2 — "the class-label
    space is opaque"). Coercing to ``str`` here means callers can
    pass int / bool / numpy-typed labels and the wrapper does the
    right thing without surprising the engine.
    """
    arr = np.asarray(y)
    return arr.astype(str)


def _coerce_y_regression(y) -> np.ndarray:
    """Coerce the regression target to ``float64``.

    The engine requires a finite, numeric target. NaN handling is
    delegated to the engine's own validators (see
    :func:`molactagent.utils.validate_inputs`) so the error messages
    stay consistent regardless of which entry point the user came
    from.
    """
    return np.asarray(y, dtype=np.float64)


class _NotFittedError(RuntimeError):
    """Raised when ``predict``/``score`` is called before ``fit``.

    Matches sklearn's ``NotFittedError`` semantics without inheriting
    from sklearn — keeps the import dependency optional for users
    who only want the estimator wrapper.
    """


def _check_fitted(estimator: Any) -> None:
    """Assert that the estimator has been ``fit()``-ed."""
    if getattr(estimator, "_agent", None) is None:
        raise _NotFittedError(
            f"{type(estimator).__name__} is not fitted yet. "
            "Call `fit(X, y)` before `predict` / `predict_proba` / "
            "`score`."
        )
