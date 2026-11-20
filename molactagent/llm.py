import logging
import math
import threading
import time

from openai import APIConnectionError, OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client with per-call timing, token tracking and timeout."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_retries: int = 10,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_tokens: int | None = 2000,
    ):
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds.")
        # Keep HTTP attempts in one loop so that the limit and log agree.
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

        self._lock = threading.Lock()
        self._call_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_time = 0.0
        self._call_log: list[dict] = []
        self._throttle_count = 0
        self._thread_local = threading.local()

    _UNSET = object()

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        tag: str = "",
        max_tokens: int | None | object = _UNSET,
        response_format: dict | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        if max_tokens is self._UNSET:
            mt = self.max_tokens
        else:
            mt = max_tokens
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        with self._lock:
            self._call_count += 1
            call_id = self._call_count
        self._thread_local.last_call = None

        logger.debug(
            f"[LLM #{call_id}] START  tag={tag!r}  "
            f"msgs={len(messages)}  prompt_chars={prompt_chars}"
        )

        create_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temp,
        )
        if mt is not None:
            create_kwargs["max_tokens"] = mt
        if response_format is not None:
            create_kwargs["response_format"] = response_format

        retry_failures = []
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            t0 = time.time()
            try:
                response = self.client.chat.completions.create(**create_kwargs)
                elapsed = time.time() - t0

                usage = response.usage
                p_tok = usage.prompt_tokens if usage else 0
                c_tok = usage.completion_tokens if usage else 0
                content = response.choices[0].message.content
                finish = response.choices[0].finish_reason

                truncated = (finish == "length")
                entry = {
                    "call_id": call_id,
                    "tag": tag,
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "prompt_chars": prompt_chars,
                    "response_chars": len(content) if content else 0,
                    "truncated": truncated,
                    "finish_reason": finish,
                    "response_model": getattr(response, "model", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "http_attempts": attempt + 1,
                    "retry_failures": retry_failures,
                }
                with self._lock:
                    self._total_prompt_tokens += p_tok
                    self._total_completion_tokens += c_tok
                    self._total_time += elapsed
                    self._call_log.append(entry)
                self._thread_local.last_call = entry

                trunc_flag = " TRUNCATED" if truncated else ""
                logger.info(
                    f"[LLM #{call_id}] {elapsed:.1f}s  "
                    f"tok={p_tok}+{c_tok}{trunc_flag}  {tag}"
                )
                logger.debug(
                    f"[LLM #{call_id}] RESPONSE tag={tag!r}:\n{content}"
                )
                if truncated:
                    logger.warning(
                        f"[LLM #{call_id}] output truncated at {c_tok} tokens "
                        f"(max_tokens={mt})  tag={tag!r}"
                    )
                return content

            except Exception as e:
                status = getattr(e, "status_code", None)
                retry_failures.append({
                    "type": type(e).__name__,
                    "status": status,
                })
                self._thread_local.last_call = {
                    "call_id": call_id,
                    "tag": tag,
                    "http_attempts": attempt + 1,
                    "retry_failures": retry_failures,
                }
                elapsed = time.time() - t0
                is_throttle = status == 429
                if is_throttle:
                    with self._lock:
                        self._throttle_count += 1
                logger.warning(
                    f"[LLM #{call_id}] FAIL   attempt={attempt+1}/{attempts}  "
                    f"{elapsed:.1f}s  {'THROTTLED ' if is_throttle else ''}"
                    f"error={status or type(e).__name__}  tag={tag!r}"
                )
                transient = isinstance(e, APIConnectionError) or (
                    isinstance(status, int) and (status in (408, 409, 429) or status >= 500)
                )
                if transient and attempt + 1 < attempts:
                    delay = self.retry_delay * (attempt + 1)
                    if is_throttle:
                        delay *= 2
                    logger.info(f"[LLM #{call_id}] retry in {delay:.0f}s...")
                    time.sleep(delay)
                else:
                    raise

    def ping(self, *, timeout: float = 30.0) -> dict:
        """Perform a tiny round-trip call to verify the connection.

        Returns a dict with ``ok`` (bool), ``elapsed_s`` (float), and
        either ``response`` (str, on success) or ``error`` (str) and
        ``error_kind`` (one of ``auth`` / ``model`` / ``network`` /
        ``unknown``).

        This intentionally bypasses the retry loop so the user gets
        immediate feedback when their API key, base URL, or model name
        is wrong, instead of sitting through ten quiet retries.
        """
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        try:
            kwargs["timeout"] = timeout  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            pass

        t0 = time.time()
        try:
            response = self.client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0
            content = response.choices[0].message.content
            return {
                "ok": True,
                "elapsed_s": round(elapsed, 2),
                "response": content or "",
            }
        except Exception as e:
            elapsed = time.time() - t0
            err_str = str(e).lower()
            if any(t in err_str for t in (
                "401", "403", "unauthorized", "invalid api key",
                "authentication", "permission denied", "forbidden",
                "incorrect api key",
            )):
                kind = "auth"
            elif any(t in err_str for t in (
                "404", "not found", "no such model", "model not found",
                "unknown model", "does not exist",
            )):
                kind = "model"
            elif any(t in err_str for t in (
                "timeout", "timed out", "connection", "network",
                "name resolution", "unreachable", "refused",
                "ssl", "tls",
            )):
                kind = "network"
            else:
                kind = "unknown"
            return {
                "ok": False,
                "elapsed_s": round(elapsed, 2),
                "error": f"{type(e).__name__}: {e}",
                "error_kind": kind,
            }

    @property
    def throttle_count(self) -> int:
        return self._throttle_count

    # ── statistics ───────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_tokens(self) -> int:
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self._total_time, 1),
            "avg_time_s": round(self._total_time / max(1, self._call_count), 1),
        }

    @property
    def call_log(self) -> list[dict]:
        return list(self._call_log)

    @property
    def last_call(self) -> dict | None:
        """Metadata for the latest call on the current thread."""
        return getattr(self._thread_local, "last_call", None)

    def print_stats(self):
        s = self.stats
        logger.info(
            f"[LLM STATS] calls={s['calls']}  "
            f"tokens={s['total_tokens']} "
            f"(prompt={s['total_prompt_tokens']}, comp={s['total_completion_tokens']})  "
            f"time={s['total_time_s']}s  avg={s['avg_time_s']}s/call"
        )
