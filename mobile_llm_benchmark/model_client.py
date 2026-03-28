"""Model client abstraction for OpenRouter and HuggingFace Inference API."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
_HF_BASE = os.getenv("HF_INFERENCE_BASE_URL", "https://api-inference.huggingface.co/v1")
_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
# Base wait (seconds) between retries; actual wait = base * 2^attempt + jitter
_RETRY_BASE_WAIT = float(os.getenv("RETRY_BASE_WAIT", "1.0"))
# Extra wait when a 429 rate-limit is hit (seconds)
_RATE_LIMIT_WAIT = float(os.getenv("RATE_LIMIT_WAIT", "10.0"))


@dataclass
class GenerateResult:
    """Rich result from a single model call."""

    text: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: float = 0.0
    model_id: str = ""
    provider: str = ""

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion


class ModelClient:
    """Unified client for OpenRouter and HuggingFace Inference API.

    Both APIs expose an OpenAI-compatible chat completions endpoint, so we
    use the ``openai`` SDK for both — just different ``base_url`` and ``api_key``.
    """

    def __init__(self) -> None:
        self._openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self._hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        self._openrouter_client: Optional[object] = None
        self._hf_client: Optional[object] = None

    def _get_openrouter_client(self):
        if self._openrouter_client is None:
            from openai import OpenAI

            self._openrouter_client = OpenAI(
                api_key=self._openrouter_key,
                base_url=_OPENROUTER_BASE,
                timeout=_TIMEOUT,
                default_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/dakshjain-1616/mobile-llm-benchmark-suite"),
                    "X-Title": "Mobile LLM Benchmark Suite",
                },
            )
        return self._openrouter_client

    def _get_hf_client(self):
        if self._hf_client is None:
            from openai import OpenAI

            self._hf_client = OpenAI(
                api_key=self._hf_token or "hf_placeholder",
                base_url=_HF_BASE,
                timeout=_TIMEOUT,
            )
        return self._hf_client

    def generate(
        self,
        model_id: str,
        prompt: str,
        provider: str = "openrouter",
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str = "You are a helpful, accurate assistant. Answer concisely.",
    ) -> str:
        """Generate a response. Returns the text string (backward-compatible).

        Use ``generate_full`` to get token counts and latency as well.
        """
        return self.generate_full(
            model_id=model_id,
            prompt=prompt,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        ).text

    def generate_full(
        self,
        model_id: str,
        prompt: str,
        provider: str = "openrouter",
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str = "You are a helpful, accurate assistant. Answer concisely.",
    ) -> GenerateResult:
        """Generate a response and return a rich ``GenerateResult`` with token usage."""
        client = self._get_openrouter_client() if provider == "openrouter" else self._get_hf_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                text = response.choices[0].message.content or ""

                # Extract token usage when available
                usage = getattr(response, "usage", None)
                tokens_prompt = getattr(usage, "prompt_tokens", 0) or 0
                tokens_completion = getattr(usage, "completion_tokens", 0) or 0

                return GenerateResult(
                    text=text,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    latency_ms=latency_ms,
                    model_id=model_id,
                    provider=provider,
                )

            except Exception as exc:
                last_exc = exc
                is_rate_limit = _is_rate_limit_error(exc)
                wait = _backoff_wait(attempt, rate_limited=is_rate_limit)

                logger.warning(
                    "Attempt %d/%d for model %s failed (%s)%s. Retrying in %.1fs.",
                    attempt + 1,
                    _MAX_RETRIES,
                    model_id,
                    type(exc).__name__,
                    " [rate-limited]" if is_rate_limit else "",
                    wait,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)

        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a 429 rate-limit response."""
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _backoff_wait(attempt: int, rate_limited: bool = False) -> float:
    """Exponential backoff with ±25% jitter. Extra delay when rate-limited."""
    base = _RETRY_BASE_WAIT * (2 ** attempt)
    if rate_limited:
        base = max(base, _RATE_LIMIT_WAIT)
    jitter = base * 0.25 * (2 * random.random() - 1)
    return max(0.1, base + jitter)
