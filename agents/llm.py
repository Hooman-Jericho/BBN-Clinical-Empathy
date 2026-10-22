"""Provider-agnostic LLM client: bounded retries, reproducibility settings, usage accounting.

Fixes over the previous implementation:
* ``temperature`` is now sent to **both** providers (the Anthropic path ignored it).
* optional ``seed`` (OpenAI) and ``json_mode`` (OpenAI ``response_format``) are supported.
* the retry/auth logic exists once instead of being copy-pasted in two classes.
* token and call counters make cost comparable between agent designs.
"""

import logging
import threading
import time
from typing import Any, Dict

from .base import AgentConfig

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class LLMClient:
    """Unified wrapper for LLM API calls with bounded retries and informative errors."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = None
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    # ------------------------------------------------------------------ plumbing
    def _get_client(self):
        if self._client is None:
            if self.config.provider == "openai":
                from openai import OpenAI

                self._client = OpenAI(api_key=self.config.api_key)
            elif self.config.provider == "anthropic":
                from anthropic import Anthropic

                self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if self.config.provider == "openai":
            inp = _as_int(getattr(usage, "prompt_tokens", 0))
            out = _as_int(getattr(usage, "completion_tokens", 0))
        else:
            inp = _as_int(getattr(usage, "input_tokens", 0))
            out = _as_int(getattr(usage, "output_tokens", 0))
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out

    def usage(self) -> Dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }

    # ------------------------------------------------------------------ request
    def _openai_kwargs(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if self.config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _anthropic_kwargs(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

    def chat(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
        """Send one chat request with bounded retries and actionable error messages."""
        client = self._get_client()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if self.config.provider == "openai":
                    response = client.chat.completions.create(
                        **self._openai_kwargs(system_prompt, user_prompt)
                    )
                    self._record_usage(response)
                    return response.choices[0].message.content or ""

                if self.config.provider == "anthropic":
                    response = client.messages.create(
                        **self._anthropic_kwargs(system_prompt, user_prompt)
                    )
                    self._record_usage(response)
                    return response.content[0].text if response.content else ""

                raise ValueError(f"Unknown provider: {self.config.provider}")

            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                last_error = exc
                msg = str(exc).lower()
                if "authentication" in msg or "api key" in msg or "unauthorized" in msg:
                    raise RuntimeError(
                        f"LLM Provider Authentication Failure ({self.config.provider}): {exc}. "
                        f"Verify your {self.config.provider.upper()}_API_KEY."
                    ) from exc
                if isinstance(exc, ValueError):
                    raise
                if attempt < max_retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc
                    )
                    time.sleep(0.5 * (2**attempt))
                    continue

        raise RuntimeError(
            f"LLM Provider Error ({self.config.provider}, model '{self.config.model}') "
            f"after {max_retries + 1} attempts: {last_error}"
        ) from last_error
