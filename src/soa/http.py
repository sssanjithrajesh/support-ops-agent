"""One HTTP client for every integration: retries with jitter, Retry-After, offline guard.

Hand-rolled so the behaviour is readable in one screen: what happens on a 429 with Retry-After,
on a 503, on a timeout, and when SOA_OFFLINE is set. Lifted from contract-compliance-reviewer.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import httpx


class OfflineError(RuntimeError):
    """Raised when an outbound call is attempted while SOA_OFFLINE is set."""


class RetryableError(RuntimeError):
    """Transient failure: the caller (usually the outbox) may retry later."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(RuntimeError):
    """The request is wrong (4xx other than 429). Retrying will not help."""


def offline() -> bool:
    return os.environ.get("SOA_OFFLINE", "") in ("1", "true", "yes")


@dataclass
class RetryPolicy:
    attempts: int = 4
    base_seconds: float = 0.5
    max_seconds: float = 20.0
    jitter: float = 0.25  # fraction of the computed delay

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(self.max_seconds, retry_after)
        exp = min(self.max_seconds, self.base_seconds * (2**attempt))
        return exp * (1 + random.uniform(-self.jitter, self.jitter))  # noqa: S311


class Client:
    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        policy: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        http: httpx.Client | None = None,
        sleep=time.sleep,
        timeout: float = 15.0,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self.attempts_made = 0  # for tests and the write-up: how many HTTP calls were actually issued
        self._http = http or httpx.Client(
            base_url=base_url, headers=headers or {}, auth=auth, transport=transport, timeout=timeout
        )

    def request(self, method: str, url: str, attempts: int | None = None, **kwargs) -> httpx.Response:
        """attempts=1 disables in-process retries: for writes that are not idempotent at the transport
        level (Zendesk ticket updates), the caller must reconcile before retrying. See zendesk.update_ticket."""
        if offline():
            raise OfflineError(f"SOA_OFFLINE is set; refused {method} {url}")
        last: Exception | None = None
        total = attempts or self.policy.attempts
        for attempt in range(total):
            self.attempts_made += 1
            try:
                response = self._http.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = RetryableError(f"{method} {url}: {exc.__class__.__name__}")
            else:
                if response.status_code < 400:
                    return response
                if response.status_code == 429 or response.status_code >= 500:
                    last = RetryableError(f"{method} {url}: HTTP {response.status_code}", _retry_after(response))
                else:
                    raise PermanentError(f"{method} {url}: HTTP {response.status_code} {response.text[:200]}")
            if attempt < total - 1:
                self._sleep(self.policy.delay(attempt, getattr(last, "retry_after", None)))
        assert last is not None
        raise last

    def get(self, url: str, **kw) -> httpx.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> httpx.Response:
        return self.request("POST", url, **kw)

    def put(self, url: str, **kw) -> httpx.Response:
        return self.request("PUT", url, **kw)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
