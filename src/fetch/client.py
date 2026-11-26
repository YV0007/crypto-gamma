"""Async Deribit client: rate limiting, retries, structured logging.

All endpoints used by this project are PUBLIC. Credentials are optional and
exist only for rate-limit headroom (20 req/s standard, 100 req/s burst); the
client must work with an empty .env, and is tested that way.

Hosts
-----
`history.deribit.com` serves trade history beyond the 24h window AND -- a
non-obvious recon finding -- the full historical instrument universe. The www
host returns only the most recently expired batch. See config `fetch`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

from src import config

HOSTS = {
    "www": "https://www.deribit.com/api/v2",
    "history": "https://history.deribit.com/api/v2",
}

# JSON-RPC error codes worth retrying rather than aborting on.
# 10028 too_many_requests, 10006 too_many_concurrent, 13009 temporarily_unavailable
RETRYABLE_RPC = {10028, 10006, 13009}

log = logging.getLogger("fetch.client")


def setup_logging(stage: str) -> Path:
    """Configure root logging to logs/fetch_{stage}.log plus stderr."""
    log_dir = config.path("fetch.storage.log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    dest = log_dir / f"fetch_{stage}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(dest)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return dest


class RateLimiter:
    """Token bucket shared across all concurrent workers."""

    def __init__(self, rate_per_sec: float) -> None:
        self.rate = float(rate_per_sec)
        self._tokens = self.rate
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self.rate, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self.rate
            await asyncio.sleep(deficit)


class DeribitClient:
    """Rate-limited, retrying async client over the Deribit public API."""

    def __init__(
        self,
        host: str = "history",
        rate_per_sec: float | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.base = HOSTS[host]
        self.host_name = host
        self.rate = rate_per_sec or config.get("api.rate_limit_per_sec", 15)
        self.max_attempts = max_attempts or config.get("fetch.max_attempts", 10)
        self.backoff_base = config.get("fetch.backoff_base_sec", 0.5)
        self.backoff_max = config.get("fetch.backoff_max_sec", 60.0)
        self.jitter = config.get("fetch.jitter", True)
        self._limiter = RateLimiter(self.rate)
        self._client: httpx.AsyncClient | None = None
        self.n_requests = 0
        self.n_retries = 0

    async def __aenter__(self) -> DeribitClient:
        headers = {"User-Agent": "crypto-gamma/0.1 (research; contact via repo)"}
        # Optional auth -- purely for rate-limit headroom. All scopes are `none`.
        cid, secret = os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET")
        auth = (cid, secret) if cid and secret else None
        if auth:
            log.info("using Deribit client credentials for rate-limit headroom")
        else:
            log.info("no credentials found -- using public access (all endpoints are public)")
        self._client = httpx.AsyncClient(
            base_url=self.base,
            headers=headers,
            auth=auth,
            timeout=httpx.Timeout(config.get("api.timeout_sec", 30), connect=15.0),
            limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _sleep_for(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_max)
        delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        if self.jitter:
            delay *= 0.5 + random.random()  # full-ish jitter, never zero
        return delay

    async def get(self, method: str, **params: Any) -> Any:
        """Call a public method, retrying transient failures.

        Raises RuntimeError on a non-retryable JSON-RPC error, or after
        `max_attempts` transient failures.
        """
        url = f"/public/{method}"
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            await self._limiter.acquire()
            try:
                assert self._client is not None
                resp = await self._client.get(url, params=params)
                self.n_requests += 1
                if resp.status_code == 429 or resp.status_code >= 500:
                    ra = resp.headers.get("Retry-After")
                    delay = self._sleep_for(attempt, float(ra) if ra else None)
                    self.n_retries += 1
                    log.warning(
                        "HTTP %s on %s (attempt %d/%d) -- sleeping %.1fs",
                        resp.status_code, method, attempt + 1, self.max_attempts, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                if "error" in payload:
                    err = payload["error"]
                    code = err.get("code")
                    if code in RETRYABLE_RPC:
                        delay = self._sleep_for(attempt, None)
                        self.n_retries += 1
                        log.warning("RPC %s on %s -- sleeping %.1fs", err, method, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(f"{method} failed: {err} params={params}")
                return payload["result"]
            except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                # Large pages occasionally trigger "Server disconnected without
                # sending a response" -- observed during recon. Transient.
                last = exc
                self.n_retries += 1
                delay = self._sleep_for(attempt, None)
                log.warning(
                    "%s on %s (attempt %d/%d) -- sleeping %.1fs",
                    type(exc).__name__, method, attempt + 1, self.max_attempts, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"{method} exhausted {self.max_attempts} attempts; last={last!r}")
