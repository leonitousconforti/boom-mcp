"""A small async client for the Babamul API, BOOM's public interface.

Reads the same environment variables as the `babamul` package, so the `.env`
from the school setup works unchanged.
"""

import hashlib
import itertools
import os
import random
import re
import time

import anyio
import httpx
from mcp.server.mcpserver.exceptions import ToolError

API_URLS = {
    "production": "https://babamul.caltech.edu/api/babamul",
    "backup": "https://babamul.umn.edu/api/babamul",
    "local": "http://localhost:4000/babamul",
}
USER_AGENT = "boom-mcp/0.1"

# BOOM answers 503 when its large-query slots stay busy for a minute and 504
# when a query runs long; both usually pass. Waits before each retry, stretched
# by up to half at random so a room of retries does not arrive together.
RETRY_STATUSES = frozenset({429, 502, 503, 504})
RETRY_DELAYS_S = (2.0, 5.0)
# A retry starts only if it should end within this many seconds of the first
# try, taking the next try to last as long as the one that failed, and it is
# cut off there if it runs longer. BOOM waits 60 s for a large-query slot
# before its 503, so that 503 is retried once and reaches the model at about
# 123 s, inside the assistant's 180 s for tools (TOOL_SHARE x answer_timeout).
# A 120 s timeout is not retried at all. The cut-off matters because BOOM's
# alert search has no time limit of its own: a retry that gets a slot late
# would otherwise run to the full timeout, about 183 s in all.
RETRY_WITHIN_S = 150.0

# Object ids go into URL paths, where a slash or a dot segment would send the
# caller's token to another Babamul route.
OBJECT_ID = re.compile(r"[A-Za-z0-9_:-][A-Za-z0-9_.:-]{0,63}")

# Tokens Babamul accepted lately, by hash, and until when that holds. Lets data
# fetched with one token be handed to another only once Babamul takes it too.
ACCEPTED_TTL_S = 600.0
_accepted: dict[str, float] = {}


class BabamulError(ToolError):
    """A failure the model can act on: the message says what to change."""


def base_url_from_env() -> str:
    # Compose and env files pass unset values as empty strings; treat those as unset.
    if url := os.environ.get("BABAMUL_API_URL", "").strip():
        return url.rstrip("/")
    env = os.environ.get("BABAMUL_ENV", "").strip().lower() or "production"
    if env not in API_URLS:
        raise ValueError(f"BABAMUL_ENV must be one of {sorted(API_URLS)}, not {env!r}")
    return API_URLS[env]


def checked_object_id(value: str) -> str:
    """value without surrounding spaces, if it is safe as one URL path segment."""
    text = value.strip()
    if not OBJECT_ID.fullmatch(text):
        raise BabamulError(
            f"{value!r} is not an object id: use letters, digits and _ . : - only "
            "(ZTF ids look like ZTF26aaaaaaa, LSST ids are numbers)."
        )
    return text


def _retry_delay(attempt: int, started: float, tried: float) -> float | None:
    """How long to wait before trying again, or None to give up.

    started is when the first try began and tried when the failed one did.
    """
    if attempt >= len(RETRY_DELAYS_S):
        return None
    delay = RETRY_DELAYS_S[attempt] * random.uniform(1.0, 1.5)
    now = time.monotonic()
    if now - started + delay + (now - tried) > RETRY_WITHIN_S:
        return None
    return delay


class Babamul:
    def __init__(self, http: httpx.AsyncClient, token: str | None, base_url: str):
        self.http = http
        self.token = token
        self.base_url = base_url

    @property
    def identity(self) -> str:
        """Whose results these are, for cache keys: a hash, never the token itself."""
        return hashlib.sha256((self.token or "").encode()).hexdigest()

    async def check_token(self):
        """Raise BabamulError unless Babamul accepts this token; asks at most once per ACCEPTED_TTL_S."""
        if _accepted.get(self.identity, 0.0) <= time.monotonic():
            # The cheapest call Babamul authenticates; success marks the token accepted.
            await self.get("/profile")

    def _accept(self):
        now = time.monotonic()
        if len(_accepted) > 4096:
            for key in [k for k, until in _accepted.items() if until <= now]:
                del _accepted[key]
        _accepted[self.identity] = now + ACCEPTED_TTL_S

    async def request(self, method: str, path: str, *, params=None, json=None, timeout=120.0):
        if not self.token:
            raise BabamulError(
                "No Babamul API token. Set BABAMUL_API_TOKEN to the token on your "
                "profile page at https://babamul.caltech.edu (or send it as a Bearer "
                "token when connecting over HTTP)."
            )
        params = {k: v for k, v in (params or {}).items() if v is not None}
        started = time.monotonic()
        for attempt in itertools.count():
            tried = time.monotonic()
            # A retry gets what is left of RETRY_WITHIN_S, never more than timeout.
            limit = timeout if attempt == 0 else min(timeout, started + RETRY_WITHIN_S - tried)
            try:
                response = await self.http.request(
                    method,
                    f"{self.base_url}{path}",
                    params=params,
                    json=json,
                    headers={"Authorization": f"Bearer {self.token}", "User-Agent": USER_AGENT},
                    timeout=limit,
                )
            except httpx.TimeoutException as e:
                if (delay := _retry_delay(attempt, started, tried)) is None:
                    raise BabamulError(
                        f"Babamul did not answer within {limit:.0f} s. Narrow the query "
                        "(shorter time window, magnitude or reliability cuts) and retry."
                    ) from e
            except httpx.RequestError as e:
                raise BabamulError(f"Could not reach Babamul at {self.base_url}: {e}") from e
            else:
                if response.status_code not in RETRY_STATUSES:
                    break
                if (delay := _retry_delay(attempt, started, tried)) is None:
                    break
            await anyio.sleep(delay)

        if response.status_code == 401:
            _accepted.pop(self.identity, None)
            raise BabamulError("Babamul rejected the API token. Check BABAMUL_API_TOKEN and BABAMUL_ENV.")
        if response.status_code >= 400:
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            kind = "Not found" if response.status_code == 404 else f"Babamul error {response.status_code}"
            raise BabamulError(f"{kind}: {message}")

        self._accept()
        body = response.json()
        return body.get("data", body) if isinstance(body, dict) else body

    async def get(self, path: str, **params):
        return await self.request("GET", path, params=params)

    async def post(self, path: str, json: dict):
        return await self.request("POST", path, json=json)
