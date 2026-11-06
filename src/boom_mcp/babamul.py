"""A small async client for the Babamul API, BOOM's public interface.

Reads the same environment variables as the `babamul` package, so the `.env`
from the school setup works unchanged.
"""

import os

import httpx
from mcp.server.mcpserver.exceptions import ToolError

API_URLS = {
    "production": "https://babamul.caltech.edu/api/babamul",
    "backup": "https://babamul.umn.edu/api/babamul",
    "local": "http://localhost:4000/babamul",
}
USER_AGENT = "boom-mcp/0.1"


class BabamulError(ToolError):
    """A failure the model can act on: the message says what to change."""


def base_url_from_env() -> str:
    if url := os.environ.get("BABAMUL_API_URL"):
        return url.rstrip("/")
    env = os.environ.get("BABAMUL_ENV", "production").lower()
    if env not in API_URLS:
        raise ValueError(f"BABAMUL_ENV must be one of {sorted(API_URLS)}, not {env!r}")
    return API_URLS[env]


class Babamul:
    def __init__(self, http: httpx.AsyncClient, token: str | None, base_url: str):
        self.http = http
        self.token = token
        self.base_url = base_url

    async def request(self, method: str, path: str, *, params=None, json=None, timeout=120.0):
        if not self.token:
            raise BabamulError(
                "No Babamul API token. Set BABAMUL_API_TOKEN to the token on your "
                "profile page at https://babamul.caltech.edu (or send it as a Bearer "
                "token when connecting over HTTP)."
            )
        params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            response = await self.http.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                json=json,
                headers={"Authorization": f"Bearer {self.token}", "User-Agent": USER_AGENT},
                timeout=timeout,
            )
        except httpx.TimeoutException as e:
            raise BabamulError(
                f"Babamul did not answer within {timeout:.0f} s. Narrow the query "
                "(shorter time window, magnitude or reliability cuts) and retry."
            ) from e
        except httpx.RequestError as e:
            raise BabamulError(f"Could not reach Babamul at {self.base_url}: {e}") from e

        if response.status_code == 401:
            raise BabamulError("Babamul rejected the API token. Check BABAMUL_API_TOKEN and BABAMUL_ENV.")
        if response.status_code >= 400:
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            kind = "Not found" if response.status_code == 404 else f"Babamul error {response.status_code}"
            raise BabamulError(f"{kind}: {message}")

        body = response.json()
        return body.get("data", body) if isinstance(body, dict) else body

    async def get(self, path: str, **params):
        return await self.request("GET", path, params=params)

    async def post(self, path: str, json: dict):
        return await self.request("POST", path, json=json)
