"""The server instance every tool module registers on."""

import inspect
import os
from contextlib import asynccontextmanager

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from .babamul import Babamul, base_url_from_env

# Conventions a model cannot infer from the tool schemas, and that silently give
# a wrong answer when missed. Same idea as SkyPortal's /mcp instructions.
INSTRUCTIONS = """Query BOOM, the alert broker for ZTF and Rubin/LSST, through its \
public Babamul API: alerts, light curves, image cutouts, catalog cross-matches, \
and the machine-learning outputs BOOM computes for every public ZTF alert.

Before you trust a result:

- Times. Alerts are indexed by Julian Date. Use convert_time instead of doing \
date arithmetic yourself, and call it with no argument to learn the current \
time. "Last night" for ZTF (Palomar, UTC-8) is roughly 02:00-14:00 UTC; for \
Rubin (Chile, UTC-3) roughly 23:00-10:00 UTC.
- Magnitudes. Smaller is brighter. A source that is brightening has a \
decreasing magnitude. Limits are 3-sigma upper limits: the source was fainter \
than that, not that bright.
- Reliability (ZTF drb, LSST reliability) is a 0-1 real/bogus score. ZTF \
alerts below ~0.5 are mostly artifacts.
- Babamul serves public data only (ZTF programid 1, public Rubin alerts), so \
"not found" can mean "not public".
- Volume. One hour of public ZTF is ~20,000 alerts. A time-only \
search_alerts query covers at most 1 day; without quality cuts (min_reliability, \
stationary_only, exclude_stars, exclude_solar_system, magnitude) it stops at \
max_scanned long before the night is done. Cut first, then widen.
- Counts. Only report a number of alerts or objects that a tool returned \
(matched, objects). When a result says truncated, that number is a lower bound \
and you must say so. Never estimate a count you did not measure.
- Machine learning (ZTF only). ACAI scores (hosted, nuclear, variable_star, \
orphan, bogus) are independent 0-1 scores, not a distribution. BTSbot is the \
probability that a bright (<18.5 mag) transient is a supernova worth Bright \
Transient Survey follow-up. AppleCiDEr is hierarchical (domain, family, class) \
and abstains when unsure: an abstention or a family-level decision is \
information, not a missing value. Its out_of_distribution flag marks alerts \
unlike its training set, which is how BOOM flags anomalies.
- Candidate ids (candid) are 19-digit integers. Pass and quote them as \
strings: as JSON numbers they lose their last digits.
- An empty result is more often a too-tight cut or the wrong survey than an \
empty sky. Loosen one cut at a time before concluding nothing is there."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)

# Set by server.main(): over HTTP, each caller must send their own token.
# Tests swap http_client for one with a mock transport.
settings = {"require_client_token": False, "http_client": httpx.AsyncClient}


@asynccontextmanager
async def lifespan(server):
    async with settings["http_client"]() as http:
        yield {"http": http, "base_url": base_url_from_env()}


mcp = MCPServer(
    name="boom",
    title="BOOM alert broker",
    instructions=INSTRUCTIONS,
    version="0.1.0",
    lifespan=lifespan,
)


def tool(**kwargs):
    """mcp.tool(), with the docstring dedented and the tool marked read-only."""

    def register(fn):
        kwargs.setdefault("description", inspect.cleandoc(fn.__doc__ or ""))
        kwargs.setdefault("annotations", READ_ONLY)
        return mcp.tool(**kwargs)(fn)

    return register


def babamul(ctx: Context) -> Babamul:
    """A Babamul client acting as the caller: their Bearer token over HTTP,
    BABAMUL_API_TOKEN over stdio."""
    state = ctx.request_context.lifespan_context
    token = None
    auth = (ctx.headers or {}).get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if token is None and not settings["require_client_token"]:
        token = os.environ.get("BABAMUL_API_TOKEN")
    return Babamul(state["http"], token, state["base_url"])
