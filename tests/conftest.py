"""A fake Babamul, shaped like the responses of boom/src/api/routes/babamul."""

import base64
import gzip
import io
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from astropy.io import fits

from boom_mcp import plugins, tools  # noqa: F401
from boom_mcp.app import settings

plugins.load_builtin()

JD0 = 2461350.70  # 2026-11-06T04:48Z

CLASSIFICATIONS = {
    "acai_h": 0.91,
    "acai_n": 0.02,
    "acai_v": 0.01,
    "acai_o": 0.03,
    "acai_b": 0.02,
    "btsbot": 0.83,
    "applecider_fusion": {
        "decision_label": "Ia-like SN",
        "decision_level": "class",
        "decision_probability": 0.78,
        "abstain_completely": False,
        "class_probabilities": {
            "AGN-like": 0.01,
            "Accreting WD Var": 0.0,
            "Other Stellar Var": 0.01,
            "TDE": 0.05,
            "Ia-like SN": 0.78,
            "Stripped Envelope SN": 0.08,
            "H-rich CCSN": 0.06,
            "Superluminous SN": 0.01,
        },
    },
    "applecider_outputs": {
        "domain": {"pred": "transient", "prob": 0.97, "kept": True},
        "family": {"pred": "SN", "prob": 0.93, "kept": True},
        "ood": {"score": 0.12, "votes": 0, "flag": False},
        "uncertainty": {"vacuity": 0.1, "epistemic": 0.05, "aleatoric": 0.2},
        "n_detections_used": 6,
    },
}
WEIRD = {
    **CLASSIFICATIONS,
    "applecider_fusion": {**CLASSIFICATIONS["applecider_fusion"], "decision_label": None, "abstain_completely": True},
    "applecider_outputs": {**CLASSIFICATIONS["applecider_outputs"], "ood": {"score": 0.97, "votes": 5, "flag": True}},
}


def candidate(jd, mag, ra=150.1, dec=2.2, drb=0.99):
    return {
        "jd": jd, "ra": ra, "dec": dec, "magpsf": mag, "sigmapsf": 0.05, "drb": drb, "fid": 2, "band": "r",
        "isdiffpos": True, "ndethist": 6, "jdstarthist": JD0, "programid": 1,
    }


def alert(object_id, candid, jd, mag, classifications=CLASSIFICATIONS, **kw):
    return {
        "candid": candid, "objectId": object_id, "candidate": candidate(jd, mag, **kw),
        "properties": {
            "rock": False, "star": False, "near_brightstar": False, "stationary": True, "hosted": True,
            "best_host": {"name": "WISEA J100024", "catalog": "LSDR10", "z": 0.031, "dist_mpc": 135.2, "sep_arcsec": 3.1, "d_dlr": 0.8},
        },
        "classifications": classifications,
    }


def prv(jd, band, mag=None, limit=None):
    fid = {"g": 1, "r": 2}[band]
    return {"jd": jd, "fid": fid, "band": band, "magpsf": mag, "sigmapsf": 0.05 if mag else None, "diffmaglim": limit,
            "isdiffpos": True if mag else None, "programid": 1}


OBJECT = {
    **alert("ZTF26aaaaaaa", 3123456789012345678, JD0 + 4, 17.6),
    "prv_candidates": [prv(JD0, "r", 19.4), prv(JD0 + 1, "g", 18.9), prv(JD0 + 2, "r", 18.2), prv(JD0 + 4, "r", 17.6)],
    "prv_nondetections": [prv(JD0 - 3, "r", limit=20.6), prv(JD0 - 1.5, "r", limit=20.8)],
    "fp_hists": [],
    "classifications_history": [WEIRD, CLASSIFICATIONS, CLASSIFICATIONS],
    "cross_matches": {
        "TNS": [{"name": "2026abc", "type": "SN Ia", "redshift": 0.031, "distance_arcsec": 0.4}],
        "NED": [{"objname": "WISEA J100024", "objtype": "G", "z": 0.031, "distance_arcsec": 3.1}],
        "Gaia_DR3": [],
    },
    "survey_matches": {"lsst": {"objectId": "3141592653589793", "ra": 150.1, "dec": 2.2, "distance_arcsec": 0.3,
                                "prv_candidates": [{"jd": JD0 + 3, "band": "g", "psfFlux": 5000.0, "psfFluxErr": 50.0}],
                                "fp_hists": []}},
}

ALERTS = [
    alert("ZTF26aaaaaaa", 1, JD0 + 0.10, 18.0),
    alert("ZTF26aaaaaaa", 2, JD0 + 0.20, 17.9),
    alert("ZTF26bbbbbbb", 3, JD0 + 0.30, 19.5, classifications=WEIRD),
    alert("ZTF26ccccccc", 4, JD0 + 0.40, 16.2, classifications=None),
]


def fits_gz(seed, background=100.0, source=300.0):
    y, x = np.mgrid[:63, :63]
    data = np.random.default_rng(seed).normal(background, 5, (63, 63))
    data += source * np.exp(-((x - 31) ** 2 + (y - 31) ** 2) / (2 * 1.8**2))
    buf = io.BytesIO()
    fits.PrimaryHDU(data).writeto(buf)
    return base64.b64encode(gzip.compress(buf.getvalue())).decode()


# A login token of the web app, a JWT, for the user bbml_test belongs to, and one whose profile has no id.
LOGIN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyLTEifQ.ZmFrZS1zaWduYXR1cmU"
LOGIN_NO_ID = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJub2JvZHkifQ.bm8taWQtc2lnbmF0dXJl"
FAKE_ANTHROPIC_KEY = "sk-ant-api03-FAKE" + "0" * 76 + "ab12"
FAKE_OPENROUTER_KEY = "sk-or-v1-FAKE" + "0" * 60 + "cd34"
KEYS_SECRET = bytes(range(32))


class FakeBabamul:
    def __init__(self):
        self.requests = []
        # Accepted tokens, and the user each belongs to (None: a profile without an id).
        self.tokens = {"bbml_test": "user-1", LOGIN: "user-1", LOGIN_NO_ID: None}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in self.tokens:
            return httpx.Response(401, json={"message": "Unauthorized"})
        path, q = request.url.path.removeprefix("/api/babamul"), request.url.params
        if path == "/profile":
            return ok({"id": self.tokens[token], "username": "tester", "email": "tester@example.org"})
        if path == "/surveys/ZTF/alerts":
            lo, hi = float(q.get("start_jd", 0)), float(q.get("end_jd", 1e9))
            rows = [a for a in ALERTS if lo <= a["candidate"]["jd"] <= hi]
            skip, limit = int(q.get("skip", 0)), int(q.get("limit", 1000))
            return ok(rows[skip : skip + limit])
        if path == "/surveys/ZTF/objects/ZTF26aaaaaaa":
            return ok(OBJECT)
        if path.startswith("/surveys/ZTF/objects/") and path.endswith("/cross-matches"):
            return ok(OBJECT["cross_matches"])
        if path.startswith("/surveys/ZTF/objects/"):
            return httpx.Response(404, json={"message": "no object found"})
        if path == "/surveys/ZTF/cutouts":
            return ok({"candid": q["candid"], "cutoutScience": fits_gz(1), "cutoutTemplate": fits_gz(2, source=0), "cutoutDifference": fits_gz(3, background=0)})
        if path == "/surveys/ZTF/objects/cone-search":
            return ok({"target": [{"objectId": "ZTF26aaaaaaa", "ra": 150.1001, "dec": 2.2, "survey": "ZTF"}]})
        if path == "/similarity/objects":
            body = json.loads(request.content)
            return ok([{"object_id": "ZTF26aaaaaaa", "distance": 0.123456}, {"object_id": "ZTF26zzzzzzz", "distance": 0.2}][: body["top_k"]])
        return httpx.Response(404, json={"message": f"unrouted {path}"})


def ok(data):
    return httpx.Response(200, json={"status": "success", "message": "ok", "data": data})


@pytest.fixture
def fake(monkeypatch):
    fake = FakeBabamul()
    tools._cache.clear()
    monkeypatch.setenv("BABAMUL_API_TOKEN", "bbml_test")
    monkeypatch.setenv("BABAMUL_ENV", "production")
    monkeypatch.setitem(settings, "http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake)))
    return fake


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeProviders:
    """The key checks of api.anthropic.com and openrouter.ai. A key in `good` passes;
    a refusal quotes the key back, as a provider may. `reply` is a status or an
    exception every check gets instead."""

    def __init__(self):
        self.good = {FAKE_ANTHROPIC_KEY, FAKE_OPENROUTER_KEY}
        self.requests = []
        self.reply = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        if request.url.host == "api.anthropic.com" and request.url.path == "/v1/messages/count_tokens":
            key = request.headers.get("x-api-key", "")
            refused = {"type": "error", "error": {"type": "authentication_error", "message": f"invalid x-api-key: {key}"}}
            good = {"input_tokens": 9}
        elif request.url.host == "openrouter.ai" and request.url.path == "/api/v1/key":
            key = request.headers.get("authorization", "").removeprefix("Bearer ")
            refused = {"error": {"code": 401, "message": f"No auth credentials found for {key}"}}
            good = {"data": {"limit": 5, "limit_remaining": 4.2, "usage": 0.8}}
        else:
            return httpx.Response(404, json={"error": {"message": f"unrouted {request.url.path}"}})
        if isinstance(self.reply, int):
            return httpx.Response(self.reply, json=refused)
        return httpx.Response(200, json=good) if key in self.good else httpx.Response(401, json=refused)


@pytest.fixture
def byok(fake, tmp_path):
    """An assistant whose people may bring their own key. byok(*script, shared=[...],
    config={...}, env={...}, **own_settings): script is what the providers' chat
    completions stream, shared what the operator's model at llm.test does."""
    from boom_mcp import assistant as a
    from boom_mcp import providers
    from boom_mcp.keystore import KeyStore
    from starlette.applications import Starlette

    from tests.test_assistant import FakeModel, say

    def build(*script, shared=None, config=None, env=None, **own_settings):
        own_model = FakeModel(*(script or [say("From your own key.")]))
        shared_model = FakeModel(*(shared or [say("From the shared model.")]))
        checks, hosts = FakeProviders(), []

        async def route(request):
            hosts.append(request.url.host)
            if request.url.host == "llm.test":
                return await shared_model(request)
            if request.url.host in ("api.anthropic.com", "openrouter.ai"):
                if request.url.path.endswith("/chat/completions"):
                    return await own_model(request)
                return checks(request)
            return fake(request)

        store = KeyStore(tmp_path / "assistant" / "keys.sqlite3", KEYS_SECRET)
        own = a.OwnKeys(store=store, providers=providers.from_env(env or {}), **own_settings)
        cfg = a.Config(llm=a.Endpoint("http://llm.test/v1", "test-model", "sk-model"), own_keys=own, **(config or {}))
        bot = a.Assistant(cfg, http=httpx.AsyncClient(transport=httpx.MockTransport(route)))
        return SimpleNamespace(
            bot=bot, app=a.mount(Starlette(), bot), store=store, own=own_model, shared=shared_model,
            checks=checks, babamul=fake, hosts=hosts,
        )

    return build
