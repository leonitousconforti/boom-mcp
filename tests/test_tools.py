import json
import math
import threading
import time
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest
from mcp.client import Client

from boom_mcp import babamul, plots, tools
from boom_mcp.assistant import TOOL_SHARE, Config
from boom_mcp import summarize as s
from boom_mcp.app import caller, mcp, settings

from . import conftest
from .conftest import JD0, OBJECT, alert

pytestmark = pytest.mark.anyio


async def call(name, **args):
    async with Client(mcp) as client:
        result = await client.call_tool(name, args)
    return result


def payload(result):
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


def test_time_round_trip():
    jd = s.to_jd("2026-11-10T20:45:00Z")
    assert s.jd_to_iso(jd) == "2026-11-10T20:45:00Z"
    assert s.to_jd(str(jd - 2400000.5)) == pytest.approx(jd)
    with pytest.raises(ValueError):
        s.to_jd("12")
    # Every whole second comes back as itself, not one second early.
    start = s.to_jd("2026-11-10T00:00:00Z")
    for seconds in range(0, 86400, 7):
        iso = s.jd_to_iso(start + seconds / 86400)
        assert s.jd_to_iso(s.to_jd(iso)) == iso and iso.endswith(f"{seconds % 60:02d}Z"), iso


async def test_convert_time_echoes_whole_seconds(fake):
    out = payload(await call("convert_time", value="2026-11-06T04:00:00Z"))
    assert out["iso"] == "2026-11-06T04:00:00Z"
    assert payload(await call("convert_time", value="61353.30"))["iso"] == "2026-11-09T07:12:00Z"


async def test_unix_seconds_and_infinities_are_refused_with_a_reason(fake):
    for value in ("1762400000", "20261106", "2100000"):
        with pytest.raises(ValueError, match="unix seconds"):
            s.to_jd(value)
    for value in ("inf", "1e300", "nan", "1762400000"):
        result = await call("convert_time", value=value)
        assert result.is_error and "neither a JD" in result.content[0].text, value
    result = await call("search_alerts", start="1762400000")
    assert result.is_error and "neither a JD" in result.content[0].text
    assert not fake.requests


async def test_lists_tools_with_instructions(fake):
    async with Client(mcp) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
    assert {"search_alerts", "get_object", "get_cutouts", "find_similar_objects", "estimate_explosion_time"} <= tools


async def test_search_alerts_filters_and_counts(fake):
    out = payload(await call("search_alerts", start=str(JD0), end=str(JD0 + 1)))
    assert out["scanned"] == 4 and out["matched"] == 4 and out["objects"] == 3
    # Newest first, one row per object, candids as strings.
    assert [a["object_id"] for a in out["alerts"]] == ["ZTF26ccccccc", "ZTF26bbbbbbb", "ZTF26aaaaaaa"]
    assert out["alerts"][2]["candid"] == "2"

    out = payload(await call("search_alerts", start=str(JD0), anomalous_only=True))
    assert [a["object_id"] for a in out["alerts"]] == ["ZTF26bbbbbbb"]
    assert out["alerts"][0]["ml"]["anomalous"] is True

    out = payload(await call("search_alerts", start=str(JD0), applecider_class="Ia-like SN", sort_by="class_probability"))
    assert out["matched"] == 3 and out["objects"] == 2


async def test_search_alerts_rejects_long_windows_and_reports_truncation(fake):
    result = await call("search_alerts", start=str(JD0), end=str(JD0 + 5))
    assert result.is_error and "at most 1 day" in result.content[0].text
    out = payload(await call("search_alerts", start=str(JD0), max_scanned=100))
    assert not out["truncated"]


# Search caching, sharing and retries -------------------------------------------


def hour_start(jd):
    return 0.5 + math.floor((jd - 0.5) * 24) / 24


H0 = hour_start(JD0 + 0.1)  # 07:00Z, the hour holding the first fixture alert


def busy(h0, n=150, first=10_000):
    """n alerts spread through the hour starting at h0, in candid (and time) order."""
    return [alert(f"ZTF26busy{first + k}", first + k, h0 + (k + 0.5) / n / 24, 18.0) for k in range(n)]


def within(h0, hours=1):
    """A window inside the given whole hours, clear of the float edges."""
    return {"start": str(h0 + 0.001 / 24), "end": str(h0 + (hours - 0.001) / 24)}


def alert_requests(fake):
    return [r for r in fake.requests if r.url.path.endswith("/alerts")]


def serve(monkeypatch, handler):
    monkeypatch.setitem(settings, "http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.fixture
def per_user(monkeypatch):
    monkeypatch.delenv("BOOM_MCP_SHARED_ALERT_CACHE", raising=False)


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(babamul, "RETRY_DELAYS_S", (0.0, 0.0))


@pytest.fixture
def slow(fake, monkeypatch):
    """The fake, with /alerts taking long enough for searches to overlap."""

    async def handler(request):
        if request.url.path.endswith("/alerts"):
            await anyio.sleep(0.05)
        return fake(request)

    serve(monkeypatch, handler)
    return fake


async def test_cache_is_per_token_by_default(fake, per_user, monkeypatch):
    fake.tokens["bbml_other"] = "user-2"
    first = payload(await call("search_alerts", **within(H0)))
    n = len(alert_requests(fake))
    assert n == 1 and first["scanned"] == 1
    assert payload(await call("search_alerts", **within(H0))) == first
    assert len(alert_requests(fake)) == n  # the same person again: from the cache

    monkeypatch.setenv("BABAMUL_API_TOKEN", "bbml_other")
    assert payload(await call("search_alerts", **within(H0))) == first
    assert len(alert_requests(fake)) == 2 * n  # someone else: fetched with their own token
    assert alert_requests(fake)[-1].headers["authorization"] == "Bearer bbml_other"
    # Keys hold a hash of the token, never the token.
    assert tools._cache and "bbml" not in repr(list(tools._cache))
    assert not [r for r in fake.requests if r.url.path.endswith("/profile")]


async def test_shared_cache_is_opt_in_and_still_checks_the_token(fake, monkeypatch):
    monkeypatch.setenv("BOOM_MCP_SHARED_ALERT_CACHE", "1")
    monkeypatch.setattr(babamul, "_accepted", {})
    fake.tokens["bbml_other"] = "user-2"
    first = payload(await call("search_alerts", **within(H0)))
    monkeypatch.setenv("BABAMUL_API_TOKEN", "bbml_other")
    assert payload(await call("search_alerts", **within(H0))) == first
    assert len(alert_requests(fake)) == 1  # the second person shares the first fetch
    profiles = [r.headers["authorization"] for r in fake.requests if r.url.path.endswith("/profile")]
    assert profiles == ["Bearer bbml_test", "Bearer bbml_other"]

    # A token Babamul refuses gets the refusal, not someone else's alerts.
    monkeypatch.setenv("BABAMUL_API_TOKEN", "bbml_not_a_user")
    result = await call("search_alerts", **within(H0))
    assert result.is_error and "rejected the API token" in result.content[0].text
    assert len(alert_requests(fake)) == 1


async def test_identical_searches_at_once_fetch_each_hour_once(slow, per_user):
    window = {"start": str(JD0), "end": str(JD0 + 0.25)}
    hours = len(tools._hours(JD0, JD0 + 0.25))
    outs = []

    async def search():
        outs.append(payload(await call("search_alerts", **window)))

    async with anyio.create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(search)
    assert len(alert_requests(slow)) == hours
    assert all(out == outs[0] for out in outs) and outs[0]["scanned"] == 2


async def test_different_people_at_once_do_not_share_unless_asked(slow, monkeypatch):
    slow.tokens["bbml_other"] = "user-2"
    window = {"start": str(JD0), "end": str(JD0 + 0.25)}
    hours = len(tools._hours(JD0, JD0 + 0.25))

    async def search(token):
        async with acting_as(token):
            payload(await call("search_alerts", **window))

    for shared, expected in (("0", 2 * hours), ("1", hours)):
        monkeypatch.setenv("BOOM_MCP_SHARED_ALERT_CACHE", shared)
        monkeypatch.setattr(babamul, "_accepted", {})
        tools._cache.clear()
        slow.requests.clear()
        async with anyio.create_task_group() as tg:
            for token in ("bbml_test", "bbml_other"):
                tg.start_soon(search, token)
        requests = alert_requests(slow)
        assert len(requests) == expected, shared
        if shared == "0":
            assert {r.headers["authorization"] for r in requests} == {"Bearer bbml_test", "Bearer bbml_other"}


async def test_a_stopped_fetch_hands_over_and_a_failed_one_is_shared(slow, per_user, monkeypatch):
    window = tools._hours(JD0 + 0.1, JD0 + 0.11)[0]
    async with settings["http_client"]() as http:
        bm = babamul.Babamul(http, "bbml_test", babamul.base_url_from_env())
        results = []

        async def leader():
            with anyio.move_on_after(0.02):  # stopped mid-fetch, like a person pressing Stop
                await tools._fetch(bm, "ZTF", {}, *window, budget=100)

        async def follower():
            await anyio.sleep(0.01)
            results.append(await tools._fetch(bm, "ZTF", {}, *window, budget=100))

        async with anyio.create_task_group() as tg:
            tg.start_soon(leader)
            tg.start_soon(follower)
        assert [(len(alerts), complete) for alerts, complete in results] == [(1, True)]
        assert not tools._inflight

    # A failure is the same for everyone waiting on it, so it is not fetched again.
    refused = []

    async def refuse(request):
        refused.append(request)
        await anyio.sleep(0.05)
        return httpx.Response(400, json={"message": "bad magnitude"})

    serve(monkeypatch, refuse)
    errors = []
    async with settings["http_client"]() as http:
        bm = babamul.Babamul(http, "bbml_test", babamul.base_url_from_env())

        async def search():
            try:
                await tools._fetch(bm, "ZTF", {"min_magpsf": 99}, *window, budget=100)
            except babamul.BabamulError as e:
                errors.append(str(e))

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(search)
    assert errors == ["Babamul error 400: bad magnitude"] * 3
    assert len(refused) == 1 and not tools._inflight


@asynccontextmanager
async def acting_as(token):
    """Run tools in this task as the person holding token, the way the assistant does."""
    async with settings["http_client"]() as http:
        reset = caller.set(babamul.Babamul(http, token, babamul.base_url_from_env()))
        try:
            yield
        finally:
            caller.reset(reset)


async def test_busy_hours_are_cached_and_resumed(fake, per_user, monkeypatch):
    monkeypatch.setattr(conftest, "ALERTS", busy(H0))
    out = payload(await call("search_alerts", **within(H0), max_scanned=100))
    assert out["scanned"] == 100 and out["truncated"] and "max_scanned (100)" in out["note"]
    assert [r.url.params["limit"] for r in alert_requests(fake)] == ["100"]

    # The same search again is served from the hour cut short.
    assert payload(await call("search_alerts", **within(H0), max_scanned=100)) == out
    assert len(alert_requests(fake)) == 1

    # A bigger budget carries on where the cut hour stopped.
    out = payload(await call("search_alerts", **within(H0), max_scanned=1000))
    assert out["scanned"] == 150 and not out["truncated"]
    last = alert_requests(fake)[-1].url.params
    assert (last["skip"], last["limit"]) == ("100", "900")


async def test_truncated_when_an_hour_hits_the_budget_inside_a_short_window(fake, per_user, monkeypatch):
    # 150 alerts in the hour, the second half asked for: the budget of 100 fetches
    # the first 100 by candid, so 50 of the 75 in the window are never seen.
    monkeypatch.setattr(conftest, "ALERTS", busy(H0))
    out = payload(await call("search_alerts", start=str(H0 + 0.5 / 24), end=str(H0 + 0.999 / 24), max_scanned=100))
    assert out["scanned"] == 25 and out["truncated"] and "max_scanned" in out["note"]


async def test_each_batch_fetches_only_the_budget_left(fake, per_user, monkeypatch):
    monkeypatch.setattr(conftest, "ALERTS", busy(H0) + busy(H0 + 1 / 24, first=20_000) + busy(H0 + 2 / 24, first=30_000))
    out = payload(await call("search_alerts", **within(H0, hours=3), max_scanned=100))
    assert out["scanned"] == 100 and out["truncated"]
    assert [r.url.params["limit"] for r in alert_requests(fake)] == ["100"]  # not 3 x 100

    tools._cache.clear()
    fake.requests.clear()
    out = payload(await call("search_alerts", **within(H0, hours=3), max_scanned=200))
    assert out["scanned"] == 200 and out["truncated"]
    assert [r.url.params["limit"] for r in alert_requests(fake)] == ["200", "50"]


async def test_cached_alerts_keep_only_what_search_alerts_reads(fake, per_user):
    full = {**conftest.ALERTS[0], "cutoutScience": "x" * 100}
    full["candidate"] = {**full["candidate"], "magzpsci": 26.3, "field": 600}
    full["properties"] = {**full["properties"], "large_blob": list(range(100))}
    assert s.alert_row(tools._slim(full), "ZTF") == s.alert_row(full, "ZTF")
    for a in conftest.ALERTS:
        assert s.alert_row(tools._slim(a), "ZTF") == s.alert_row(a, "ZTF")
    payload(await call("search_alerts", **within(H0)))
    (cached,) = [entry[1][0] for entry in tools._cache.values() if entry[1]]
    assert "programid" not in cached["candidate"] and set(cached["candidate"]) <= tools.CANDIDATE_FIELDS


def test_cache_is_bounded_by_alerts_held_and_drops_expired_entries(monkeypatch):
    monkeypatch.setattr(tools, "_cache", {})
    monkeypatch.setattr(tools, "CACHE_MAX_ALERTS", 10)
    tools._cache["old"] = (time.monotonic() - tools.CACHE_TTL_S - 1, [0], True)
    tools._store("a", [0] * 6, True)
    assert "old" not in tools._cache
    tools._store("b", [0] * 3, False)
    tools._store("c", [0] * 4, True)
    assert list(tools._cache) == ["b", "c"]  # a went, oldest first, to make room
    tools._store("huge", [0] * 11, True)
    assert "huge" not in tools._cache and sum(len(e[1]) for e in tools._cache.values()) <= 10


@pytest.mark.parametrize("kinds", [(503, "timeout"), (504, 429)])
async def test_busy_babamul_is_retried(fake, per_user, no_wait, monkeypatch, kinds):
    failures = [httpx.ReadTimeout("timed out") if kind == "timeout" else kind for kind in kinds]

    def handler(request):
        if request.url.path.endswith("/alerts") and failures:
            fake.requests.append(request)
            failure = failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, json={"message": "Too many large queries are running; retry in a minute"})
        return fake(request)

    serve(monkeypatch, handler)
    out = payload(await call("search_alerts", **within(H0)))
    assert out["scanned"] == 1
    assert len(alert_requests(fake)) == 3  # two failures, then the answer


async def test_babamul_errors_in_a_time_search_reach_the_model(fake, per_user, no_wait, monkeypatch):
    def handler(request):
        if request.url.path.endswith("/alerts"):
            fake.requests.append(request)
            return httpx.Response(503, json={"message": "Too many large queries are running; retry in a minute"})
        return fake(request)

    serve(monkeypatch, handler)
    result = await call("search_alerts", **within(H0, hours=2))
    assert result.is_error
    assert "Babamul error 503" in result.content[0].text and "retry in a minute" in result.content[0].text
    assert len(alert_requests(fake)) >= 3  # retried before giving up
    assert not tools._inflight


def test_retries_stop_at_the_time_limit(monkeypatch):
    now = time.monotonic()
    assert babamul._retry_delay(0, now, now) is not None
    assert babamul._retry_delay(len(babamul.RETRY_DELAYS_S), now, now) is None
    # A 503 after BOOM's 60 s wait for a slot is retried once, then given up.
    assert babamul._retry_delay(0, now - 60, now - 60) is not None
    assert babamul._retry_delay(1, now - 123, now - 60) is None
    # A try that ran into the 120 s timeout could not finish again in time.
    assert babamul._retry_delay(0, now - 120, now - 120) is None
    monkeypatch.setattr(babamul, "RETRY_WITHIN_S", 1.0)
    assert babamul._retry_delay(0, now, now) is None


# One real second stands for this many seconds of Babamul time.
SCALE = 100


@pytest.mark.parametrize(
    "script, message, last_limit",
    [
        ([(0, 503)] * 3, "retry in a minute", 120),
        ([(60, 503)] * 2, "retry in a minute", 87.5),
        ([(500, 200)], "did not answer", 120),
        # BOOM gives the retry a slot late and the query then runs long: the
        # retry is cut off at 150 s rather than running to its own 120 s.
        ([(60, 503), (500, 200)], "did not answer", 87.5),
    ],
    ids=["fast 503", "503 after the slot wait", "timeout", "slow query after a 503"],
)
async def test_retries_end_inside_the_assistants_tool_time(monkeypatch, script, message, last_limit):
    """script is (Babamul seconds, status) per try; a try longer than its timeout times out."""
    monkeypatch.setattr(babamul, "RETRY_DELAYS_S", (2.0 / SCALE, 5.0 / SCALE))
    monkeypatch.setattr(babamul, "RETRY_WITHIN_S", babamul.RETRY_WITHIN_S / SCALE)
    limits = []

    async def handler(request):
        seconds, status = script[min(len(limits), len(script) - 1)]
        # MockTransport ignores timeouts, so act out the one httpx was given.
        limit = request.extensions["timeout"]["read"] * SCALE
        limits.append(limit)
        await anyio.sleep(min(seconds, limit) / SCALE)
        if seconds > limit:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(status, json={"message": "Too many large queries are running; retry in a minute"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = babamul.Babamul(http, "bbml_test", "https://babamul.invalid")
        started = time.monotonic()
        with pytest.raises(babamul.BabamulError) as raised:
            await client.request("GET", "/surveys/ZTF/alerts", timeout=120.0 / SCALE)
        took = (time.monotonic() - started) * SCALE

    assert len(limits) == len(script)
    assert limits[-1] == pytest.approx(last_limit, abs=2)
    assert took < Config.answer_timeout * TOOL_SHARE
    assert message in str(raised.value)


async def test_get_object_summary(fake):
    out = payload(await call("get_object", object_id="ZTF26aaaaaaa"))
    assert out["latest_alert"]["candid"] == "3123456789012345678"
    assert out["ml"]["applecider"]["decision"] == "Ia-like SN"
    assert out["ml"]["acai"]["hosted"] == 0.91
    assert out["applecider_history"] == ["abstained x1", "Ia-like SN x2"]
    assert out["cross_matches"]["TNS"]["nearest"]["name"] == "2026abc"
    assert "Gaia_DR3" not in out["cross_matches"]
    r = out["bands"]["ztfr"]
    assert r["peak"]["mag"] == 17.6 and r["last_limit_before_first_detection"]["limit"] == 20.8
    assert out["counterparts"]["LSST"]["object_id"] == "3141592653589793"


async def test_light_curve_merges_surveys_and_plots(fake):
    result = await call("get_light_curve", object_id="ZTF26aaaaaaa", plot=True)
    assert not result.is_error
    header, csv, image = result.content
    assert "lsstg" in json.loads(header.text)["bands"]
    assert csv.text.startswith("jd,band,mag")
    assert image.mime_type == "image/png" and len(image.data) > 1000


async def test_cutouts_render(fake):
    result = await call("get_cutouts", object_id="ZTF26aaaaaaa")
    assert not result.is_error
    assert result.content[1].mime_type == "image/png"


async def test_plots_render_off_the_event_loop(fake, monkeypatch):
    threads = []
    for name in ("light_curve", "cutout_triplet"):
        real = getattr(plots, name)

        def spy(*args, real=real, **kwargs):
            threads.append(threading.current_thread() is threading.main_thread())
            return real(*args, **kwargs)

        monkeypatch.setattr(plots, name, spy)
    assert not (await call("get_light_curve", object_id="ZTF26aaaaaaa", plot=True)).is_error
    assert not (await call("get_cutouts", object_id="ZTF26aaaaaaa")).is_error
    assert threads == [False, False]


async def test_cutouts_of_an_object_without_an_alert_say_so(fake, monkeypatch):
    monkeypatch.setattr(conftest, "OBJECT", {k: v for k, v in OBJECT.items() if k != "candid"})
    result = await call("get_cutouts", object_id="ZTF26aaaaaaa")
    assert result.is_error and "no alert" in result.content[0].text
    assert not [r for r in fake.requests if r.url.path.endswith("/cutouts")]


async def test_similarity_describes_neighbors(fake):
    out = payload(await call("find_similar_objects", object_id="ZTF26xxxxxxx", top_k=2))
    first, second = out["neighbors"]
    assert first["applecider"] == "Ia-like SN" and first["tns"]["type"] == "SN Ia"
    assert "applecider" not in second  # 404 neighbors are listed, just not described


async def test_plugin_bounds_explosion_time(fake):
    out = payload(await call("estimate_explosion_time", object_id="ZTF26aaaaaaa"))
    assert out["window_days"] == pytest.approx(1.5)
    assert out["last_nondetection"]["limit"] == 20.8


def forced(sigma, days_before):
    return {"jd": JD0 - days_before, "band": "g", "diffmaglim": 20.6, "psfFlux": sigma * 1e9, "psfFluxErr": 1e9}


async def test_plugin_ignores_a_3_sigma_forced_blip_but_not_a_5_sigma_one(fake, monkeypatch):
    monkeypatch.setattr(conftest, "OBJECT", {**OBJECT, "fp_hists": [forced(3.1, 30)]})
    out = payload(await call("estimate_explosion_time", object_id="ZTF26aaaaaaa"))
    assert out["window_days"] == pytest.approx(1.5) and out["first_detection"]["band"] == "ztfr"

    monkeypatch.setattr(conftest, "OBJECT", {**OBJECT, "fp_hists": [forced(8, 30)]})
    out = payload(await call("estimate_explosion_time", object_id="ZTF26aaaaaaa"))
    assert out["first_detection"]["band"] == "ztfg" and out["verdict"].startswith("unconstrained")


async def test_object_ids_cannot_change_the_url_path(fake):
    for tool, args in (
        ("get_object", {"object_id": "../../../tokens"}),
        ("get_object", {"object_id": ".."}),
        ("get_cross_matches", {"object_id": "a/b"}),
        ("get_light_curve", {"object_id": "ZTF26aaaaaaa/../../../admin/users"}),
        ("estimate_explosion_time", {"object_id": "../../profile"}),
    ):
        result = await call(tool, **args)
        assert result.is_error and "not an object id" in result.content[0].text, (tool, args)
    result = await call("estimate_explosion_time", object_id="ZTF26aaaaaaa", survey="../admin")
    assert result.is_error
    assert not fake.requests
    # Ordinary ids, with stray spaces, still work.
    assert payload(await call("get_object", object_id=" ZTF26aaaaaaa "))["object_id"] == "ZTF26aaaaaaa"
    assert babamul.checked_object_id("3141592653589793") == "3141592653589793"


async def test_errors_reach_the_model(fake, monkeypatch):
    result = await call("get_object", object_id="ZTF00nothere")
    assert result.is_error and "Not found" in result.content[0].text
    monkeypatch.setenv("BABAMUL_API_TOKEN", "wrong")
    result = await call("get_object", object_id="ZTF26aaaaaaa")
    assert result.is_error and "rejected the API token" in result.content[0].text


def test_lsst_flux_conversion():
    rows = s.light_curve_rows(OBJECT, "ZTF", other_survey=True)
    lsst = next(r for r in rows if r["band"] == "lsstg")
    assert lsst["mag"] == pytest.approx(31.4 - 2.5 * 3.69897, abs=0.01)  # 5000 nJy


def test_forced_photometry_noise_is_not_a_detection():
    # Real fp_hists points of ZTF26abypkdp: BOOM sends fluxes only, no magpsf or snr.
    noise = {"jd": 2461296.812419, "band": "g", "diffmaglim": 20.6517, "psfFlux": -552607300.0, "psfFluxErr": 2522164500.0}
    faint = {"jd": 2461297.7733218, "band": "g", "diffmaglim": 20.461, "psfFlux": 4103484700.0, "psfFluxErr": 3022670600.0}
    real = {"jd": 2461320.0, "band": "r", "diffmaglim": 20.5, "psfFlux": 1.0e11, "psfFluxErr": 3.0e9}
    rows = s.light_curve_rows({"fp_hists": [noise, faint, real]}, "ZTF", forced=True)
    assert [("mag" in r, r.get("limit")) for r in rows] == [(False, 20.652), (False, 20.461), (True, None)]
    assert rows[2]["mag"] == pytest.approx(23.9 - 2.5 * 2, abs=0.01)


def test_catalog_sentinels_are_dropped():
    out = s.xmatch_summary({"LSDR10": [{"objtype": "REX", "z_spec": -99.0, "survey": " ", "distance_arcsec": 0.41}]})
    assert out["LSDR10"]["nearest"] == {"objtype": "REX", "distance_arcsec": 0.41}


def test_empty_babamul_env_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("BABAMUL_API_URL", "  ")
    monkeypatch.setenv("BABAMUL_ENV", "")
    assert babamul.base_url_from_env() == babamul.API_URLS["production"]
    monkeypatch.setenv("BABAMUL_ENV", " Local ")
    assert babamul.base_url_from_env() == babamul.API_URLS["local"]
    monkeypatch.setenv("BABAMUL_API_URL", " http://127.0.0.1:4100/babamul/ ")
    assert babamul.base_url_from_env() == "http://127.0.0.1:4100/babamul"


@pytest.mark.parametrize(
    ("survey", "now", "last", "tonight"),
    [
        # ZTF, morning after: the night that just ended, nothing under way.
        ("ZTF", "2026-11-07T16:30", ("2026-11-07T02:00Z", "2026-11-07T14:00Z"), None),
        # ZTF, during the night: last night is the previous one.
        ("ZTF", "2026-11-07T03:00", ("2026-11-06T02:00Z", "2026-11-06T14:00Z"), "2026-11-07T02:00Z"),
        # Rubin's night crosses midnight UTC.
        ("LSST", "2026-11-07T12:00", ("2026-11-06T23:00Z", "2026-11-07T10:00Z"), None),
        ("LSST", "2026-11-07T23:30", ("2026-11-06T23:00Z", "2026-11-07T10:00Z"), "2026-11-07T23:00Z"),
    ],
)
def test_last_night_has_explicit_dates(survey, now, last, tonight):
    from datetime import UTC, datetime

    from boom_mcp import summarize

    w = summarize.night_windows(survey, datetime.fromisoformat(now).replace(tzinfo=UTC))
    assert (w["last_night"]["start"], w["last_night"]["end"]) == last
    assert (w.get("tonight_so_far") or {}).get("start") == tonight


async def test_convert_time_now_names_last_night(fake):
    out = payload(await call("convert_time"))
    assert set(out["nights"]) == {"ZTF", "LSST"}
    assert out["nights"]["ZTF"]["last_night"]["end"].endswith("T14:00Z")
    assert "nights" not in payload(await call("convert_time", value="2026-11-06T04:00:00Z"))


@pytest.mark.parametrize(
    ("text", "iso"),
    [("MJD 61353.30", "2026-11-09T07:12:00Z"), ("mjd:61353.3", "2026-11-09T07:12:00Z"), ("JD 2461353.8", "2026-11-09T07:12:00Z")],
)
def test_to_jd_takes_mjd_and_jd_prefixes(text, iso):
    from boom_mcp import summarize

    assert summarize.jd_to_iso(summarize.to_jd(text)) == iso


def test_applecider_history_tells_missing_from_abstained():
    from boom_mcp import summarize

    obj = {
        "objectId": "ZTF26x", "candidate": {"jd": 2461353.8, "ra": 1.0, "dec": 2.0, "magpsf": 14.0, "fid": 2},
        "prv_candidates": [], "classifications_history": [
            {"btsbot": 0.1},
            {"applecider_fusion": {"abstain_completely": True}},
            {"applecider_fusion": {"decision_label": "SN Ia"}},
        ],
    }
    out = summarize.object_summary(obj, "ZTF")
    assert out["applecider_history"] == ["no AppleCiDEr result x1", "abstained x1", "SN Ia x1"]
