import json

import pytest
from mcp.client import Client

from boom_mcp import summarize as s
from boom_mcp.app import mcp

from .conftest import JD0, OBJECT

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


async def test_similarity_describes_neighbors(fake):
    out = payload(await call("find_similar_objects", object_id="ZTF26xxxxxxx", top_k=2))
    first, second = out["neighbors"]
    assert first["applecider"] == "Ia-like SN" and first["tns"]["type"] == "SN Ia"
    assert "applecider" not in second  # 404 neighbors are listed, just not described


async def test_plugin_bounds_explosion_time(fake):
    out = payload(await call("estimate_explosion_time", object_id="ZTF26aaaaaaa"))
    assert out["window_days"] == pytest.approx(1.5)
    assert out["last_nondetection"]["limit"] == 20.8


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
