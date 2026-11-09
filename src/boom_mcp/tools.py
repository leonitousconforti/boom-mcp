"""The BOOM tools: thin wrappers over Babamul that return what a person would look at."""

import base64
import math
import os
import time
from typing import Annotated, Literal

import anyio
from mcp.server.mcpserver import Context, Image
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from . import plots
from . import summarize as s
from .app import INSTRUCTIONS, babamul, mcp, tool
from .babamul import BabamulError, checked_object_id

Survey = Literal["ZTF", "LSST"]
AppleCiderClass = Literal[
    "Ia-like SN",
    "Stripped Envelope SN",
    "H-rich CCSN",
    "Superluminous SN",
    "TDE",
    "AGN-like",
    "Accreting WD Var",
    "Other Stellar Var",
]
SortKey = Literal["time", "brightness", "reliability", "class_probability", "anomaly_score", "btsbot"]

PAGE_SIZE = 5000
# BOOM's /alerts times out (504 after 60 s) on windows of a few hours, while an
# hour returns in about a second, so time searches go out one UTC hour at a time.
MAX_WINDOW_DAYS = 1.0
CONCURRENCY = 3

# Finished hours are cached for a while, including hours cut short by the budget,
# which a later search with the same or a smaller budget can use. An entry
# belongs to the token that fetched it, so one person never sees what was
# fetched for another, unless BOOM_MCP_SHARED_ALERT_CACHE is on.
CACHE_TTL_S = 600
CACHE_SIZE = 256
# Alerts held across all entries. A cut-down alert is a few KB in memory, and
# one uncut ZTF hour is about 20,000 alerts.
CACHE_MAX_ALERTS = 200_000
# key -> (stored at, alerts, whether that is every alert of the window)
_cache: dict[tuple, tuple[float, list, bool]] = {}
_inflight: dict[tuple, "_Flight"] = {}

# What search_alerts reads of an alert: the window filter, the sort keys and
# summarize.alert_row. A full candidate has about 100 fields.
CANDIDATE_FIELDS = frozenset(
    "jd ra dec magpsf sigmapsf drb reliability fid band isdiffpos isNegative ndethist jdstarthist".split()
)
PROPERTY_FIELDS = frozenset("star rock near_brightstar stationary hosted sso best_host".split())

# matplotlib's caches are not built for several threads drawing at once, and
# drawing holds the GIL anyway, so renders run one at a time, off the event loop.
_render_limiter = anyio.CapacityLimiter(1)


def _check_ml(survey: str, *requested):
    if survey != "ZTF" and any(r not in (None, False) for r in requested):
        raise ToolError("BOOM computes ACAI, BTSbot and AppleCiDEr outputs for ZTF alerts only.")


def _hours(start_jd: float, end_jd: float) -> list[tuple[float, float]]:
    """The whole UTC hours covering [start_jd, end_jd], aligned so callers share cache entries."""
    if end_jd <= start_jd:
        raise ToolError(f"end ({s.jd_to_iso(end_jd)}) is not after start ({s.jd_to_iso(start_jd)}).")
    if end_jd - start_jd > MAX_WINDOW_DAYS + 1e-6:
        raise ToolError(
            f"A time-only search covers at most {MAX_WINDOW_DAYS:g} day; this one is "
            f"{end_jd - start_jd:.1f}. Search one night at a time, or by position or object_id."
        )
    first, last = math.floor((start_jd - 0.5) * 24), math.ceil((end_jd - 0.5) * 24)
    return [(0.5 + h / 24, 0.5 + (h + 1) / 24) for h in range(first, last)]


def _shared_cache() -> bool:
    """Whether cached alerts are shared between tokens (BOOM_MCP_SHARED_ALERT_CACHE=1).

    Safe today: Babamul's alert search never reads who is asking (BOOM
    src/api/routes/babamul/surveys/alerts.rs:113-118 leaves _current_user unused)
    and forces public data (candidate.programid 1, alerts.rs:131-135), so every
    token gets the same alerts. Sharing lets a room asking the same question at
    once cost one fetch. Off by default in case Babamul starts answering per user.
    """
    return os.environ.get("BOOM_MCP_SHARED_ALERT_CACHE", "").strip().lower() in ("1", "true", "yes", "on")


def _slim(alert: dict) -> dict:
    """alert with only the fields search_alerts reads, so cached hours stay small."""
    out = {k: alert[k] for k in ("candid", "_id", "objectId", "object_id", "classifications") if k in alert}
    out["candidate"] = {k: v for k, v in (alert.get("candidate") or {}).items() if k in CANDIDATE_FIELDS}
    if (props := alert.get("properties")) is not None:
        out["properties"] = {k: v for k, v in props.items() if k in PROPERTY_FIELDS}
    return out


def _store(key: tuple, alerts: list, complete: bool):
    """Cache alerts under key, dropping expired entries, then the oldest, to stay within bounds."""
    now = time.monotonic()
    for k in [k for k, entry in _cache.items() if now - entry[0] >= CACHE_TTL_S]:
        del _cache[k]
    _cache.pop(key, None)
    if len(alerts) > CACHE_MAX_ALERTS:
        return
    held = sum(len(entry[1]) for entry in _cache.values())
    while _cache and (len(_cache) >= CACHE_SIZE or held + len(alerts) > CACHE_MAX_ALERTS):
        held -= len(_cache.pop(next(iter(_cache)))[1])
    _cache[key] = (now, alerts, complete)


class _Flight:
    """A window being fetched right now; callers asking for the same one wait for it."""

    def __init__(self):
        self.done = anyio.Event()
        self.result: tuple[list, bool] | None = None
        self.error: BabamulError | None = None


async def _fetch(bm, survey: str, params: dict, start_jd, end_jd, budget: int) -> tuple[list, bool]:
    """The alerts matching params in one window, page by page, up to budget, and
    whether they are all there is."""
    shared = _shared_cache()
    if shared:
        # Shared entries were fetched with someone else's token: only hand them
        # to a token Babamul accepts.
        await bm.check_token()
    key = (None if shared else bm.identity, bm.base_url, survey, tuple(sorted(params.items())), start_jd, end_jd)
    have, complete = [], False
    while True:
        if (hit := _cache.get(key)) and time.monotonic() - hit[0] < CACHE_TTL_S and len(hit[1]) >= len(have):
            have, complete = hit[1], hit[2]
        if complete or len(have) >= budget:
            return have[:budget], complete and len(have) <= budget
        if (flight := _inflight.get(key)) is None:
            break
        # Someone is fetching this window right now: wait for theirs rather than
        # ask Babamul twice. If it was cut short or stopped, look again.
        await flight.done.wait()
        if flight.error is not None:
            raise BabamulError(str(flight.error))
        if flight.result is not None and len(flight.result[0]) >= len(have):
            have, complete = flight.result

    flight = _inflight[key] = _Flight()
    try:
        # BOOM returns alerts sorted by _id, so pages carry on from what is already here.
        alerts, complete = list(have), False
        while len(alerts) < budget:
            page_size = min(PAGE_SIZE, budget - len(alerts))
            page = await bm.get(
                f"/surveys/{survey}/alerts",
                **params,
                start_jd=start_jd,
                end_jd=end_jd,
                limit=page_size,
                skip=len(alerts),
            )
            alerts.extend(_slim(a) for a in page)
            if len(page) < page_size:
                complete = True
                break
        if start_jd is not None and end_jd is not None and end_jd < s.now_jd():
            _store(key, alerts, complete)
        flight.result = (alerts, complete)
        return alerts, complete
    except BabamulError as e:
        flight.error = e
        raise
    finally:
        del _inflight[key]
        flight.done.set()


@tool()
async def convert_time(
    ctx: Context,
    value: Annotated[
        str | None,
        Field(description='ISO date/time (UTC), JD, MJD, or "now". Omit for the current time.'),
    ] = None,
) -> dict:
    """Convert a time between ISO (UTC), Julian Date and MJD, and say how long ago it was.

    Call it with no argument to get the current time, e.g. before working out what
    "last night" or "this week" means for a query.
    """
    try:
        out = s.describe_time(s.to_jd(value if value is not None else "now"))
    except (ValueError, OverflowError, OSError) as e:
        raise ToolError(str(e)) from e
    if value is None or value.strip().lower() == "now":
        out["nights"] = {survey: s.night_windows(survey) for survey in s.NIGHTS_UTC}
    return out


@tool()
async def search_alerts(
    ctx: Context,
    survey: Survey = "ZTF",
    start: Annotated[str | None, Field(description="Window start: ISO (UTC), JD or MJD.")] = None,
    end: Annotated[str | None, Field(description="Window end. Defaults to start + 1 day.")] = None,
    ra: Annotated[float | None, Field(description="Cone search center RA, degrees.")] = None,
    dec: Annotated[float | None, Field(description="Cone search center Dec, degrees.")] = None,
    radius_arcsec: Annotated[float | None, Field(description="Cone radius, at most 600 arcsec.", gt=0, le=600)] = None,
    object_id: Annotated[str | None, Field(description="Only alerts of this object.")] = None,
    min_mag: Annotated[float | None, Field(description="Brightest magnitude to keep (smaller is brighter).")] = None,
    max_mag: Annotated[float | None, Field(description="Faintest magnitude to keep.")] = None,
    min_reliability: Annotated[float | None, Field(description="Minimum real/bogus score, 0-1.", ge=0, le=1)] = None,
    positive_only: Annotated[bool, Field(description="Only brightenings relative to the reference.")] = True,
    exclude_stars: Annotated[bool, Field(description="Drop alerts BOOM flags as stellar.")] = False,
    exclude_solar_system: Annotated[bool, Field(description="Drop alerts BOOM flags as asteroids.")] = False,
    exclude_near_bright_stars: bool = False,
    stationary_only: Annotated[
        bool, Field(description="Only sources detected at least twice, >30 min apart, so not moving.")
    ] = False,
    max_host_distance_mpc: Annotated[
        float | None, Field(description="Only alerts whose best host galaxy is closer than this.")
    ] = None,
    applecider_class: Annotated[
        AppleCiderClass | None, Field(description="Keep alerts AppleCiDEr gives this class (ZTF only).")
    ] = None,
    min_class_probability: Annotated[
        float, Field(description="Probability applecider_class must reach.", ge=0, le=1)
    ] = 0.5,
    anomalous_only: Annotated[
        bool, Field(description="Only alerts AppleCiDEr flags as out-of-distribution (ZTF only).")
    ] = False,
    min_btsbot: Annotated[float | None, Field(description="Minimum BTSbot score (ZTF only).", ge=0, le=1)] = None,
    one_per_object: Annotated[bool, Field(description="Keep only the best-ranked alert of each object.")] = True,
    sort_by: SortKey = "time",
    limit: Annotated[int, Field(description="Rows to return.", ge=1, le=200)] = 20,
    max_scanned: Annotated[
        int, Field(description="Stop after fetching this many alerts from BOOM.", ge=100, le=100_000)
    ] = 20_000,
) -> dict:
    """Search BOOM alerts by time window, position or object, with the cuts BOOM can apply
    itself (magnitude, reliability, star/asteroid/bright-star flags, host distance) and the
    machine-learning cuts this server applies afterwards (AppleCiDEr class, anomaly flag, BTSbot).

    Give a time window (at most 1 day), a cone (ra, dec, radius_arcsec), or an object_id.
    Returns how many alerts were scanned and matched, and the top `limit` rows.
    """
    _check_ml(survey, applecider_class, anomalous_only or None, min_btsbot)
    if sort_by in ("class_probability", "anomaly_score", "btsbot"):
        _check_ml(survey, True)
    if sort_by == "class_probability" and not applecider_class:
        raise ToolError("sort_by=class_probability needs applecider_class.")
    cone = (ra, dec, radius_arcsec)
    if any(v is not None for v in cone) and not all(v is not None for v in cone):
        raise ToolError("A cone search needs ra, dec and radius_arcsec together.")

    try:
        start_jd = s.to_jd(start) if start is not None else None
        end_jd = s.to_jd(end) if end is not None else None
    except (ValueError, OverflowError, OSError) as e:
        raise ToolError(str(e)) from e
    if start_jd is not None and end_jd is None:
        end_jd = min(start_jd + 1.0, s.now_jd())
    if end_jd is not None and start_jd is None:
        start_jd = end_jd - 1.0

    params = {
        "object_id": object_id,
        "ra": ra,
        "dec": dec,
        "radius_arcsec": radius_arcsec,
        "min_magpsf": min_mag,
        "max_magpsf": max_mag,
        "min_drb": min_reliability,
        "is_positive": True if positive_only else None,
        "is_star": False if exclude_stars else None,
        "is_rock": False if exclude_solar_system else None,
        "is_near_brightstar": False if exclude_near_bright_stars else None,
        "is_stationary": True if stationary_only else None,
        "max_host_dist_mpc": max_host_distance_mpc,
    }
    params = {k: v for k, v in params.items() if v is not None}
    bm = babamul(ctx)
    truncated = False
    if object_id is None and ra is None:
        if start_jd is None:
            raise ToolError("Give a time window (start/end), a cone (ra, dec, radius_arcsec), or an object_id.")
        hours = _hours(start_jd, end_jd)
        alerts, seen, i = [], set(), 0
        # Hour by hour in time order, a few at a time, until the budget runs out.
        while i < len(hours) and len(alerts) < max_scanned and not truncated:
            left = max_scanned - len(alerts)
            # Each hour of a batch may fetch all that is left of the budget, so
            # with less than a page left, one hour at a time wastes less.
            batch = hours[i : i + (CONCURRENCY if left >= PAGE_SIZE else 1)]
            i += len(batch)
            results = [None] * len(batch)

            async def run(j, window, budget):
                results[j] = await _fetch(bm, survey, params, *window, budget=budget)

            try:
                async with anyio.create_task_group() as tg:
                    for j, window in enumerate(batch):
                        tg.start_soon(run, j, window, left)
            except* ToolError as group:
                # Unwrapped, or the model gets "Error executing tool" instead of what to change.
                raise group.exceptions[0] from None
            for chunk, complete in results:
                # A chunk cut short by the budget leaves alerts unseen, even when
                # few of the ones fetched fall inside the window.
                truncated = truncated or not complete
                # BOOM's jd bounds are inclusive at both ends, so an alert on an hour
                # boundary comes back in both neighboring chunks.
                for a in chunk:
                    if start_jd <= (a.get("candidate") or {}).get("jd", 0) <= end_jd and a["candid"] not in seen:
                        seen.add(a["candid"])
                        alerts.append(a)
        # Hours never fetched once the budget ran out may hold more.
        truncated = truncated or i < len(hours) or len(alerts) > max_scanned
        alerts = alerts[:max_scanned]
    else:
        alerts, complete = await _fetch(bm, survey, params, start_jd, end_jd, budget=max_scanned)
        truncated = not complete
    scanned = len(alerts)

    def keep(a):
        c = a.get("classifications")
        if applecider_class and (s.applecider_prob(c, applecider_class) or 0) < min_class_probability:
            return False
        if anomalous_only and not s.is_ood(c):
            return False
        if min_btsbot is not None and ((c or {}).get("btsbot") or 0) < min_btsbot:
            return False
        return True

    def key(a):
        cand, c = a.get("candidate") or {}, a.get("classifications")
        rel = cand.get("drb") if survey == "ZTF" else cand.get("reliability")
        return {
            "time": -(cand.get("jd") or 0),
            "brightness": cand.get("magpsf") or 99,
            "reliability": -(rel or 0),
            "class_probability": -(s.applecider_prob(c, applecider_class) or 0) if applecider_class else 0,
            "anomaly_score": -(s.ood_score(c) or 0),
            "btsbot": -((c or {}).get("btsbot") or 0),
        }[sort_by]

    matched = sorted((a for a in alerts if keep(a)), key=key)
    n_matched = len(matched)
    if one_per_object:
        seen, best = set(), []
        for a in matched:
            if a.get("objectId") not in seen:
                seen.add(a.get("objectId"))
                best.append(a)
        matched = best

    result = {
        "query": {
            "survey": survey,
            "start": s.jd_to_iso(start_jd),
            "end": s.jd_to_iso(end_jd),
            **params,
        },
        "scanned": scanned,
        "matched": n_matched,
        "objects": len({a.get("objectId") for a in matched}) if one_per_object else None,
        "truncated": truncated,
        "returned": min(limit, len(matched)),
        "alerts": [s.alert_row(a, survey) for a in matched[:limit]],
    }
    if truncated:
        result["note"] = (
            f"Part of this search reached max_scanned ({max_scanned}) before BOOM ran out of "
            "alerts, so some were never scanned. matched and objects are lower bounds; add cuts "
            "or narrow the window for complete counts."
        )
    return {k: v for k, v in result.items() if v is not None}


async def _object(ctx: Context, survey: str, object_id: str) -> dict:
    return await babamul(ctx).get(f"/surveys/{survey}/objects/{checked_object_id(object_id)}")


async def _render(fn, *args) -> bytes:
    """fn(*args) in a worker thread, so a plot does not stall every other stream."""
    return await anyio.to_thread.run_sync(fn, *args, limiter=_render_limiter)


@tool()
async def get_object(ctx: Context, object_id: str, survey: Survey = "ZTF") -> dict:
    """Everything worth knowing about one object, summarized: position, latest alert, per-band
    light-curve shape (peak, rise and fade rates, last limit before first detection), flags,
    host galaxy, machine-learning outputs and how they changed over time, nearest catalog
    match per catalog, and any counterpart in the other survey.
    """
    return s.object_summary(await _object(ctx, survey, object_id), survey)


@tool(structured_output=False)
async def get_light_curve(
    ctx: Context,
    object_id: str,
    survey: Survey = "ZTF",
    include_forced_photometry: bool = False,
    include_upper_limits: bool = True,
    include_other_survey: Annotated[
        bool, Field(description="Add the matched ZTF/LSST counterpart's photometry.")
    ] = True,
    plot: Annotated[bool, Field(description="Also return a PNG of the light curve.")] = False,
    max_points: Annotated[int, Field(description="Most recent points to list.", ge=10, le=2000)] = 300,
):
    """An object's photometry as CSV (jd, band, mag, magerr, limit, source), with a per-band
    summary and, if plot is true, an image of the light curve. Bands are prefixed by survey
    (ztfg, lsstr, ...). A negative mag value marks a negative difference (fading below the
    reference).
    """
    obj = await _object(ctx, survey, object_id)
    rows = s.light_curve_rows(
        obj, survey, forced=include_forced_photometry, limits=include_upper_limits, other_survey=include_other_survey
    )
    if not rows:
        raise ToolError(f"{object_id} has no photometry in Babamul.")
    shown = rows[-max_points:]
    header = {
        "object_id": object_id,
        "n_points": len(rows),
        "listed": len(shown),
        "bands": s.band_summaries(rows),
    }
    content = [header, s.rows_as_csv(shown)]
    if plot:
        content.append(Image(data=await _render(plots.light_curve, rows, f"{survey} {object_id}"), format="png"))
    return content


@tool(structured_output=False)
async def get_cutouts(
    ctx: Context,
    object_id: Annotated[str | None, Field(description="Uses the object's latest alert.")] = None,
    candid: Annotated[str | None, Field(description="A specific alert, as a string.")] = None,
    survey: Survey = "ZTF",
):
    """The science, template and difference image cutouts of one alert, as a single PNG.
    A real transient is a point source in the science and difference images with nothing
    (or a galaxy) in the template; dipoles, streaks and edge effects are artifacts.
    """
    bm = babamul(ctx)
    if candid is None:
        if object_id is None:
            raise ToolError("Give object_id or candid.")
        candid = (await _object(ctx, survey, object_id)).get("candid")
        if candid is None:
            raise ToolError(f"{object_id} has no alert to show cutouts of.")
        candid = str(candid)
    data = await bm.get(f"/surveys/{survey}/cutouts", candid=candid)
    images = [base64.b64decode(data.get(k) or "") for k in ("cutoutScience", "cutoutTemplate", "cutoutDifference")]
    if not any(images):
        raise ToolError(f"No cutouts stored for candid {candid}.")
    title = f"{survey} {object_id or ''} candid {candid}".replace("  ", " ")
    png = await _render(lambda: plots.cutout_triplet(*images, title, flip=survey == "ZTF"))
    return [f"Cutouts of {title}", Image(data=png, format="png")]


@tool()
async def get_cross_matches(
    ctx: Context,
    object_id: str,
    survey: Survey = "ZTF",
    catalogs: Annotated[
        list[str] | None, Field(description="e.g. TNS, NED, Gaia_DR3, VSX, milliquas_v8. Default: all.")
    ] = None,
    per_catalog: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict:
    """An object's matches in the reference catalogs BOOM cross-matches against (TNS, NED,
    Gaia DR3, VSX, Milliquas, DESI, PS1, Legacy Survey, WISE, 2MASS, GALEX, X-ray catalogs),
    nearest first.
    """
    data = await babamul(ctx).get(f"/surveys/{survey}/objects/{checked_object_id(object_id)}/cross-matches")
    if catalogs:
        wanted = {c.lower() for c in catalogs}
        data = {k: v for k, v in data.items() if k.lower() in wanted}
    return {"object_id": object_id, "cross_matches": s.xmatch_summary(data, per_catalog)}


def _separation_arcsec(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h)))) * 3600


@tool()
async def cone_search(
    ctx: Context,
    ra: Annotated[float, Field(description="Degrees.")],
    dec: Annotated[float, Field(description="Degrees.")],
    radius_arcsec: Annotated[float, Field(gt=0, le=600)] = 5.0,
    survey: Survey = "ZTF",
) -> dict:
    """The objects BOOM knows within radius_arcsec of a position, nearest first."""
    data = await babamul(ctx).post(
        f"/surveys/{survey}/objects/cone-search",
        {"coordinates": {"target": [ra, dec]}, "radius_arcsec": radius_arcsec},
    )
    objects = [
        {
            "object_id": o.get("objectId"),
            "ra": round(o["ra"], 6),
            "dec": round(o["dec"], 6),
            "separation_arcsec": round(_separation_arcsec(ra, dec, o["ra"], o["dec"]), 2),
        }
        for o in (data or {}).get("target", [])
    ]
    return {"n": len(objects), "objects": sorted(objects, key=lambda o: o["separation_arcsec"])}


@tool()
async def find_similar_objects(
    ctx: Context,
    object_id: str,
    top_k: Annotated[int, Field(ge=1, le=50)] = 10,
    describe: Annotated[
        bool, Field(description="Also fetch each neighbor's AppleCiDEr label and TNS name/type (slower).")
    ] = True,
) -> dict:
    """ZTF objects nearest to this one in AppleCiDEr's embedding space (the fusion of light
    curve, cutouts and metadata), i.e. objects the model considers alike. Useful to find
    more examples of something, or to see what an unusual object resembles.
    """
    hits = await babamul(ctx).post("/similarity/objects", {"object_id": object_id.strip(), "top_k": top_k})
    neighbors = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()} for h in hits or []]
    if describe and neighbors:
        sem = anyio.Semaphore(4)

        async def label(n):
            async with sem:
                try:
                    obj = await _object(ctx, "ZTF", n.get("object_id") or n.get("objectId"))
                except BabamulError:
                    return
            summary = s.object_summary(obj, "ZTF")
            n["applecider"] = ((summary.get("ml") or {}).get("applecider") or {}).get("decision")
            if tns := (summary.get("cross_matches") or {}).get("TNS"):
                n["tns"] = tns.get("nearest")
            n["n_detections"] = summary.get("n_detections")

        async with anyio.create_task_group() as tg:
            for n in neighbors:
                tg.start_soon(label, n)
    return {"seed": object_id, "neighbors": neighbors}


@tool()
async def get_light_curve_fit(
    ctx: Context,
    object_id: str,
    which: Literal["last", "first", "brightest", "faintest"] = "last",
) -> dict:
    """The Villar (2019) supernova light-curve model BOOM fits on GPU for ZTF alerts: peak
    time and magnitude, rise and decline timescales per band, and the reduced chi-squared.
    A good fit (reduced chi2 near 1) supports a supernova-like light curve.
    """
    fit = await babamul(ctx).get("/surveys/ZTF/villar-fit", objectId=object_id.strip(), which=which)
    if fit.get("t_peak"):
        fit["t_peak_iso"] = s.jd_to_iso(fit["t_peak"])
    fit["candid"] = str(fit.get("candid"))
    return fit


@mcp.resource("boom://guide", name="guide", description="Conventions for reading BOOM data correctly.")
def guide() -> str:
    return INSTRUCTIONS


@mcp.prompt(name="triage", description="Decide whether an object is worth following up.")
def triage(object_id: str) -> str:
    return (
        f"Triage {object_id}. Use get_object, then get_light_curve with plot=true and "
        "get_cutouts. Decide: is it real or an artifact? What is it most likely (cite the "
        "light curve, host, cross-matches and the ML outputs, and say where they disagree)? "
        "Is it worth spectroscopic follow-up tonight, and why? Finish with a one-line verdict."
    )
