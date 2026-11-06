"""Turn BOOM documents into answers small enough for a model's context.

A ZTF object from Babamul is tens of kilobytes (every detection, every catalog
match, every classifier output). These functions keep what a person triaging
the object would look at, and are pure so they can be tested without network.
"""

import math
from datetime import UTC, datetime

JD_UNIX_EPOCH = 2440587.5
MJD_OFFSET = 2400000.5
# The flux zero points the babamul package uses (fluxes are in units of 1e-9 of these).
FLUX_ZP = {"ZTF": 23.9, "LSST": 8.9}
ZTF_FID_BAND = {1: "g", 2: "r", 3: "i"}
ACAI_NAMES = {"h": "hosted", "n": "nuclear", "v": "variable_star", "o": "orphan", "b": "bogus"}


# Time ------------------------------------------------------------------------


def now_jd() -> float:
    return datetime.now(UTC).timestamp() / 86400 + JD_UNIX_EPOCH


def to_jd(value) -> float:
    """JD from an ISO date/time (UTC), a JD, an MJD, or "now"."""
    if isinstance(value, bool):
        raise ValueError(f"not a time: {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "now":
            return now_jd()
        try:
            value = float(text)
        except ValueError:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.timestamp() / 86400 + JD_UNIX_EPOCH
    value = float(value)
    if value > 2_000_000:
        return value
    if 30_000 < value < 100_000:
        return value + MJD_OFFSET
    raise ValueError(f"{value} is neither a JD (~2.46e6) nor an MJD (~6e4)")


def jd_to_iso(jd: float | None) -> str | None:
    if jd is None:
        return None
    dt = datetime.fromtimestamp((jd - JD_UNIX_EPOCH) * 86400, UTC)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def describe_time(jd: float) -> dict:
    return {
        "iso": jd_to_iso(jd),
        "jd": round(jd, 5),
        "mjd": round(jd - MJD_OFFSET, 5),
        "days_ago": round(now_jd() - jd, 3),
    }


# Small helpers ---------------------------------------------------------------


def _r(value, digits=3):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    return value


def _is_negative(point: dict) -> bool:
    flag = point.get("isdiffpos")
    if flag is None:
        return point.get("isNegative") is True
    if isinstance(flag, str):
        return flag.lower() in ("f", "0", "false")
    return flag is False


def _band(point: dict, survey: str) -> str:
    band = point.get("band") or ZTF_FID_BAND.get(point.get("fid"), "?")
    return f"{survey.lower()}{band}"


def _flux_to_mag(flux, fluxerr, survey):
    if flux is None or fluxerr is None or flux == 0:
        return None, None
    zp = FLUX_ZP[survey]
    f = abs(flux) * 1e-9
    return zp - 2.5 * math.log10(f), 1.0857 * fluxerr * 1e-9 / f


def _flux_to_limit(fluxerr, survey):
    if not fluxerr or fluxerr <= 0:
        return None
    return FLUX_ZP[survey] - 2.5 * math.log10(3 * fluxerr * 1e-9)


# Photometry ------------------------------------------------------------------


def _rows(points, survey: str, source: str) -> list[dict]:
    rows = []
    for p in points or []:
        jd = p.get("jd")
        if jd is None:
            continue
        mag, magerr, limit = p.get("magpsf"), p.get("sigmapsf"), p.get("diffmaglim")
        flux, fluxerr = p.get("psfFlux"), p.get("psfFluxErr")
        snr = p.get("snr_psf") if p.get("snr_psf") is not None else p.get("snr")
        if snr is None and flux is not None and fluxerr:
            snr = flux / fluxerr
        negative = _is_negative(p)
        if source == "forced":
            # Forced photometry measures every epoch, mostly noise; BOOM sends it as
            # fluxes only, so a magnitude needs a significant positive flux.
            if snr is None or snr < 3:
                mag = magerr = None
            elif mag is None:
                mag, magerr = _flux_to_mag(flux, fluxerr, survey)
        elif mag is None and source == "alert" and flux is not None and snr is not None and abs(snr) >= 3:
            mag, magerr = _flux_to_mag(flux, fluxerr, survey)
            negative = negative or flux < 0
        if mag is None and limit is None:
            limit = _flux_to_limit(p.get("psfFluxErr"), survey)
        if mag is None and limit is None:
            continue
        row = {"jd": round(jd, 5), "band": _band(p, survey)}
        if mag is not None:
            row.update(mag=_r(mag), magerr=_r(magerr))
            if negative:
                row["negative"] = True
        else:
            row["limit"] = _r(limit)
        row["source"] = source
        rows.append(row)
    return rows


def light_curve_rows(obj: dict, survey: str, *, forced=False, limits=True, other_survey=True) -> list[dict]:
    """Every photometry point of an object, oldest first, de-duplicated."""
    rows = _rows(obj.get("prv_candidates"), survey, "alert")
    if limits:
        rows += _rows(obj.get("prv_nondetections"), survey, "nondetection")
    if forced:
        rows += _rows(obj.get("fp_hists"), survey, "forced")
    if other_survey:
        for other, match in (obj.get("survey_matches") or {}).items():
            if not match:
                continue
            other = other.upper()
            rows += _rows(match.get("prv_candidates"), other, "alert")
            if limits:
                rows += _rows(match.get("prv_nondetections"), other, "nondetection")
            if forced:
                rows += _rows(match.get("fp_hists"), other, "forced")
    seen, unique = set(), []
    for row in sorted(rows, key=lambda r: r["jd"]):
        key = (round(row["jd"], 4), row["band"], row["source"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def band_summaries(rows: list[dict]) -> dict:
    """Per band: first, peak and last detection, rise and fade rates, and the
    deepest limit before the first detection (which bounds when it started)."""
    out = {}
    for band in sorted({r["band"] for r in rows}):
        points = [r for r in rows if r["band"] == band]
        dets = [r for r in points if "mag" in r and not r.get("negative")]
        lims = [r for r in points if "limit" in r]
        summary = {"n_detections": len(dets), "n_limits": len(lims)}
        if dets:
            first, last = dets[0], dets[-1]
            peak = min(dets, key=lambda r: r["mag"])
            summary.update(
                first={"iso": jd_to_iso(first["jd"]), "mag": first["mag"]},
                peak={"iso": jd_to_iso(peak["jd"]), "mag": peak["mag"]},
                last={"iso": jd_to_iso(last["jd"]), "mag": last["mag"]},
            )
            if peak["jd"] - first["jd"] > 0.5:
                summary["rise_mag_per_day"] = _r((first["mag"] - peak["mag"]) / (peak["jd"] - first["jd"]))
            if last["jd"] - peak["jd"] > 0.5:
                summary["fade_mag_per_day"] = _r((last["mag"] - peak["mag"]) / (last["jd"] - peak["jd"]))
            before = [r for r in lims if r["jd"] < first["jd"]]
            if before:
                lim = before[-1]
                summary["last_limit_before_first_detection"] = {
                    "iso": jd_to_iso(lim["jd"]),
                    "limit": lim["limit"],
                    "days_before": _r(first["jd"] - lim["jd"], 2),
                }
        out[band] = summary
    return out


def rows_as_csv(rows: list[dict]) -> str:
    lines = ["jd,band,mag,magerr,limit,source"]
    for r in rows:
        mag = "" if "mag" not in r else f"{'-' if r.get('negative') else ''}{r['mag']}"
        lines.append(
            f"{r['jd']},{r['band']},{mag},{r.get('magerr') or ''},{r.get('limit') or ''},{r['source']}"
        )
    return "\n".join(lines)


# Machine learning ------------------------------------------------------------


def ml_summary(classifications: dict | None) -> dict | None:
    """The classifier outputs BOOM attaches to a ZTF alert, in plain words."""
    if not classifications:
        return None
    c = classifications
    out = {}
    acai = {ACAI_NAMES[k[5:]]: _r(v) for k, v in c.items() if k.startswith("acai_") and k[5:] in ACAI_NAMES}
    if acai:
        out["acai"] = acai
    if c.get("btsbot") is not None:
        out["btsbot"] = _r(c["btsbot"])
    fusion, outputs = c.get("applecider_fusion"), c.get("applecider_outputs")
    if fusion:
        probs = sorted((fusion.get("class_probabilities") or {}).items(), key=lambda kv: -kv[1])
        out["applecider"] = {
            "decision": fusion.get("decision_label"),
            "decision_level": fusion.get("decision_level"),
            "decision_probability": _r(fusion.get("decision_probability")),
            "abstained": fusion.get("abstain_completely"),
            "top_classes": {k: _r(v) for k, v in probs[:3]},
        }
    if outputs:
        a = out.setdefault("applecider", {})
        for level in ("domain", "family"):
            if lv := outputs.get(level):
                a[level] = {"pred": lv.get("pred"), "prob": _r(lv.get("prob")), "kept": lv.get("kept")}
        if ood := outputs.get("ood"):
            a["out_of_distribution"] = {"flag": ood.get("flag"), "score": _r(ood.get("score")), "votes": ood.get("votes")}
        if unc := outputs.get("uncertainty"):
            a["uncertainty"] = {k: _r(unc.get(k)) for k in ("vacuity", "epistemic", "aleatoric")}
        a["n_detections_used"] = outputs.get("n_detections_used")
    return out


def applecider_prob(classifications: dict | None, label: str) -> float | None:
    fusion = (classifications or {}).get("applecider_fusion") or {}
    return (fusion.get("class_probabilities") or {}).get(label)


def ood_score(classifications: dict | None) -> float | None:
    ood = ((classifications or {}).get("applecider_outputs") or {}).get("ood") or {}
    return ood.get("score")


def is_ood(classifications: dict | None) -> bool:
    ood = ((classifications or {}).get("applecider_outputs") or {}).get("ood") or {}
    return ood.get("flag") is True


# Alerts and objects ----------------------------------------------------------


def _flags(props: dict | None) -> dict:
    props = props or {}
    flags = {k: props.get(k) for k in ("star", "rock", "near_brightstar", "stationary", "hosted") if k in props}
    if (sso := props.get("sso")) and sso.get("is_sso"):
        flags["solar_system_object"] = sso.get("designation") or True
    return flags


def host_summary(props: dict | None) -> dict | None:
    host = (props or {}).get("best_host")
    if not host:
        return None
    keys = ("name", "catalog", "z", "dist_mpc", "sep_arcsec", "sep_kpc", "d_dlr")
    return {k: _r(host.get(k)) for k in keys if host.get(k) is not None}


def alert_row(alert: dict, survey: str) -> dict:
    """One alert as a single row: when, where, how bright, how real, what the models say."""
    cand = alert.get("candidate") or {}
    props = alert.get("properties") or {}
    reliability = cand.get("drb") if survey == "ZTF" else cand.get("reliability")
    row = {
        "object_id": alert.get("objectId") or alert.get("object_id"),
        "candid": str(alert.get("candid") or alert.get("_id")),
        "time": jd_to_iso(cand.get("jd")),
        "jd": _r(cand.get("jd"), 5),
        "ra": _r(cand.get("ra"), 6),
        "dec": _r(cand.get("dec"), 6),
        "band": _band(cand, survey),
        "mag": _r(cand.get("magpsf")),
        "magerr": _r(cand.get("sigmapsf")),
        "reliability": _r(reliability),
        "negative": _is_negative(cand) or None,
        "n_detections": cand.get("ndethist"),
        "first_detected": jd_to_iso(cand.get("jdstarthist")),
        "flags": _flags(props) or None,
        "host": host_summary(props),
    }
    if ml := ml_summary(alert.get("classifications")):
        a = ml.get("applecider") or {}
        decision = a.get("decision") or ("abstained" if a.get("abstained") else None)
        compact = {
            "applecider": decision,
            "applecider_p": a.get("decision_probability") if a.get("decision") else None,
            "anomalous": (a.get("out_of_distribution") or {}).get("flag"),
            "btsbot": ml.get("btsbot"),
            "acai_bogus": (ml.get("acai") or {}).get("bogus"),
        }
        row["ml"] = {k: v for k, v in compact.items() if v is not None}
    return {k: v for k, v in row.items() if v is not None}


# Catalog fields worth showing, by catalog; anything else shows its first few scalars.
XMATCH_FIELDS = {
    "TNS": ("name", "type", "redshift", "discoverydate", "distance_arcsec"),
    "NED": ("objname", "objtype", "z", "DistMpc", "distance_arcsec", "distance_kpc"),
    "Gaia_DR3": ("parallax", "parallax_error", "pm", "phot_g_mean_mag", "ruwe", "distance_arcsec"),
    "VSX": ("name", "types", "period", "max", "min", "distance_arcsec"),
    "milliquas_v8": ("objtype", "z", "distance_arcsec"),
    "DESI_DR1": ("z", "spectype", "distance_arcsec"),
    "LSPSC": ("score", "mag_white", "distance_arcsec"),
    "AllWISE": ("w1mpro", "w2mpro", "distance_arcsec"),
}


SENTINELS = (-99, -99.0, -999, -999.0, "", " ")


def _trim_match(catalog: str, match: dict) -> dict:
    keys = XMATCH_FIELDS.get(catalog)
    if keys:
        out = {k: _r(match[k]) for k in keys if match.get(k) is not None}
    else:
        scalars = {
            k: _r(v)
            for k, v in match.items()
            if isinstance(v, (str, int, float)) and not k.startswith("_") and v not in SENTINELS
        }
        out = dict(list(scalars.items())[:8])
        if match.get("distance_arcsec") is not None:
            out["distance_arcsec"] = _r(match["distance_arcsec"], 2)
    return {k: v for k, v in out.items() if v not in SENTINELS}


def xmatch_summary(cross_matches: dict | None, per_catalog: int = 1) -> dict:
    out = {}
    for catalog, matches in (cross_matches or {}).items():
        if not matches:
            continue
        if isinstance(matches, dict):
            matches = [matches]
        ordered = sorted(matches, key=lambda m: m.get("distance_arcsec", float("inf")))
        out[catalog] = {
            "n": len(matches),
            "nearest" if per_catalog == 1 else "matches": (
                _trim_match(catalog, ordered[0])
                if per_catalog == 1
                else [_trim_match(catalog, m) for m in ordered[:per_catalog]]
            ),
        }
    return out


def object_summary(obj: dict, survey: str) -> dict:
    cand = obj.get("candidate") or {}
    props = obj.get("properties") or {}
    rows = light_curve_rows(obj, survey, forced=False, limits=True, other_survey=False)
    dets = [r for r in rows if "mag" in r and not r.get("negative")]
    out = {
        "object_id": obj.get("objectId") or obj.get("object_id"),
        "survey": survey,
        "ra": _r(cand.get("ra"), 6),
        "dec": _r(cand.get("dec"), 6),
        "latest_alert": alert_row(obj, survey),
        "n_detections": len(dets),
        "first_detection": jd_to_iso(dets[0]["jd"]) if dets else None,
        "last_detection": jd_to_iso(dets[-1]["jd"]) if dets else None,
        "bands": band_summaries(rows),
        "flags": _flags(props),
        "host": host_summary(props),
        "ml": ml_summary(obj.get("classifications")),
        "cross_matches": xmatch_summary(obj.get("cross_matches")),
    }
    out["latest_alert"].pop("ml", None)
    out["latest_alert"].pop("host", None)
    out["latest_alert"].pop("flags", None)
    history = [
        ((h.get("applecider_fusion") or {}).get("decision_label") or "abstained")
        for h in obj.get("classifications_history") or []
    ]
    if history:
        # Run-length encoded, so a label that held for 20 alerts is one entry.
        runs = []
        for label in history:
            if runs and runs[-1][0] == label:
                runs[-1][1] += 1
            else:
                runs.append([label, 1])
        out["applecider_history"] = [f"{label} x{n}" for label, n in runs]
    for other, match in (obj.get("survey_matches") or {}).items():
        if match:
            out.setdefault("counterparts", {})[other.upper()] = {
                "object_id": match.get("objectId") or match.get("object_id"),
                "distance_arcsec": _r(match.get("distance_arcsec"), 2),
                "n_detections": len(match.get("prv_candidates") or []),
            }
    if obj.get("tns_photometry"):
        out["has_tns_photometry"] = True
    return {k: v for k, v in out.items() if v not in (None, {}, [])}
