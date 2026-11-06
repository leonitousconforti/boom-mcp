"""Scan a few nights of quality-cut ZTF alerts and list candidate hunt targets by category."""

import asyncio
import json
import os
import sys

import httpx

from boom_mcp import summarize as s
from boom_mcp import tools
from boom_mcp.babamul import Babamul, base_url_from_env

CUTS = {"min_drb": 0.8, "is_positive": True, "is_star": False, "is_rock": False, "is_stationary": True}
NIGHTS = ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]


def probs(c):
    return ((c or {}).get("applecider_fusion") or {}).get("class_probabilities") or {}


async def main(out):
    async with httpx.AsyncClient() as http:
        bm = Babamul(http, os.environ["BABAMUL_API_TOKEN"], base_url_from_env())
        alerts = []
        for night in NIGHTS:
            hours = tools._hours(s.to_jd(f"{night}T02:00:00Z"), s.to_jd(f"{night}T14:00:00Z"))
            for i in range(0, len(hours), 3):
                chunks = await asyncio.gather(*(tools._fetch(bm, "ZTF", CUTS, a, b, 20000) for a, b in hours[i : i + 3]))
                for chunk in chunks:
                    alerts += chunk
            print(night, "cumulative alerts", len(alerts), file=sys.stderr)

        latest = {}
        for a in alerts:
            oid = a["objectId"]
            if oid not in latest or a["candidate"]["jd"] > latest[oid]["candidate"]["jd"]:
                latest[oid] = a
        print("objects", len(latest), file=sys.stderr)

        def row(a):
            c, p, cand = a.get("classifications") or {}, a.get("properties") or {}, a["candidate"]
            host = p.get("best_host") or {}
            ood = ((c.get("applecider_outputs") or {}).get("ood") or {})
            return {
                "id": a["objectId"], "mag": round(cand["magpsf"], 2), "band": cand.get("band"),
                "ndet": cand.get("ndethist"), "age_d": round(cand["jd"] - cand.get("jdstarthist", cand["jd"]), 1),
                "ra": round(cand["ra"], 4), "dec": round(cand["dec"], 4),
                "label": (c.get("applecider_fusion") or {}).get("decision_label"),
                "ood": ood.get("flag"), "ood_score": ood.get("score"),
                "btsbot": c.get("btsbot"), "acai_n": c.get("acai_n"), "acai_o": c.get("acai_o"), "acai_v": c.get("acai_v"),
                "hosted": p.get("hosted"), "host": host.get("name"), "host_z": host.get("z"), "host_mpc": host.get("dist_mpc"),
                "host_sep": host.get("sep_arcsec"), "p": probs(c),
            }

        rows = [row(a) for a in latest.values()]
        cats = {
            "tde": sorted([r for r in rows if r["p"].get("TDE", 0) >= 0.5], key=lambda r: -r["p"]["TDE"]),
            "slsn": sorted([r for r in rows if r["p"].get("Superluminous SN", 0) >= 0.4], key=lambda r: -r["p"]["Superluminous SN"]),
            "anomalous": sorted([r for r in rows if r["ood"] and (r["ndet"] or 0) >= 5], key=lambda r: -(r["ood_score"] or 0)),
            "young": sorted([r for r in rows if r["age_d"] <= 2.5 and 2 <= (r["ndet"] or 0) <= 6 and r["mag"] < 19.5 and r["hosted"]], key=lambda r: r["mag"]),
            "nearby": sorted([r for r in rows if r["host_mpc"] and r["host_mpc"] < 60 and (r["acai_v"] or 0) < 0.5], key=lambda r: r["host_mpc"]),
            "nuclear": sorted([r for r in rows if (r["acai_n"] or 0) >= 0.8 and (r["host_sep"] or 9) < 1.0], key=lambda r: -(r["acai_n"] or 0)),
            "bright_sn": sorted([r for r in rows if (r["btsbot"] or 0) >= 0.9 and r["mag"] < 18.5], key=lambda r: r["mag"]),
            "hostless": sorted([r for r in rows if r["hosted"] is False and (r["acai_o"] or 0) >= 0.5 and (r["ndet"] or 0) >= 3], key=lambda r: -(r["acai_o"] or 0)),
        }
        for k, v in cats.items():
            print(f"{k}: {len(v)}", file=sys.stderr)

        detail = {}
        for k, v in cats.items():
            for r in v[:4]:
                if r["id"] in detail:
                    continue
                try:
                    obj = await bm.get(f"/surveys/ZTF/objects/{r['id']}")
                except Exception as e:
                    print("skip", r["id"], e, file=sys.stderr)
                    continue
                sm = s.object_summary(obj, "ZTF")
                detail[r["id"]] = {
                    "bands": {b: {kk: x.get(kk) for kk in ("n_detections", "peak", "rise_mag_per_day", "fade_mag_per_day", "last_limit_before_first_detection")} for b, x in sm.get("bands", {}).items()},
                    "first_detection": sm.get("first_detection"), "last_detection": sm.get("last_detection"),
                    "tns": (sm.get("cross_matches") or {}).get("TNS"), "ned": (sm.get("cross_matches") or {}).get("NED"),
                    "milliquas": (sm.get("cross_matches") or {}).get("milliquas_v8"), "vsx": (sm.get("cross_matches") or {}).get("VSX"),
                    "gaia": (sm.get("cross_matches") or {}).get("Gaia_DR3"),
                    "counterparts": sm.get("counterparts"), "history": sm.get("applecider_history"),
                    "ml": sm.get("ml"), "host": sm.get("host"),
                }
        lsst_both = [oid for oid, d in detail.items() if d.get("counterparts")]
        json.dump({"cats": {k: v[:8] for k, v in cats.items()}, "detail": detail, "n_alerts": len(alerts), "n_objects": len(latest), "lsst_both": lsst_both}, open(out, "w"), indent=1, default=str)


asyncio.run(main(sys.argv[1]))
