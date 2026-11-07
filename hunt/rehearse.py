"""Play the hunt through the MCP server, one tool per step, and print the answer key.

Runs against whatever Babamul the environment points at (BABAMUL_API_URL or
BABAMUL_ENV, and BABAMUL_API_TOKEN), so the same script checks a local rehearsal
and, once the BOOM team has ingested the packets, production.

    python hunt/rehearse.py OUT_DIR
"""

import base64
import json
import sys
from pathlib import Path

import anyio
from mcp.client import Client

from boom_mcp import plugins, tools  # noqa: F401
from boom_mcp import summarize as s
from boom_mcp.app import mcp

plugins.load_builtin()
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
TRUTH = json.loads((Path(__file__).parent / "out" / "truth.json").read_text())


async def main():
    answers = {}
    async with Client(mcp) as client:

        async def call(step, name, **args):
            r = await client.call_tool(name, args)
            texts = [c.text for c in r.content if getattr(c, "type", "") == "text"]
            for i, c in enumerate(c for c in r.content if getattr(c, "type", "") == "image"):
                (OUT / f"rehearse_{name}_{args.get('object_id', '')}_{i}.png").write_bytes(base64.b64decode(c.data))
            status = "ERROR" if r.is_error else "ok"
            print(f"\n[{step}] {name}({json.dumps(args)}) -> {status}")
            print("   ", (texts[0] if texts else "")[:400].replace("\n", " "))
            return r, (json.loads(texts[0]) if texts and not r.is_error and texts[0].startswith("{") else None)

        _, t = await call("trail 1", "convert_time", value="61353.30")
        jd = t["jd"]
        _, found = await call("trail 2", "search_alerts", start=str(jd - 1 / 1440), end=str(jd + 1 / 1440), max_mag=14.0, positive_only=False)
        signpost = found["alerts"][0]
        await call("trail 3", "get_object", object_id=signpost["object_id"])
        await call("trail 4", "get_cutouts", object_id=signpost["object_id"])
        _, near = await call("trail 5", "cone_search", ra=signpost["ra"], dec=signpost["dec"], radius_arcsec=60)
        treasure = next(o["object_id"] for o in near["objects"] if o["object_id"].startswith("ZTF26rubin"))

        await call("question 1", "get_cutouts", object_id=treasure)
        await call("question 2", "get_light_curve", object_id=treasure, plot=True, max_points=10)
        _, xm = await call("question 3", "get_cross_matches", object_id=treasure)
        _, ex = await call("question 4", "estimate_explosion_time", object_id=treasure)
        _, obj = await call("question 5", "get_object", object_id=treasure)
        r6, sim = await call("question 6", "find_similar_objects", object_id=treasure, top_k=3)
        r7, fit = await call("question 7", "get_light_curve_fit", object_id=treasure)

        ml = (obj or {}).get("ml") or {}
        answers = {
            "1 hidden word": TRUTH["treasure"]["cutout_text"] + " (check the image above)",
            "2 light curve": TRUTH["treasure"]["light_curve"] + " (check the plot above)",
            "3 host": {"host": (obj or {}).get("host"), "NED": ((xm or {}).get("cross_matches") or {}).get("NED")},
            "4 began": {k: (ex or {}).get(k) for k in ("last_nondetection", "first_detection", "window_days")},
            "5 anomaly": (ml.get("applecider") or {}).get("out_of_distribution"),
            "6 look-alike": (sim or {}).get("neighbors") if not r6.is_error else r6.content[0].text[:120],
            "7 supernova fit": fit if not r7.is_error else r7.content[0].text[:120],
            "debrief": {"btsbot": ml.get("btsbot"), "applecider": (ml.get("applecider") or {}).get("decision")},
        }
    print("\n=== Answer key from BOOM ===")
    print(json.dumps(answers, indent=1, default=str))
    print("\nsignpost found at", s.jd_to_iso(jd), "->", signpost["object_id"], "| treasure:", treasure)


anyio.run(main)
