"""Call every tool against the real Babamul API. Needs BABAMUL_API_TOKEN."""

import json
import sys
import time
from pathlib import Path

import anyio
from mcp.client import Client

from boom_mcp import plugins, tools  # noqa: F401
from boom_mcp.app import mcp

plugins.load_builtin()
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
NIGHT = {"start": "2026-11-06T02:00:00Z", "end": "2026-11-06T14:00:00Z"}
CUTS = {"min_reliability": 0.8, "stationary_only": True, "exclude_stars": True, "exclude_solar_system": True}


async def main():
    async with Client(mcp) as client:

        async def call(name, **args):
            t = time.time()
            r = await client.call_tool(name, args)
            dt = time.time() - t
            texts = [c.text for c in r.content if getattr(c, "type", "") == "text"]
            images = [c for c in r.content if getattr(c, "type", "") == "image"]
            print(f"\n=== {name} {json.dumps(args)} -> {'ERROR ' if r.is_error else ''}{dt:.1f}s, {sum(map(len, texts))} chars, {len(images)} images")
            for i, img in enumerate(images):
                import base64
                (OUT / f"{name}_{i}.png").write_bytes(base64.b64decode(img.data) if isinstance(img.data, str) else img.data)
            return r, texts

        await call("convert_time")
        _, t = await call("search_alerts", **NIGHT, **CUTS, sort_by="anomaly_score", limit=5)
        print(t[0][:2500])
        _, t = await call("search_alerts", **NIGHT, **CUTS, applecider_class="Ia-like SN", min_class_probability=0.5, sort_by="class_probability", limit=5)
        res = json.loads(t[0])
        print(json.dumps({k: res[k] for k in ("scanned", "matched", "objects", "truncated")}), [a["object_id"] for a in res["alerts"]])
        oid = res["alerts"][0]["object_id"] if res["alerts"] else "ZTF22abdaqpz"
        _, t = await call("get_object", object_id=oid)
        obj = json.loads(t[0])
        print(t[0][:3500])
        _, t = await call("get_light_curve", object_id=oid, plot=True, max_points=15)
        print(t[1][:800])
        await call("get_cutouts", object_id=oid)
        _, t = await call("get_cross_matches", object_id=oid, per_catalog=2)
        print(t[0][:1500])
        await call("cone_search", ra=obj["ra"], dec=obj["dec"], radius_arcsec=30)
        _, t = await call("find_similar_objects", object_id=oid, top_k=5)
        print(t[0][:1500])
        _, t = await call("get_light_curve_fit", object_id=oid)
        print(t[0][:800])
        _, t = await call("estimate_explosion_time", object_id=oid)
        print(t[0][:800])
        # Second identical search should come from the cache.
        await call("search_alerts", **NIGHT, **CUTS, sort_by="anomaly_score", limit=5)


anyio.run(main)
