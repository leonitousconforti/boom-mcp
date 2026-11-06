"""An example of putting your own model behind the BOOM tools.

A plugin is any Python file whose functions are decorated with @tool(). It can
fetch BOOM data with babamul(ctx), run anything on it (a scikit-learn model, an
ONNX export, a LUNA pipeline), and return a dict. Load your own with
`boom-mcp --plugin my_model.py`.

This one is deliberately simple: it bounds when a transient started from its
last non-detection and first detection, which is what "is this young?" means
when choosing what to follow up tonight.
"""

from mcp.server.mcpserver import Context

from boom_mcp import summarize as s
from boom_mcp.app import babamul, tool


@tool()
async def estimate_explosion_time(ctx: Context, object_id: str, survey: str = "ZTF") -> dict:
    """Bound when a transient first appeared, from the deepest non-detection before its first
    detection (in any band). Returns the window, its midpoint, and how many days ago that was.
    A window under ~2 days with a first detection in the last few days marks a young transient.
    """
    obj = await babamul(ctx).get(f"/surveys/{survey}/objects/{object_id}")
    rows = s.light_curve_rows(obj, survey, forced=True, limits=True, other_survey=True)
    detections = [r for r in rows if "mag" in r and not r.get("negative")]
    if not detections:
        return {"object_id": object_id, "verdict": "no detections"}
    first = detections[0]
    limits = [r for r in rows if "limit" in r and r["jd"] < first["jd"]]
    # Only a limit deeper than the first detection says the source was not there yet.
    useful = [r for r in limits if r["limit"] > first["mag"]]
    out = {"object_id": object_id, "first_detection": {**s.describe_time(first["jd"]), "mag": first["mag"], "band": first["band"]}}
    if not useful:
        out["verdict"] = "unconstrained: no limit before the first detection is deeper than it"
        return out
    last = useful[-1]
    mid = (last["jd"] + first["jd"]) / 2
    out.update(
        last_nondetection={**s.describe_time(last["jd"]), "limit": last["limit"], "band": last["band"]},
        window_days=round(first["jd"] - last["jd"], 3),
        estimate=s.describe_time(mid),
        verdict="young" if first["jd"] - last["jd"] < 2 and s.now_jd() - first["jd"] < 5 else "not obviously young",
    )
    return out
