"""PNG renderings a vision model can read: cutout triplets and light curves."""

import gzip
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.io.fits.verify import VerifyWarning  # noqa: E402
from astropy.visualization import AsymmetricPercentileInterval, ImageNormalize, LinearStretch  # noqa: E402

BAND_COLORS = {"g": "#2ca02c", "r": "#d62728", "i": "#ff7f0e", "z": "#8c564b", "y": "#7f7f7f", "u": "#1f77b4"}
SURVEY_MARKERS = {"ztf": "o", "lsst": "s", "decam": "D"}


def _image(data: bytes) -> np.ndarray:
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    # ZTF cutout headers trip astropy's strict SIMPLE-card check; the data is fine.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        with fits.open(io.BytesIO(data)) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            return np.nan_to_num(np.asarray(hdu.data, dtype=float))


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def cutout_triplet(science: bytes, template: bytes, difference: bytes, title: str, flip: bool = False) -> bytes:
    """`flip` turns ZTF stamps north-up, as the Babamul client and website show them."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, name, raw in zip(axes, ("science", "template", "difference"), (science, template, difference)):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        if not raw:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            continue
        img = _image(raw)
        if flip:
            img = np.flipud(img)
        if name == "difference":
            img = img - np.median(img)
            lim = np.percentile(np.abs(img), 99.7) or 1.0
            ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            norm = ImageNormalize(img, interval=AsymmetricPercentileInterval(1, 99.7), stretch=LinearStretch())
            ax.imshow(img, origin="lower", cmap="gray", norm=norm)
        # Tick marks beside the alert position rather than on it, so the source stays visible.
        cy, cx = (np.array(img.shape) - 1) / 2
        gap, arm = 4, 6
        ax.plot([cx - gap - arm, cx - gap], [cy, cy], color="yellow", lw=1.2)
        ax.plot([cx, cx], [cy + gap, cy + gap + arm], color="yellow", lw=1.2)
    fig.suptitle(title)
    return _png(fig)


def light_curve(rows: list[dict], title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t0 = min(r["jd"] for r in rows)
    for band in sorted({r["band"] for r in rows}):
        survey = next((s for s in SURVEY_MARKERS if band.startswith(s)), "ztf")
        color = BAND_COLORS.get(band[len(survey) :], "black")
        marker = SURVEY_MARKERS[survey]
        pts = [r for r in rows if r["band"] == band]
        det = [r for r in pts if "mag" in r and r["source"] != "forced"]
        forced = [r for r in pts if "mag" in r and r["source"] == "forced"]
        lim = [r for r in pts if "limit" in r]
        if det:
            ax.errorbar(
                [r["jd"] - t0 for r in det], [r["mag"] for r in det], [r.get("magerr") or 0 for r in det],
                fmt=marker, color=color, label=band, ms=5,
            )
        if forced:
            ax.errorbar(
                [r["jd"] - t0 for r in forced], [r["mag"] for r in forced], [r.get("magerr") or 0 for r in forced],
                fmt=marker, mfc="none", color=color, alpha=0.6, ms=5, label=f"{band} forced",
            )
        if lim:
            ax.scatter([r["jd"] - t0 for r in lim], [r["limit"] for r in lim], marker="v", color=color, alpha=0.35, s=18)
    ax.invert_yaxis()
    ax.set_xlabel(f"days since JD {t0:.2f}")
    ax.set_ylabel("magnitude (triangles: 3-sigma upper limits)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.2)
    return _png(fig)
