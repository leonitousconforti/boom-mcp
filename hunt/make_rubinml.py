"""Build the ZTF alert packets for the school's hidden-object hunt.

Two fake public ZTF objects in M74:
  ZTF26openeye  the signpost: one bright alert whose difference image reads GO N 45"
  ZTF26rubinml  the treasure: a light curve that draws an apple, and HONEYCRISP in its
                difference image

Every packet is a copy of a real ZTF 4.02 alert from BOOM's test data with our values
swapped in, so it parses wherever a real one does. Candidate ids all start with 9
(real ones are near 3.6e18), so they cannot collide and are easy to delete.

    python hunt/make_rubinml.py hunt/out
"""

import copy
import gzip
import io
import json
import math
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import fastavro
import numpy as np
from astropy.io import fits

HERE = Path(__file__).parent
TEMPLATE = HERE / "template.avro"

TREASURE = {"object_id": "ZTF26rubinml", "ra": 24.1620, "dec": 15.8085}
SIGNPOST = {"object_id": "ZTF26openeye", "ra": 24.1620, "dec": 15.8085 - 45 / 3600}
# ZTF field 1548, CCD 10, quadrant 4 covers M74; its reference images give the templates.
FIELD, CCDID, QID = 1548, 10, 4
RCID = (CCDID - 1) * 4 + QID - 1
FID = {"g": 1, "r": 2, "i": 3}
RFID = {b: int(f"{FIELD}120{FID[b]}{RCID:02d}") for b in FID}

APPLE_START = 2461341.75  # 2026-10-28T06:00Z
SIGNPOST_JD = 2461353.80  # 2026-11-09T07:12Z, MJD 61353.30

FONT = {  # 5x7, top row first
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    '"': ["01010", "01010", "00000", "00000", "00000", "00000", "00000"],
    " ": ["00000"] * 7,
}


def jd_to_dt(jd):
    return datetime.fromtimestamp((jd - 2440587.5) * 86400, UTC)


def nid(jd):
    # ZTF night ids count nights from 2017-01-01 (checked against the template packet).
    return int(jd - 2457754.5)


# Light curve ----------------------------------------------------------------


def apple():
    """(x, y, band) points of an apple: a red outline, an orange stem, a green leaf."""
    pts = []
    cx, r = 0.33, 0.62
    for sign in (-1, 1):
        for k in range(26):
            phi = 2 * math.pi * k / 26
            x, y = sign * cx + r * math.cos(phi), r * math.sin(phi)
            if math.hypot(x + sign * cx, y) < r - 1e-6:
                continue  # inside the other lobe
            if y < 0:
                x *= 0.92
            pts.append((x, y, "r"))
    for k in range(5):
        pts.append((0.015 * k * k, 0.50 + 0.09 * k, "i"))
    for k in range(10):
        t = 2 * math.pi * k / 10
        u, v = 0.17 * math.cos(t), 0.065 * math.sin(t)
        a = math.radians(28)
        pts.append((0.21 + u * math.cos(a) - v * math.sin(a), 0.80 + u * math.sin(a) + v * math.cos(a), "g"))
    return pts


def photometry():
    """Detections and non-detections of the treasure, oldest first."""
    dets = []
    for k, (x, y, band) in enumerate(sorted(apple(), key=lambda p: p[0])):
        # 5 days and 1.6 mag per unit keep the apple round on a 16:9 plot.
        jd = APPLE_START + 8.0 + 5.0 * x + 0.0003 * k  # unique epochs
        dets.append({"jd": jd, "band": band, "mag": round(18.6 - 1.6 * y, 3), "magerr": 0.04})
    # Earlier visits of the field saw nothing; they also leave room on the plot.
    nondets = [
        {"jd": APPLE_START - 9.0, "band": "g", "limit": 20.0},
        {"jd": APPLE_START - 6.0, "band": "r", "limit": 20.1},
        {"jd": APPLE_START - 3.0, "band": "g", "limit": 19.9},
        {"jd": APPLE_START - 0.5, "band": "r", "limit": 20.0},
    ]
    return dets, nondets


# Images ---------------------------------------------------------------------


def reference(ra, dec, cache):
    """A 63x63 ZTF r-band reference cutout from IRSA, in the alert stamps' orientation."""
    path = cache / f"ref_{ra:.4f}_{dec:.4f}.fits"
    if not path.exists():
        url = (
            f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/{FIELD // 1000:03d}/field{FIELD:06d}/zr/"
            f"ccd{CCDID:02d}/q{QID}/ztf_{FIELD:06d}_zr_c{CCDID:02d}_q{QID}_refimg.fits"
            f"?center={ra},{dec}&size=63pix&gzip=false"
        )
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    data = fits.getdata(path).astype(np.float32)
    out = np.zeros((63, 63), np.float32)
    h, w = min(63, data.shape[0]), min(63, data.shape[1])
    out[:h, :w] = data[:h, :w]
    return out - np.median(out) + 150.0  # sky near a science frame's


def text_mask(lines, scale=2, gap=3):
    """Text as a 63x63 0/1 mask, row 0 on top: viewers flip ZTF stamps so north is up,
    which shows row 0 at the top."""
    mask = np.zeros((63, 63), np.float32)
    h = 7 * scale
    top = (63 - (len(lines) * h + (len(lines) - 1) * gap)) // 2
    for i, line in enumerate(lines):
        width = len(line) * 6 * scale - scale
        left = (63 - width) // 2
        for j, ch in enumerate(line):
            for row, bits in enumerate(FONT[ch]):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        y0 = top + i * (h + gap) + row * scale
                        x0 = left + j * 6 * scale + col * scale
                        mask[y0 : y0 + scale, x0 : x0 + scale] = 1
    return mask


def stamps(template, lines, rng, amplitude=260.0, noise=9.0):
    diff = rng.normal(0, noise, (63, 63)).astype(np.float32) + amplitude * text_mask(lines)
    sci = template + diff + rng.normal(0, noise / 2, (63, 63)).astype(np.float32)
    return sci, template, diff


def fits_gz(data, header):
    hdu = fits.PrimaryHDU(data.astype(">f4"), header=header)
    buf = io.BytesIO()
    hdu.writeto(buf)
    return gzip.compress(buf.getvalue())


# Packets --------------------------------------------------------------------


class Ids:
    def __init__(self):
        self.pid = 9_000_000_000_000

    def next(self):
        self.pid += 1
        return self.pid, self.pid * 1_000_000 + 10_000 + RCID


def detection_fields(base, d, ra, dec, pid, candid):
    p = copy.deepcopy(base)
    band = d["band"]
    p.update(
        jd=d["jd"], fid=FID[band], pid=pid, candid=candid, programid=1, isdiffpos="t", tblid=0,
        nid=nid(d["jd"]), rcid=RCID, field=FIELD, xpos=1532.4, ypos=1611.8, ra=ra, dec=dec,
        magpsf=d["mag"], sigmapsf=d["magerr"], chipsf=1.1, magap=d["mag"] + 0.02, sigmagap=0.05,
        diffmaglim=20.4, distnr=0.62, magnr=19.4, sigmagnr=0.05, chinr=1.2, sharpnr=0.05, sky=0.3,
        magdiff=0.02, fwhm=2.1, classtar=0.98, mindtoedge=480.0, magfromlim=20.4 - d["mag"],
        seeratio=1.02, aimage=0.92, bimage=0.88, aimagerat=0.44, bimagerat=0.42, elong=1.05,
        nneg=3, nbad=0, rb=0.93, ssdistnr=-999.0, ssmagnr=-999.0, ssnamenr="null", sumrat=0.98,
        magapbig=d["mag"] + 0.03, sigmagapbig=0.05, ranr=ra + 0.00012, decnr=dec - 0.00009, scorr=18.0,
        magzpsci=26.25, magzpsciunc=3e-6, magzpscirms=0.02, clrcoeff=0.08, clrcounc=6e-6, rbversion="t17_f5_c3",
        pdiffimfilename=f"ztf_{jd_to_dt(d['jd']):%Y%m%d}{int((d['jd'] % 1) * 1e6):06d}_{FIELD:06d}_z{band}_c{CCDID:02d}_o_q{QID}_scimrefdiffimg.fits",
    )
    return p


def nondetection_fields(base, d, pid):
    p = copy.deepcopy(base)
    p.update(jd=d["jd"], fid=FID[d["band"]], pid=pid, diffmaglim=d["limit"], nid=nid(d["jd"]), rcid=RCID, field=FIELD, programid=1)
    return p


def assign(points, obj, ids, jitter):
    """Give every epoch its ids, and every detection its measured position, once, so
    an epoch looks the same in every packet that carries it."""
    for p in points:
        p["pid"], p["candid"] = ids.next()
        if "mag" in p:
            p["ra"], p["dec"] = obj["ra"] + jitter(), obj["dec"] + jitter()


def build_alert(template, det, history, nondets, obj, stamp_set):
    """One alert for detection `det`, with `history` (earlier detections) and `nondets`
    as prv_candidates."""
    tpl_prv = template["prv_candidates"]
    det_base = next(p for p in tpl_prv if p.get("candid"))
    non_base = next(p for p in tpl_prv if not p.get("candid"))

    prv = [nondetection_fields(non_base, n, n["pid"]) for n in nondets]
    prv += [detection_fields(det_base, h, h["ra"], h["dec"], h["pid"], h["candid"]) for h in history]
    prv.sort(key=lambda p: p["jd"])

    pid, candid = det["pid"], det["candid"]
    cand = copy.deepcopy(template["candidate"])
    cand.update(detection_fields(det_base, det, det["ra"], det["dec"], pid, candid))
    first = min([det] + list(history), key=lambda d: d["jd"])
    cand.update(
        ndethist=len(history) + 1, ncovhist=len(history) + 1 + len(nondets) + 6,
        jdstarthist=first["jd"], jdendhist=det["jd"], rfid=RFID[det["band"]],
        sgmag1=19.9, srmag1=19.3, simag1=19.1, szmag1=19.0, sgscore1=0.08, distpsnr1=0.7, nmtchps=9,
        neargaia=18.5, maggaia=17.9, neargaiabright=48.0, maggaiabright=11.6,
        drb=0.996, tooflag=0, exptime=30.0,
    )

    record = copy.deepcopy(template)
    record.update(objectId=obj["object_id"], candid=candid, candidate=cand, prv_candidates=prv, fp_hists=[])
    header = fits.Header()
    header["BUNIT"] = "DN"
    sci, tmpl, diff = stamp_set
    record["cutoutScience"] = {"fileName": f"candid{candid}_pid{pid}_targ_sci.fits.gz", "stampData": fits_gz(sci, header)}
    record["cutoutTemplate"] = {"fileName": f"candid{candid}_ref.fits.gz", "stampData": fits_gz(tmpl, header)}
    record["cutoutDifference"] = {"fileName": f"candid{candid}_pid{pid}_targ_refmsci.fits.gz", "stampData": fits_gz(diff, header)}
    return record


def main(out: Path):
    with TEMPLATE.open("rb") as fh:
        reader = fastavro.reader(fh)
        schema = fastavro.parse_schema(reader.writer_schema)
        template = next(reader)
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261110)
    jitter = lambda: float(rng.normal(0, 0.1 / 3600))  # noqa: E731  (0.1" astrometric scatter)
    ids = Ids()
    written = []

    def write(record):
        night = jd_to_dt(record["candidate"]["jd"])
        folder = out / "public" / f"{night:%Y%m%d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{record['candid']}.avro"
        with path.open("wb") as fh:
            fastavro.writer(fh, schema, [record], codec="null")
        written.append(path)

    # The treasure: four alerts through the apple, each carrying everything before it.
    dets, nondets = photometry()
    assign(nondets + dets, TREASURE, ids, jitter)
    ref = reference(TREASURE["ra"], TREASURE["dec"], cache)
    alert_epochs = [round(len(dets) * f) - 1 for f in (0.3, 0.55, 0.8, 1.0)]
    treasure_candids = []
    for i in alert_epochs:
        write(build_alert(template, dets[i], dets[:i], nondets, TREASURE, stamps(ref, ["HONEY", "CRISP"], rng)))
        treasure_candids.append(dets[i]["candid"])

    # The signpost: one bright alert out of nowhere.
    sign = {"jd": SIGNPOST_JD, "band": "r", "mag": 13.62, "magerr": 0.02}
    sign_nondets = [{"jd": SIGNPOST_JD - d, "band": b, "limit": 20.3} for d, b in ((2.0, "r"), (2.02, "g"), (4.0, "r"))]
    assign(sign_nondets + [sign], SIGNPOST, ids, jitter)
    ref = reference(SIGNPOST["ra"], SIGNPOST["dec"], cache)
    write(build_alert(template, sign, [], sign_nondets, SIGNPOST, stamps(ref, ["GO N", '45"'], rng)))
    sign_candid = sign["candid"]

    first, last_lim = dets[0], nondets[-1]
    truth = {
        "start_card": f"At MJD {SIGNPOST_JD - 2400000.5:.2f}, something brighter than 14th magnitude appeared that no survey should have seen.",
        "signpost": {**SIGNPOST, "candid": str(sign_candid), "utc": jd_to_dt(SIGNPOST_JD).isoformat(), "mag_r": sign["mag"], "cutout_text": 'GO N 45"'},
        "treasure": {
            **TREASURE,
            "candids": [str(c) for c in treasure_candids],
            "separation_from_signpost_arcsec": 45.0,
            "host": "M74 (NGC 628), the Phantom Galaxy",
            "cutout_text": "HONEYCRISP",
            "light_curve": "an apple: red (r) outline, orange (i) stem, green (g) leaf",
            "first_detection_utc": jd_to_dt(first["jd"]).isoformat(),
            "last_nondetection_utc": jd_to_dt(last_lim["jd"]).isoformat(),
            "n_detections": len(dets),
        },
        "read_after_ingestion": ["applecider decision and out_of_distribution flag", "nearest neighbors outside the hunt (cone_search, get_object)", "btsbot score"],
        "cleanup": "delete objectId ZTF26openeye and ZTF26rubinml (alerts, aux, cutouts, embeddings); every candid is >= 9e18",
        "files": [str(p.relative_to(out)) for p in written],
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(json.dumps(truth, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out"))
