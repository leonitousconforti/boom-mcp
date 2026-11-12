# ZTF26rubinml: a fake object for the Rubin ML Training School hunt

Five ZTF alert packets for two fake public objects in M74, for the hidden-object
hunt in the MCP session (Tuesday 2026-11-10, 2:45 CT). Participants find them
through the Babamul API.

| objectId | What it is | Packets |
| --- | --- | --- |
| `ZTF26openeye` | The signpost: one 13.6 mag alert at 2026-11-09T07:12Z whose difference image reads `GO N 45"` | 1 |
| `ZTF26rubinml` | The treasure, 45″ north: 49 detections whose light curve draws an apple, and `HONEYCRISP` in its difference image | 4 |

Each packet is a copy of `tests/data/alerts/ztf/2695378462115010012.avro` (schema 4.02,
`programid` 1) with new ids, positions, photometry and cutouts. The template images are
the real ZTF r-band reference for field 1548, CCD 10, quadrant 4.

## Ingesting

`out/public/YYYYMMDD/` matches the layout `kafka_producer` reads
(`data/alerts/ztf/public/YYYYMMDD`), one folder per night (2026-11-04 to 2026-11-10).
Copy each folder there and produce it to `ztf_YYYYMMDD_programid1`, or send the files
to whichever topic the consumer is reading. Those dates may be outside the live
consumer's window.

Please keep the objects off the public Babamul Kafka topics and out of user filters
(Fritz auto-save and TNS reporting included). They are obviously fake, but they should
not notify anyone.

## Tested

Ingested on a local BOOM (`ghcr.io/boom-astro/boom:latest`, 2026-11-06) through
`kafka_producer` and the ZTF consumer: all five alerts stored and enriched, both objects
associated with M74 (from a NED row), and AppleCiDEr flagging every treasure alert as
out-of-distribution. Not tested locally: Villar fits (no GPU) and similarity (no Milvus).

## Removing

Everything has `objectId` `ZTF26openeye` or `ZTF26rubinml`, and every candid is at least
9e18 (real ones are near 3.6e18):

- `ZTF_alerts`: `{"objectId": {"$in": ["ZTF26openeye", "ZTF26rubinml"]}}`
- `ZTF_alerts_aux`: `{"_id": {"$in": ["ZTF26openeye", "ZTF26rubinml"]}}`
- `ZTF_alerts_cutouts`: `{"_id": {"$gte": 9000000000000000000}}`
- Milvus: the embeddings of both object ids

## Regenerating

```bash
python hunt/make_rubinml.py hunt/out   # needs fastavro, astropy, numpy; fetches the references from IRSA once
```

`out/truth.json` is the answer key; keep it away from participants.
`ANSWER-SHEET.md` is the grading sheet from the 2026-11-10 session. Questions 6 and 7 and the
real objects it names depend on that week's alerts, so re-check them against new data.
