# Hunt answer sheet (Rubin ML Training School, 2026-11-10)

Current as of Tue 2026-11-10 11:50 CT (facts re-checked against live tools, 98 of 98): after the morning topup (real data through the night of
Nov 9 to 10) and a rehearsal with three simulated teams through the public Ask panel plus the
scripted run. Evidence in logs/rehearse/: 20261110113109/ (scripted run, all steps ok),
team-A-20261110-1104/, team-B-20261110-1104/, team-C-20261110-1104/, q67-20261110104936/.
Earlier versions: ANSWER-SHEET.md.pre-topup-20261110 and ANSWER-SHEET.md.pre-rehearsal-20261110.
The logs and earlier versions named here were on the session's own BOOM, deleted 2026-11-12.

Start card (as on the hunt slide since 2026-11-10): "At MJD 61353.30, something brighter than
14th magnitude appeared that no survey should have seen. It appeared within a minute of that time."
The second sentence is new: with last night's real data in, a whole-day search buries the
signpost and the assistant then names a real object (it did, 2 out of 2, without the sentence).

## Trail

1. convert_time 61353.30 -> 2026-11-09 07:12:00Z (JD 2461353.8).
2. search_alerts around that minute, max_mag 14 -> **ZTF26openeye** (signpost): r 13.62 mag,
   1 detection at exactly 07:12:00Z, RA 24.16197 Dec 15.79599, about 61" from M74's center.
   Real alerts now share the window: asteroid (399), ZTF26acggbgw (r 13.25 at 07:12:24Z), is in
   the same minute, and 07:00 to 07:30 has about 26 alerts under 14 mag. The signpost is the one
   at exactly 07:12:00Z with a single detection that is not an asteroid. All three rehearsal
   teams got it in their first turn.
3. get_cutouts ZTF26openeye -> **GO N 45"**, written in both the science and difference panels.
   The assistant has no vision ("the cutouts are in the chat"), so the team has to look at the
   picture itself. Hint 1 is for this step.
4. 45 arcsec north is Dec 15.80849 (+0.0125 deg): cone_search there, radius 5 -> **ZTF26rubinml**,
   45.06" from the signpost (accept 45.0 to 45.2"). The assistant often aims only 22 to 28" north
   first and its small cone comes back empty. A 60" cone around the signpost lists 3:
   ZTF26openeye 0", ZTF26acesbfi 39.6" (asteroid 44423, SOUTH: wrong one), ZTF26rubinml 45.2".

## Questions (treasure ZTF26rubinml) and what to accept

When the questions go up (first treasure, or 3:25), say out loud: "Question 6 means nearest on
the sky: search a wider circle around the treasure, then look up each neighbor."

Checking "from a tool call": ask to see the team's chat only when a value is not on this sheet.
A value that matches this sheet counts if any tool call in the team's chats returned it, even
when the turn they copied it from has no tool chip (team B's BTSbot 0.985 came from a turn
without a tool call, but get_object had returned 0.985 earlier). Half credit is for the cases
marked Half below.

1. Hidden word: **HONEYCRISP**. Also accept "HONEY CRISP" (it is drawn on two lines). It shows
   in the science and difference panels. Teams usually see one alert's cutouts, so do not ask
   about all 4 alerts. It must come from the team looking at the image.
2. Light curve (get_light_curve, plot=true): **an apple**. Red r outline (34 points) with the
   orange i and green g points on top as stem and leaf; those two overlap around days 17 to 19,
   so do not require which band is which part. 49 detections, 4 upper limits. Wrong: shapes the
   model describes without the picture (it called it a periodic or sawtooth variable).
3. Host (get_object "host"; get_cross_matches NED): **M74** = MESSIER 074 (NGC 628), NED galaxy,
   z 0.002 (about 600 km/s), 10.05 Mpc, 99" (4.8 kpc) from its center. get_cross_matches shows
   the NED row without its name; the name is in get_object. Grade the redshift number: z 0.002
   is right even if the team adds a velocity the model made up (one rehearsal team, with a tool
   call, wrote "z about 0.002 (about 840 km/s)"; 0.002 is about 600 km/s). Wrong as the redshift
   (from memory, no tool call): z 0.0028, z 0.0048 (with 20.8 Mpc).
4. When it first appeared: first detection **2026-10-31 14:09:43Z** (MJD 61344.590, r 18.36).
   Last non-detection 2026-10-27 18:00Z (r fainter than 20.0); window 3.84 days, midpoint
   2026-10-29 16:05Z, estimate_explosion_time says "not obviously young". get_light_curve and
   get_object give the same dates. Full: the first detection, 2026-10-31, from a tool (14:09:43Z
   or MJD 61344.59 welcome, not required), or the estimate_explosion_time window 2026-10-27 18:00Z
   to 2026-10-31 14:09Z, or its midpoint 2026-10-29 16:05Z. Half: 2026-10-31 with a 07:12Z time
   (the signpost's minute leaking in; one rehearsal team). Wrong: any other date, and dates
   given without a tool call.
5. Anomaly (get_object, ml.applecider): **out_of_distribution flag true** (score 9.536; accept
   9.5), so BOOM does NOT find it normal. "votes 4" is a field of the flag, not a count of
   alerts; the flag is also true on all 4 alerts. AppleCiDEr abstained (top class Accreting WD
   Var 0.72). If a team asks which tool is "the anomaly detector": AppleCiDEr's
   out-of-distribution flag. find_similar_objects is not it (it returns a Milvus error today).
6. Nearest neighbors on the sky outside the hunt (cone_search about 120", then get_object on
   each): **asteroids**. Nearest is now (88562), ZTF26acgsogb at 56", one detection at
   2026-11-10 07:07:50Z (mag 20.0). Asteroid (44423) was seen three times on 2026-11-05 as it
   crossed M74, each detection its own object id: ZTF26acepbjv 79" (i 17.39), ZTF26acesbfi 85"
   (g 18.25), ZTF26acenzkw 107" (r 17.51); each has rock=true, stationary=false. Further out:
   (2432) on 2026-11-03 (ZTF26aceevfw 177", ZTF26acedrqd 201", ZTF26acebzpu 224") and (855) three
   times on 2026-11-10 (178", 191", 212"). (ZTF26openeye at 45" is the signpost, part of the hunt.)
   Accept "asteroids" with any of 88562, 44423, 855 or 2432. Count a team that lists asteroids
   from a cone search as correct even if it then doubts that is what the question meant.
   Wrong: anything from similarity search (find_similar_objects returns "Babamul error 501:
   Milvus is not enabled or misconfigured" today) and made-up lists. Seen in rehearsal with no tool call:
   ZTF26honeycrisp, ZTF26apple, ZTF26pie, ZTF26cider, ZTF26orchard, and ten rows of
   "ZTF26bqzjzq" labelled TNS SN 2026bqzjzq (TNS is also against the rules).
   The slide wording ("nearest neighbors outside the hunt") stays as is, so expect this:
   all three rehearsal teams first read it as similarity neighbors, got the Milvus error,
   and the assistant then made up neighbor lists. Say "nearest on the sky" when the
   questions go up. A team stuck on Milvus errors: point them to a cone search about 2
   arcminutes (120") around the treasure, then get_object on each neighbor.
7. BTSbot (get_object, ml.btsbot): **0.985** on the latest alert (0.990, 0.694, 0.991, 0.985 over
   the 4 alerts): "a bright supernova worth following up". Accept 0.985 (latest alert) or
   0.990 / 0.991 (another alert), so 0.98 to 0.991, with that reading,
   including answers with caveats: the latest alert (18.6 mag) is a bit fainter than the 18.5 mag
   BTSbot is tuned for, and BTSbot disagrees with AppleCiDEr.

Debrief: BTSbot rates the treasure a likely bright supernova while AppleCiDEr calls it out of
distribution; BTSbot also gives the asteroid detection ZTF26acepbjv 0.91. Signpost: BTSbot
0.065, no AppleCiDEr (only 1 point).

## What the assistant did in rehearsal (for walking around)

- No vision: it cannot read GO N 45", HONEYCRISP or the apple; teams must open the pictures.
- In long chats it stops calling tools and makes up results, even labelled "from get_object" or
  "raw result from find_similar_objects". Telling it so worked only 3 times out of 11; a new chat
  fixed it every time. Rule of thumb for teams: no tool chips under the answer means do not trust it; ask again
  in a new chat.
- It made up an object id (ZTF26abq) and called get_object on it 8 times.
- For Q6 it calls find_similar_objects unprompted and gets the Milvus error.
- get_light_curve_fit on the treasure: "no Villar fit" (expected; the treasure is too new).
- search_alerts over more than 1 day: "covers at most 1 day" (it retries with 1 day).
- Long answers can end with "This answer reached its length limit".

## Rehearsal 2026-11-10 11:04 to 11:30 CT (5 conversations at once on the new Qwen)

- All three teams found the signpost in turn 1 and the treasure in turn 2, with no hints.
  14 to 21 minutes per team (B about 14, A about 18 over two chats, C about 21); slowest turn
  231 s; no 429s, no timeouts.
- Grades: team A 6 of 7 (Q6 missing), team B 5.5 (Q4 half, Q6 missing), team C 7 of 7 (Q6 counted
  by the rule above: a cone_search plus get_object listed asteroids 88562, 44423, 2432 and 855,
  then the team said asteroids were not what it wanted). Q6 is the question that fails through
  the "similarity" reading: A and B never got it, and C got it only after asking for an on-sky
  fallback.

## History

- 2026-11-08: Q6 was similarity search (no Milvus here) and Q7 the Villar fit (none for the
  treasure); both tools return errors on our instance, so both were changed.
- "days_ago" values in tool output change with the date.
