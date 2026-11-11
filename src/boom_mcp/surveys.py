"""What the model needs to know about each survey's data, beside the
conventions in app.CONVENTIONS that hold for all of them. Every survey the
tools accept needs an entry."""

SURVEY_NOTES = {
    "ZTF": """ZTF (the Zwicky Transient Facility, at Palomar):

- Bands. g, r and i (ztfg, ztfr, ztfi). The public survey covers the northern \
sky in g and r every few nights, so a gap of a few days is the cadence, not a \
fade. Single exposures reach about 20.5 mag.
- History. A new transient on top of an older ZTF source inherits its object id \
(ZTF, then the two-digit year it was first seen), and n_detections and \
first_detected on an alert (ndethist, jdstarthist) count every detection there \
since the survey began. Date the current event from the light curve, not from \
those.
- Reliability is drb, ZTF's deep-learning real/bogus score. Alerts below ~0.5 \
are mostly artifacts.
- Flags. star means the nearest Pan-STARRS source within 2 arcsec looks \
point-like, so a transient in a compact galaxy or a quasar can be flagged a \
star. rock and solar_system_object mean a known asteroid was predicted nearby; \
unknown asteroids are not flagged, so a lone detection that is not stationary \
may still be one.
- Machine learning. ACAI scores (hosted, nuclear, variable_star, orphan, bogus) \
are independent 0-1 scores, not a distribution. BTSbot is the probability that \
a bright (<18.5 mag) transient is a supernova worth Bright Transient Survey \
follow-up. AppleCiDEr is hierarchical (domain, family, class) and abstains when \
unsure: an abstention or a family-level decision is information, not a missing \
value. Its out_of_distribution flag marks alerts unlike its training set, which \
is how BOOM flags anomalies. Many flagged alerts are known asteroids \
(flags.rock), so pass exclude_solar_system=true when looking for anomalous \
transients.""",
    "LSST": """Rubin/LSST (the Vera C. Rubin Observatory, in Chile):

- Bands. u, g, r, i, z and y (lsstu to lssty). Single visits reach about 24.5 \
mag in r, some 4 mag deeper than ZTF, so most LSST transients are too faint \
for ZTF and a missing ZTF counterpart says little.
- Object ids. A diaObjectId, a large integer (also written LSST<digits>): pass \
and quote it as a string, like a candid. A solar-system source with no \
diaObject is sso<ssObjectId>.
- Upper limits. LSST alerts carry none: non-detections come only from forced \
photometry, so call get_light_curve with include_forced_photometry=true to see \
them. get_object never shows a limit before the first detection, which does \
not mean the rise is unconstrained.
- Reliability comes from Rubin's own real/bogus model, so ZTF's ~0.5 rule \
does not carry over.
- Flags. star and near_brightstar come from the Legacy Survey point-source \
catalog and are null outside its footprint (the Galactic plane, for one). \
exclude_stars and exclude_near_bright_stars keep only alerts flagged false, so \
they also drop every alert outside that footprint. rock means Rubin linked the \
source to a known solar-system object.
- Machine learning. BOOM runs no classifiers on LSST alerts: AppleCiDEr, ACAI \
and BTSbot, and the cuts on them, are ZTF only.""",
}
