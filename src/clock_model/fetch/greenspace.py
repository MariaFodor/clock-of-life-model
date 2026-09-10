"""Measured residential greenness per city, and the join that gives it coordinates.

The model prices greenness through Rojas-Rueda 2019, whose exposure variable is POPULATION-WEIGHTED
ANNUAL MEAN NDVI in a residential buffer. Exactly one published dataset carries that quantity at city
scale: Stowell et al.'s Global Greenspace Indicator Dataset. Forest-cover percentage was rejected —
applying a coefficient fitted on residential NDVI to land cover is the unit substitution the ontology
gates exist to prevent.

THE JOIN, and why it is shaped like this. Stowell's table carries a city NAME and a country NAME and
nothing else: no coordinates, no identifier, despite being derived from the GHS Urban Centre Database,
whose ids were dropped at publication. WHO's air settlements carry latitude and longitude. So greenness
reaches a settlement through a gazetteer: name -> GeoNames coordinates -> nearest WHO settlement within
a declared tolerance, ISO3 equality required. The decisive hop is GEOGRAPHIC; the name lookup exists
only because the data leaves no alternative, and it is also what makes the join work at all — matching
on names alone missed Bucharest/Bucuresti and 144 other endonyms (383 joins by name, 527 by gazetteer).

WHAT THIS CANNOT DO, stated because the product has to say it out loud. Only 63 of the 1,915 measured
settlements inside the 30 scoreable countries have an NDVI value, and 24 of those 30 countries have
exactly one green city — the capital. Stowell covers ~1,000 large cities worldwide and Europe's share is
small. So `country_ndvi()` below derives a national figure from however many cities a country has, and
records `cities` alongside it, because a "national greenness" built from one city has to be labelled as
one rather than presented as a measurement of the country.

Licence: CC0 1.0 for the greenness values (no obligations), CC BY 4.0 for the GeoNames gazetteer, which
is used ONLY to supply coordinates and never as a measurement.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import math
import os
import re
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict

TIMEOUT = 180
CACHE_SCHEMA = 1
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")
MAX_BYTES = 256 * 1024 ** 2

#: Stowell et al., Data in Brief 2023. The data DOI resolves to Harvard Dataverse; the file id is stable
#: per version, which is why it is pinned rather than resolved through the DOI at fetch time.
GREEN_URL = "https://dataverse.harvard.edu/api/access/datafile/6903364"
GREEN_FILE = "greenspace_data_share.csv"
GREEN_DOI = "10.1016/j.dib.2023.109140"
GREEN_DATA_DOI = "10.7910/DVN/TMWYHB"
GREEN_CITATION = ("Stowell JD, Ngo S, Bell ML (2023). A global dataset of greenspace indicators for "
                  "urban areas. Data in Brief 49:109140.")
GREEN_LICENCE = "CC0 1.0"
GREEN_LICENCE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

#: GeoNames, used for coordinates only.
GAZ_URL = "https://download.geonames.org/export/dump/cities15000.zip"
GAZ_FILE = "cities15000.zip"
COUNTRYINFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
COUNTRYINFO_FILE = "countryInfo.txt"
GAZ_LICENCE = "CC BY 4.0"
GAZ_LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"

#: Two settlements this far apart are the same urban agglomeration for exposure purposes. Owner decision
#: 2026-09-10, carried over from EXP-16 which reported the counts at 10 / 25 / 50 km so the choice stays
#: auditable. The circle the map draws around each measured settlement is THIS radius — it is what the
#: reading is being claimed to speak for.
TOLERANCE_KM = 25.0

#: The population-weighted annual mean is the quantity Rojas-Rueda 2019 measured. `peak_NDVI_*` (summer
#: maximum) and `indicator_*` (a categorical label) are the other two things in the file and are neither.
#: Years are tried newest first; the year that supplied the value travels with it.
NDVI_COLUMN = "annual_weight_avg_{year}"
NDVI_YEARS = (2021, 2020, 2015, 2010)

#: Owner decision: the air-vs-greenness comparison window is 2020-2025, so a 2015 or 2010 NDVI is parsed
#: but never used as a current figure. It stays in NDVI_YEARS so a city missing from the recent sweep is
#: reported as stale rather than as absent.
MIN_YEAR = 2020

#: NDVI is a ratio in [-1, 1] by construction (water and snow are negative, dense vegetation approaches
#: 1). Population-weighted urban annual means occupy a much narrower band, and the measured envelope of
#: this file is 0.04 to 0.64 — which is why the service's RO_NDVI_REF = 0.5 is not a plausible national
#: average for Romania: it sits in the top few per cent of cities on Earth.
NDVI_MIN, NDVI_MAX = 0.0, 0.9

#: Stowell's country column uses UN long forms; GeoNames uses short ones. Every entry here was read off
#: `countryInfo.txt` rather than guessed, and the map is EXPLICIT rather than fuzzy on purpose: a string
#: distance that matches "Republic of Korea" to "South Korea" also matches it to "North Korea". Without
#: this, 132 cities — including every city in the United States, the Netherlands, Russia, Iran and South
#: Korea — were silently dropped as "country not found".
COUNTRY_ALIASES = {
    "bolivarian republic of venezuela": "Venezuela",
    "cape verde": "Cabo Verde",
    "cote d ivoire": "Ivory Coast",
    "democratic people s republic of korea": "North Korea",
    "islamic republic of iran": "Iran",
    "macedonia": "North Macedonia",
    "netherlands": "The Netherlands",
    "occupied palestinian territory": "Palestinian Territory",
    "republic of korea": "South Korea",
    "russian federation": "Russia",
    "syrian arab republic": "Syria",
    "united republic of tanzania": "Tanzania",
    "united states of america": "United States",
}

#: Deliberately NOT aliased: Stowell has one row whose country is "France;Monaco", a city the dataset
#: itself could not assign to a country. It is rejected as ambiguous rather than given to either.
AMBIGUOUS_COUNTRIES = {"france monaco"}


def _fetch_bytes(url: str, name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={
            "User-Agent": "clock-of-life-model/1.0 (research; +https://github.com/MariaFodor)",
            "Accept": "*/*",
        })
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            if r.url.split(":", 1)[0] != "https":
                raise ValueError(f"refusing a redirect off https: {r.url}")
            data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"{name}: larger than {MAX_BYTES} bytes — refusing")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return path


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _today() -> str:
    return datetime.date.today().isoformat()


def norm(s: str) -> str:
    """Fold to something two datasets can agree on: no accents, no case, no punctuation.

    Also strips WHO's "/ISO3" suffix ("Kabul/AFG"). This is NOT fuzzy matching — nothing here matches two
    different strings; it only removes typography that the two files render differently.
    """
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\s*/\s*[a-z]{3}\s*$", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def haversine(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r, p = 6371.0, math.pi / 180
    return 2 * r * math.asin(math.sqrt(
        math.sin((b_lat - a_lat) * p / 2) ** 2
        + math.cos(a_lat * p) * math.cos(b_lat * p) * math.sin((b_lon - a_lon) * p / 2) ** 2))


def gazetteer() -> tuple[dict, dict]:
    """`{(iso2, normalised name): [(pop, lat, lon), ...]}` and `{iso2: iso3}` from GeoNames.

    Every alternate name a place has is indexed, most populous first — that is the hop that resolves
    Bucharest to Bucuresti. Ambiguity is handled by population rather than by guessing, and the distance
    check downstream is what actually decides whether the resolved point is the right place.
    """
    z = zipfile.ZipFile(_fetch_bytes(GAZ_URL, GAZ_FILE))
    by_name: dict[tuple[str, str], list] = defaultdict(list)
    with z.open("cities15000.txt") as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8"):
            f = raw.rstrip("\n").split("\t")
            if len(f) < 15:
                continue
            name, ascii_name, alts = f[1], f[2], f[3]
            lat, lon, cc, pop = float(f[4]), float(f[5]), f[8], int(f[14] or 0)
            for n in {norm(name), norm(ascii_name)} | {norm(a) for a in alts.split(",") if a}:
                if n:
                    by_name[(cc, n)].append((pop, lat, lon))
    for k in by_name:
        by_name[k].sort(reverse=True)

    iso2_to_iso3 = {}
    with open(_fetch_bytes(COUNTRYINFO_URL, COUNTRYINFO_FILE), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) > 2:
                iso2_to_iso3[f[0]] = f[1]
    if len(iso2_to_iso3) < 200:
        raise ValueError(f"gazetteer: only {len(iso2_to_iso3)} country codes — refusing a partial read")
    return by_name, iso2_to_iso3


def fetch_city_ndvi() -> tuple[list[dict], dict]:
    """Stowell's table as `[{city, country, ndvi, year, hdi_level, climate, region}]`, plus provenance.

    Each city is taken at its newest available year; the year travels with the value because a 2010 NDVI
    and a 2021 one are eleven years of urban change apart and the product has to be able to say which.
    """
    path = _fetch_bytes(GREEN_URL, GREEN_FILE)
    # cp1252, not utf-8 and not latin-1. The published file is Windows-encoded and utf-8 raises on the
    # e-acute in "Medellin" and 22 other names. EXP-16 read it as latin-1 and that was very nearly
    # right: the two encodings differ only over 0x80-0x9F, and this file contains exactly one such byte
    # — 0x91, the typographic apostrophe in the Chinese city Tianjia'an. latin-1 turns it into a control
    # character. The join is unaffected either way because norm() strips punctuation, but this module
    # STORES the name, so the encoding that reads it correctly is the one to use.
    with open(path, encoding="cp1252", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("greenspace: empty file — refusing")
    for year in NDVI_YEARS:
        if NDVI_COLUMN.format(year=year) not in rows[0]:
            raise ValueError(f"greenspace: column {NDVI_COLUMN.format(year=year)!r} is gone — "
                             "the file changed shape")

    out = []
    for r in rows:
        value = year_used = None
        for year in NDVI_YEARS:
            raw = (r.get(NDVI_COLUMN.format(year=year)) or "").strip()
            if raw:
                try:
                    value, year_used = float(raw), year
                except ValueError:
                    continue
                break
        if value is None:
            continue
        if not NDVI_MIN <= value <= NDVI_MAX:
            raise ValueError(f"greenspace {r.get('City')}: ndvi {value} outside "
                             f"[{NDVI_MIN}, {NDVI_MAX}] — refusing")
        out.append({
            "city": r["City"],
            "country": r["Country"],
            "ndvi": round(value, 4),
            "year": year_used,
            "hdi_level": r.get("HDI_level") or None,
            "climate_region": r.get("Climate_region") or None,
            "region": r.get("Major_Geo_Region") or None,
        })

    if len(out) < 900:
        raise ValueError(f"greenspace: only {len(out)} cities — expected ~1,000; refusing a partial read")
    source = {
        "name": "Global Greenspace Indicator Dataset",
        "citation": GREEN_CITATION,
        "doi": GREEN_DOI,
        "data_doi": GREEN_DATA_DOI,
        "url": f"https://doi.org/{GREEN_DATA_DOI}",
        "file_url": GREEN_URL,
        "licence": GREEN_LICENCE,
        "licence_url": GREEN_LICENCE_URL,
        "retrieved": _today(),
        "sha256": _sha256(path),
        "measure": "population-weighted annual mean NDVI",
        "cities": len(out),
        "verified": True,
    }
    return out, source


def join_to_settlements(settlements: dict, tolerance_km: float = TOLERANCE_KM) -> tuple[dict, dict]:
    """Attach a greenness value to every WHO settlement that has one.

    `settlements` is `fetch.air_quality.fetch_city_pm25()`'s payload. Returns
    `({settlement_key: {ndvi, ndvi_year, distance_km, matched_city}}, report)` where `report` counts every
    rejection by reason — the rejections are the coverage story and are never silently dropped.

    ISO3 equality is REQUIRED before distance is even considered, so no reading can cross a border; a
    25 km radius near a frontier would otherwise let one country's monitor supply another's greenness.
    """
    green, _ = fetch_city_ndvi()
    by_name, iso2_to_iso3 = gazetteer()
    iso3_to_iso2 = {v: k for k, v in iso2_to_iso3.items()}

    # Country NAMES in Stowell's file against ISO2 in the gazetteer. Done once, reported, never guessed.
    name_to_iso2 = {}
    with open(_fetch_bytes(COUNTRYINFO_URL, COUNTRYINFO_FILE), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) > 4:
                name_to_iso2[norm(f[4])] = f[0]

    by_country: dict[str, list] = defaultdict(list)
    for key, rec in settlements.items():
        if key != "_source":
            by_country[rec["iso3"]].append((key, rec))

    joined: dict[str, dict] = {}
    # Every greenspace city ends up in exactly one bucket. `superseded` is the one that is easy to
    # forget and was: two greenspace cities can resolve to the same WHO settlement, the nearer one wins,
    # and the loser was initially counted nowhere — six cities vanished from a report that claimed to
    # account for all 1,038. The test that caught it asserts the buckets sum to the input.
    report = {"matched": 0, "no_country": 0, "ambiguous_country": 0, "no_geocode": 0,
              "country_unmeasured": 0, "too_far": 0, "superseded": 0}
    for g in green:
        want = norm(g["country"])
        if want in AMBIGUOUS_COUNTRIES:
            report["ambiguous_country"] += 1
            continue
        iso2 = name_to_iso2.get(norm(COUNTRY_ALIASES.get(want, g["country"])))
        if not iso2:
            report["no_country"] += 1
            continue
        iso3 = iso2_to_iso3.get(iso2)
        if not iso3 or iso3 not in by_country:
            report["country_unmeasured"] += 1
            continue
        candidates = by_name.get((iso2, norm(g["city"])))
        if not candidates:
            report["no_geocode"] += 1
            continue
        _, glat, glon = candidates[0]
        best_key, best_d = None, None
        for key, rec in by_country[iso3]:
            d = haversine(glat, glon, rec["lat"], rec["lon"])
            if best_d is None or d < best_d:
                best_key, best_d = key, d
        if best_d is None or best_d > tolerance_km:
            report["too_far"] += 1
            continue
        # One WHO settlement can be the nearest to two greenspace cities; the closer one wins so the
        # value attached to a settlement is always the nearest measurement to it.
        if best_key in joined and joined[best_key]["distance_km"] <= best_d:
            report["superseded"] += 1
            continue
        if best_key in joined:
            # This city is closer than the one already there; the incumbent becomes the superseded one.
            report["superseded"] += 1
        joined[best_key] = {
            "ndvi": g["ndvi"],
            "ndvi_year": g["year"],
            "distance_km": round(best_d, 1),
            "matched_city": g["city"],
        }
        report["matched"] = len(joined)
    accounted = sum(v for k, v in report.items() if k != "matched") + report["matched"]
    if accounted != len(green):
        raise ValueError(f"join: {accounted} cities accounted for out of {len(green)} — "
                         "a rejection reason is missing and cities are vanishing silently")
    return joined, report


def country_ndvi(joined: dict, settlements: dict) -> dict:
    """A national greenness figure per ISO3, derived from that country's own measured cities.

    `{iso3: {ndvi, cities, year_min, year_max, derived_from}}`. This is the fallback the product shows
    where a settlement has no city measurement of its own, and it is NOT a measurement of the country:
    24 of the 30 scoreable countries have exactly one green city, so `cities` travels with the value and
    `derived_from` names them. The display obligation that follows is the owner's decision of
    2026-09-10 — every value carries the word for where it came from.

    Population-weighted where populations are known, unweighted where they are not, and which of the two
    was used is recorded rather than assumed: a plain mean over Berlin and a Bavarian town would hand
    them equal say in what Germans breathe.
    """
    groups: dict[str, list] = defaultdict(list)
    for key, g in joined.items():
        rec = settlements[key]
        groups[rec["iso3"]].append((g, rec))

    out = {}
    for iso3, items in groups.items():
        pops = [rec.get("population") or 0 for _, rec in items]
        weighted = all(p > 0 for p in pops) and len(items) > 1
        if weighted:
            total = sum(pops)
            value = sum(g["ndvi"] * p for (g, _), p in zip(items, pops)) / total
        else:
            value = sum(g["ndvi"] for g, _ in items) / len(items)
        years = [g["ndvi_year"] for g, _ in items]
        out[iso3] = {
            "ndvi": round(value, 4),
            "cities": len(items),
            "weighted": weighted,
            "year_min": min(years),
            "year_max": max(years),
            "derived_from": sorted(rec["city"] for _, rec in items)[:10],
        }
    return out
