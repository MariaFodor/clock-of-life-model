"""Measured air pollution, at two granularities, from WHO.

The model prices PM2.5 through Burnett 2014, whose exposure variable is an ANNUAL MEAN PM2.5
CONCENTRATION at the person's residence. Two WHO products carry that quantity and the model needs both
for different jobs:

  * the COUNTRY layer (GHO indicator `SDGPM25`) is population-weighted and complete — 227 countries, and
    split by where people live (total / urban / rural / city / town). This is the CENTRING REFERENCE: the
    number that answers "what is an average person in this country breathing", which is what an exposure
    coefficient has to be centred on before it can price a deviation. The service hardcodes
    `RO_PM25_REF = 14.0` today; WHO's measured Romanian total for 2023 is 10.41.

  * the CITY layer (Ambient Air Quality Database v8.0) is what a reader actually wants to see — 5,311
    settlements with a PM2.5 figure across 116 countries, each a real monitoring result with a year, a
    station count and coordinates. It is also the honest half of the coverage story: 122 of the 237
    countries the atlas draws have no monitoring at all, and a map that greys them silently says
    "clean air" about places nobody has measured.

The two are NOT interchangeable and are never averaged together. A city figure is what monitors in that
settlement recorded; a country figure is a modelled population-weighted surface. Mixing them would let a
country with three monitored cities look like a country with a national estimate.

Licence: CC BY-NC-SA 3.0 IGO for both. That is non-commercial and share-alike, which is why nothing here
writes measured values into a file that other things inherit from — see `data/city_join.README.md`.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

TIMEOUT = 180

#: Bumped whenever the filter, the column set or the cached payload's shape changes, so a stale extract
#: built under different rules cannot be served as if it were this one. `fetch/wpp.py` documents the bug
#: that earned this convention: adding a field without bumping produced a cache hit missing that field.
CACHE_SCHEMA = 1

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")

#: Ceiling on any single download. The real files are 2 MB (WHO workbook) and 3.2 MB (gazetteer); a
#: redirect to something enormous should fail rather than fill the disk.
MAX_BYTES = 256 * 1024 ** 2

LICENCE = "CC BY-NC-SA 3.0 IGO"
LICENCE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/igo/"

# ── The country layer ─────────────────────────────────────────────────────────

GHO_URL = "https://ghoapi.azureedge.net/api/SDGPM25"
GHO_CITATION = ("World Health Organization. Global Health Observatory indicator SDGPM25: "
                "Concentrations of fine particulate matter (PM2.5). Geneva: WHO.")
GHO_SOURCE_URL = "https://www.who.int/data/gho/data/indicators/indicator-details/GHO/concentrations-of-fine-particulate-matter-(pm2-5)"

#: WHO's residence-area dimension, mapped to the vocabulary the service's `location.area_type` uses.
#: `TOTL` is the national population-weighted mean and is the one the centring reference is taken from;
#: the other four exist so a reader in a village is not centred on a capital-city average.
AREAS = {
    "RESIDENCEAREATYPE_TOTL": "total",
    "RESIDENCEAREATYPE_URB": "urban",
    "RESIDENCEAREATYPE_RUR": "rural",
    "RESIDENCEAREATYPE_CITY": "city",
    "RESIDENCEAREATYPE_TOWN": "town",
}

#: Outside this band a "PM2.5 concentration" is not one. Both bands are set on the MEASURED envelope of
#: the two files, not on a guess — the first attempt used 1-200 for both and was rejected by three real
#: Greek, Kazakh and Turkish settlements above 200 and two Norwegian ones below 1.
#:
#: Country figures are population-weighted national means and cluster tightly (4.12 Finland to 58.09 Qatar).
#: City figures are single-settlement annual means and legitimately reach 278.72 (Mamak, Turkiye, 2025)
#: and fall to 0.92 (Birkeland, Norway, 2025). The job of each band is to catch a unit swap — mg/m3 for
#: ug/m3 would land two orders of magnitude low — or an AQI index parsed as a concentration, not to
#: second-guess a monitor.
PM25_MIN, PM25_MAX = 1.0, 150.0
PM25_CITY_MIN, PM25_CITY_MAX = 0.5, 300.0

# ── The city layer ────────────────────────────────────────────────────────────

#: WHO Ambient Air Quality Database v8.0, released 30 June 2026.
AAQ_URL = ("https://cdn.who.int/media/docs/default-source/air-pollution-documents/"
           "air-quality-and-health/who-ambient-air-quality-database-version-2026-v8.xlsx"
           "?sfvrsn=95d122a4_5&download=true")
AAQ_FILE = "who_aaq_v8.xlsx"
AAQ_CITATION = ("World Health Organization (2026). WHO Ambient Air Quality Database, "
                "version 8.0. Geneva: WHO.")
AAQ_SOURCE_URL = ("https://www.who.int/data/gho/data/themes/air-pollution/"
                  "who-air-quality-database")

#: WHO publishes PM10 and NO2 for settlements with no PM2.5. PM2.5 is the one the model prices, so it is
#: the one that decides whether a settlement counts as measured.
POLLUTANT = "pm25_concentration"

#: Owner decision 2026-09-10: the comparison window for air against greenness is 2020-2025. A settlement
#: whose most recent PM2.5 reading predates it is not shown as a current measurement.
#:
#: This costs real coverage and the number is worth stating rather than discovering later: all-time the
#: database holds 5,311 settlements in 116 countries; inside the window it holds 3,522 in 85. Romania
#: goes from 61 to 60. The alternative — showing a 2013 reading on a surface that says "air quality
#: here" — is worse, because a decade of policy sits between that number and the reader. What the map
#: must therefore say of an unmeasured country is "no measurement since 2020", not "never measured":
#: the second would be false for 31 countries.
MIN_YEAR = 2020
MAX_YEAR = 2025


def _fetch_bytes(url: str, name: str) -> str:
    """Download once into data/cache/ and return the local path.

    Cached by name, not by URL, because these are the pinned files this module names above — a caller
    cannot ask for an arbitrary URL and have it filed as WHO provenance.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={
            # WHO's CDN answers 403 to urllib's default agent. Identifying the client is the polite
            # fix and the one that keeps working; spoofing a browser would not be either.
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


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _today() -> str:
    return datetime.date.today().isoformat()


def _cache_path(kind: str) -> str:
    return os.path.join(CACHE, f"who_{kind}_v{CACHE_SCHEMA}.json")


def _write_cache(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx_rows(path: str) -> list[list[str]]:
    """Parse WHO's single sheet with the standard library.

    Deliberately no openpyxl: this repo should not grow a dependency to read one file, and the sheet is
    a CSV in disguise — one cell per line, except that `type_of_stations` contains newlines inside its
    quoted field, which Excel splits across cells. Rejoining a row's cells with newlines and parsing the
    whole document as CSV puts those fields back together.

    First written as `experiments/exp16_city_join._read_xlsx_rows`, which now imports this one so there
    is a single parser; the experiment's committed output is the regression test for the move.
    """
    z = zipfile.ZipFile(path)
    shared = [
        "".join(t.text or "" for t in si.iter(f"{_NS}t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{_NS}si")
    ]

    def col(ref: str) -> int:
        n = 0
        for ch in re.match(r"([A-Z]+)", ref).group(1):
            n = n * 26 + ord(ch) - 64
        return n - 1

    lines = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{_NS}row"):
        cells = {}
        for c in row.findall(f"{_NS}c"):
            v = c.find(f"{_NS}v")
            cells[col(c.get("r"))] = (
                None if v is None else (shared[int(v.text)] if c.get("t") == "s" else v.text)
            )
        lines.append("\n".join(cells.get(i) or "" for i in range(max(cells) + 1)) if cells else "")
    return list(csv.reader(io.StringIO("\n".join(lines))))


def _num(x):
    return float(x) if re.match(r"^-?\d+(\.\d+)?$", (x or "").strip()) else None


def _int(x):
    return int(x) if (x or "").strip().isdigit() else None


def fetch_country_pm25(refresh: bool = False) -> dict:
    """Population-weighted national PM2.5 per ISO3, at each country's latest published year.

    Returns `{iso3: {pm25, low, high, year, by_area: {total,urban,rural,city,town}}}` plus a `_source`
    key carrying provenance. `pm25` is the `total` figure — the one an average resident breathes — and
    `by_area` is kept beside it so a reader who lives in a village is not centred on a city average.

    Each country is taken at ITS OWN latest year rather than at a common year: WHO does not publish every
    country in every year, and forcing a common year would drop countries silently. The year travels with
    the value so nothing downstream has to assume one.
    """
    path = _cache_path("sdgpm25")
    if os.path.exists(path) and not refresh:
        return json.load(open(path))

    req = urllib.request.Request(GHO_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        if r.url.split(":", 1)[0] != "https":
            raise ValueError(f"refusing a redirect off https: {r.url}")
        raw = r.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("SDGPM25: response larger than the ceiling — refusing")
    body = json.loads(raw)

    latest: dict[str, int] = {}
    for row in body["value"]:
        iso3, year = row.get("SpatialDim"), row.get("TimeDim")
        # The response mixes countries with WHO regions and World; only three-letter codes with a
        # COUNTRY spatial type are places a reader can live in.
        if row.get("SpatialDimType") != "COUNTRY" or not iso3 or len(iso3) != 3 or year is None:
            continue
        latest[iso3] = max(latest.get(iso3, 0), int(year))

    out: dict[str, dict] = {}
    for row in body["value"]:
        iso3, year, area = row.get("SpatialDim"), row.get("TimeDim"), row.get("Dim1")
        if iso3 not in latest or int(year or -1) != latest[iso3] or area not in AREAS:
            continue
        value = row.get("NumericValue")
        if value is None:
            continue
        rec = out.setdefault(iso3, {"year": latest[iso3], "by_area": {}})
        rec["by_area"][AREAS[area]] = round(float(value), 3)
        if AREAS[area] == "total":
            rec["pm25"] = round(float(value), 3)
            rec["low"] = None if row.get("Low") is None else round(float(row["Low"]), 3)
            rec["high"] = None if row.get("High") is None else round(float(row["High"]), 3)

    # A country row with every area type EXCEPT total would otherwise ship without the one figure the
    # centring reference is taken from, and read as "no data" downstream instead of as a parse failure.
    without_total = sorted(k for k, v in out.items() if "pm25" not in v)
    if without_total:
        raise ValueError(f"SDGPM25: {len(without_total)} countries have area splits but no total "
                         f"({without_total[:5]}) — refusing a partial read")
    _validate_country(out)

    out["_source"] = {
        "name": "WHO Global Health Observatory — SDGPM25",
        "citation": GHO_CITATION,
        "url": GHO_SOURCE_URL,
        "api_url": GHO_URL,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "retrieved": _today(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "countries": len(out),
        "verified": True,
    }
    _write_cache(path, out)
    return out


def _validate_country(out: dict) -> None:
    """Refuse a parse that produced numbers no PM2.5 surface can contain."""
    if len(out) < 180:
        raise ValueError(f"SDGPM25: only {len(out)} countries — expected ~227; refusing a partial read")
    for iso3, rec in out.items():
        v = rec["pm25"]
        if not PM25_MIN <= v <= PM25_MAX:
            raise ValueError(f"SDGPM25 {iso3}: pm25 {v} outside [{PM25_MIN}, {PM25_MAX}]")
        lo, hi = rec.get("low"), rec.get("high")
        if lo is not None and hi is not None and not lo <= v <= hi:
            raise ValueError(f"SDGPM25 {iso3}: {v} outside its own interval [{lo}, {hi}]")
        for area, av in rec["by_area"].items():
            if not PM25_MIN <= av <= PM25_MAX:
                raise ValueError(f"SDGPM25 {iso3}/{area}: {av} outside [{PM25_MIN}, {PM25_MAX}]")


def fetch_city_pm25(refresh: bool = False) -> dict:
    """Every settlement with a PM2.5 reading inside the comparison window, at its most recent year.

    Returns `{"AAA|City": {iso3, city, pm25, year, stations, temporal_coverage, lat, lon, population}}`
    plus `_source`. Keys are strings rather than tuples so the payload round-trips through JSON.

    Settlements outside `MIN_YEAR..MAX_YEAR` are dropped rather than carried with an old year: a 2013
    reading shown beside a 2023 one invites a comparison that is mostly a decade of policy.
    """
    path = _cache_path("aaq_cities")
    if os.path.exists(path) and not refresh:
        return json.load(open(path))

    local = _fetch_bytes(AAQ_URL, AAQ_FILE)
    rows = read_xlsx_rows(local)
    hdr = rows[0]
    idx = {name: i for i, name in enumerate(hdr)}
    for column in ("iso3", "city", "year", POLLUTANT, "latitude", "longitude"):
        if column not in idx:
            raise ValueError(f"AAQ: column {column!r} is gone — the workbook changed shape")

    data = [r for r in rows[1:] if len(r) == len(hdr)]
    if len(data) != len(rows) - 1:
        raise ValueError(f"AAQ: {len(rows) - 1 - len(data)} malformed rows — refusing a partial read")

    best: dict[str, dict] = {}
    for r in data:
        year, value = _int(r[idx["year"]]), _num(r[idx[POLLUTANT]])
        if year is None or value is None or not MIN_YEAR <= year <= MAX_YEAR:
            continue
        lat, lon = _num(r[idx["latitude"]]), _num(r[idx["longitude"]])
        if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            # A settlement with no usable coordinates cannot be drawn, cannot be joined to greenness
            # and cannot be checked against the country it claims. It is not a place here.
            continue
        iso3, city = r[idx["iso3"]], r[idx["city"]]
        key = f"{iso3}|{city}"
        if key in best and best[key]["year"] >= year:
            continue
        best[key] = {
            "iso3": iso3,
            # WHO suffixes some names with their own code ("Kabul/AFG"); nothing downstream wants that.
            "city": re.sub(r"\s*/\s*[A-Z]{3}\s*$", "", city),
            "pm25": round(value, 2),
            "year": year,
            "stations": _int(r[idx["number_stations"]]) if "number_stations" in idx else None,
            "temporal_coverage": _num(r[idx["pm25_tempcov"]]) if "pm25_tempcov" in idx else None,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "population": _int(r[idx["population"]]) if "population" in idx else None,
        }

    _validate_cities(best)
    best["_source"] = {
        "name": "WHO Ambient Air Quality Database v8.0",
        "citation": AAQ_CITATION,
        "url": AAQ_SOURCE_URL,
        "file_url": AAQ_URL,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "retrieved": _today(),
        "sha256": sha256(local),
        "window": [MIN_YEAR, MAX_YEAR],
        "settlements": len(best),
        "countries": len({v["iso3"] for k, v in best.items() if k != "_source"}),
        "verified": True,
    }
    _write_cache(path, best)
    return best


def _validate_cities(best: dict) -> None:
    if len(best) < 2000:
        raise ValueError(f"AAQ: only {len(best)} settlements in {MIN_YEAR}-{MAX_YEAR} — refusing")
    for key, rec in best.items():
        if not PM25_CITY_MIN <= rec["pm25"] <= PM25_CITY_MAX:
            raise ValueError(f"AAQ {key}: pm25 {rec['pm25']} outside "
                             f"[{PM25_CITY_MIN}, {PM25_CITY_MAX}]")
        if len(rec["iso3"]) != 3:
            raise ValueError(f"AAQ {key}: iso3 {rec['iso3']!r} is not a three-letter code")
