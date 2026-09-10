"""EXP-16 (W-B0) — how many cities have BOTH a measured air figure and a measured greenness figure?

Nobody knew. Five designers proposing the per-city surface guessed 512, 612, 812 and "about 600", and
the design that came out of it said plainly: if the real number is around 180, greenness is decoration
and the surface is a different shape. So this runs before any of it is built.

It also answers a question the design assumed away. Its stated method was "match on GHS urban-centre id,
or centroids within a stated tolerance, with ISO3 equality required; never name similarity" — and that
is impossible with the published files. Stowell's greenspace table carries a city NAME and a country
NAME and nothing else: no coordinates, no identifier, despite being derived from the GHS Urban Centre
Database, whose ids were dropped at publication. WHO's side, by contrast, carries latitude and longitude
for every settlement.

So the join goes through a gazetteer: greenspace name -> GeoNames coordinates -> nearest WHO settlement
within a declared tolerance, ISO3 equality required. The decisive hop is geographic. The first hop is a
name lookup because the data leaves no alternative, and every city it fails to resolve is written to the
output as a rejection with its reason rather than dropped.

WHAT THIS EMITS, and why it is shaped this way: `data/city_join.csv` is a KEY MAPPING and carries no
measured value at all — no PM2.5, no NDVI. WHO's database is CC BY-NC-SA 3.0 IGO and Stowell's is CC0;
keeping the measurements in their own files and joining by key means the ShareAlike obligation attaches
to a WHO-derived file rather than to everything downstream of it. That was an owner decision, and this
file is where it is enforced.

Run: PYTHONPATH=src .venv/bin/python experiments/exp16_city_join.py
First run downloads ~12 MB and caches it under data/cache/ (gitignored); later runs are instant.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "data", "city_join.csv")

TIMEOUT = 180

# ── Sources, pinned ───────────────────────────────────────────────────────────

SOURCES = {
    # WHO Ambient Air Quality Database v8.0, released 30 June 2026. CC BY-NC-SA 3.0 IGO.
    "who_aaq": (
        "https://cdn.who.int/media/docs/default-source/air-pollution-documents/air-quality-and-health/"
        "who-ambient-air-quality-database-version-2026-v8.xlsx?sfvrsn=95d122a4_5&download=true",
        "who_aaq_v8.xlsx",
    ),
    # Stowell et al., Data in Brief 2023, doi:10.1016/j.dib.2023.109140.
    # Data: doi:10.7910/DVN/TMWYHB, Harvard Dataverse, CC0 1.0. File id is stable per version.
    "greenspace": (
        "https://dataverse.harvard.edu/api/access/datafile/6903364",
        "greenspace_data_share.csv",
    ),
    # GeoNames, CC BY 4.0 — used ONLY to give the greenspace cities coordinates, never as a measurement.
    "gazetteer": ("https://download.geonames.org/export/dump/cities15000.zip", "cities15000.zip"),
    "countryinfo": ("https://download.geonames.org/export/dump/countryInfo.txt", "countryInfo.txt"),
}

#: Two settlements this far apart are the same urban agglomeration for exposure purposes. Declared
#: rather than tuned: the counts at 10 / 25 / 50 km are all reported so the choice stays auditable.
TOLERANCE_KM = 25.0

#: WHO publishes PM10 and NO2 for settlements that have no PM2.5. PM2.5 is the one the model prices
#: (Burnett 2014), so it is the one that decides whether a city has "an air figure" here.
POLLUTANT = "pm25_concentration"


def _fetch(key: str) -> str:
    """Download once into data/cache/ and return the local path. Prints a SHA-256 for provenance."""
    url, name = SOURCES[key]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"  fetching {name} …", flush=True)
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            if r.url.split(":", 1)[0] != "https":
                raise ValueError(f"refusing a redirect off https: {r.url}")
            data = r.read()
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"  {name}: {os.path.getsize(path):,} bytes, sha256:16 {digest}")
    return path


# ── WHO's workbook ────────────────────────────────────────────────────────────

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_xlsx_rows(path: str) -> list[list[str]]:
    """Parse the single sheet with the standard library.

    Deliberately no openpyxl: this repo should not grow a dependency to read one file, and the sheet
    is a CSV in disguise — one cell per line, except that `type_of_stations` contains newlines inside
    its quoted field, which Excel splits across cells. Rejoining a row's cells with newlines and then
    parsing the whole document as CSV puts those fields back together.
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


def _year(x):
    return int(x) if (x or "").strip().isdigit() else None


def who_settlements() -> tuple[dict, dict]:
    """`{(iso3, city): row}` at each settlement's latest year WITH a PM2.5 figure, plus the header map."""
    rows = _read_xlsx_rows(_fetch("who_aaq"))
    hdr = rows[0]
    idx = {name: i for i, name in enumerate(hdr)}
    data = [r for r in rows[1:] if len(r) == len(hdr)]
    if len(data) != len(rows) - 1:
        raise SystemExit(f"[who] {len(rows) - 1 - len(data)} malformed rows — refusing a partial read")
    latest = {}
    for r in data:
        key = (r[idx["iso3"]], r[idx["city"]])
        y = _year(r[idx["year"]])
        if y is None or _num(r[idx[POLLUTANT]]) is None:
            continue
        if key not in latest or y > _year(latest[key][idx["year"]]):
            latest[key] = r
    every = {(r[idx["iso3"]], r[idx["city"]]) for r in data}
    print(f"  settlements with any pollutant: {len(every):,}")
    print(f"  settlements with a PM2.5 figure: {len(latest):,} "
          f"in {len({k[0] for k in latest}):,} countries")
    return latest, idx


# ── Names ─────────────────────────────────────────────────────────────────────


def norm(s: str) -> str:
    """Fold to something two datasets can agree on: no accents, no case, no punctuation.

    Also strips WHO's "/ISO3" suffix ("Kabul/AFG"). This is NOT fuzzy matching — nothing here matches
    two different strings; it only removes typography that both sides render differently.
    """
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\s*/\s*[a-z]{3}\s*$", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def gazetteer() -> tuple[dict, dict]:
    """`{(iso2, normalised name): (lat, lon)}` from GeoNames, most populous match winning."""
    path = _fetch("gazetteer")
    z = zipfile.ZipFile(path)
    by_name = defaultdict(list)
    with z.open("cities15000.txt") as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8"):
            f = raw.rstrip("\n").split("\t")
            if len(f) < 15:
                continue
            name, ascii_name, alts, lat, lon, cc, pop = (
                f[1], f[2], f[3], float(f[4]), float(f[5]), f[8], int(f[14] or 0))
            for n in {norm(name), norm(ascii_name)} | {norm(a) for a in alts.split(",") if a}:
                if n:
                    by_name[(cc, n)].append((pop, lat, lon))
    for k in by_name:
        by_name[k].sort(reverse=True)

    iso2_to_iso3 = {}
    with open(_fetch("countryinfo"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) > 2:
                iso2_to_iso3[f[0]] = f[1]
    print(f"  gazetteer: {len(by_name):,} name keys, {len(iso2_to_iso3)} country codes")
    return by_name, iso2_to_iso3


def haversine(a_lat, a_lon, b_lat, b_lon) -> float:
    r, p = 6371.0, math.pi / 180
    return 2 * r * math.asin(math.sqrt(
        math.sin((b_lat - a_lat) * p / 2) ** 2
        + math.cos(a_lat * p) * math.cos(b_lat * p) * math.sin((b_lon - a_lon) * p / 2) ** 2))


# ── The join ──────────────────────────────────────────────────────────────────


def main() -> int:
    print("EXP-16 — how many cities have both a measured air figure and a measured greenness figure?\n")
    print("sources")
    latest, idx = who_settlements()
    by_name, iso2_to_iso3 = gazetteer()

    who_points = defaultdict(list)
    for (iso3, city), r in latest.items():
        la, lo = _num(r[idx["latitude"]]), _num(r[idx["longitude"]])
        if la is not None and lo is not None:
            who_points[iso3].append((la, lo, city, _year(r[idx["year"]])))
    located = sum(len(v) for v in who_points.values())
    print(f"  of those, with coordinates: {located:,} ({located / len(latest) * 100:.0f}%)")

    # Latin-1: the published file is not UTF-8, and guessing wrong silently corrupts city names.
    with open(_fetch("greenspace"), encoding="latin-1") as fh:
        green = list(csv.DictReader(fh))
    print(f"  greenspace cities: {len(green):,}\n")

    rows, distances = [], []
    for row in green:
        gcity, gcountry = row["City"], row["Country"]
        key = norm(gcity)
        candidates = [cc for (cc, n) in by_name if n == key]
        if not candidates:
            rows.append((gcity, gcountry, "", "", "rejected_no_geocode", ""))
            continue
        pop, la, lo, cc = max(
            ((*by_name[(c, key)][0], c) for c in candidates), key=lambda t: t[0])
        iso3 = iso2_to_iso3.get(cc, "")
        points = who_points.get(iso3)
        if not points:
            rows.append((gcity, gcountry, iso3, "", "rejected_country_unmeasured", ""))
            continue
        dist, who_city = min(
            ((haversine(la, lo, p_la, p_lo), p_city) for p_la, p_lo, p_city, _ in points),
            key=lambda t: t[0])
        distances.append(dist)
        if dist <= TOLERANCE_KM:
            rows.append((gcity, gcountry, iso3, who_city, f"gazetteer_{TOLERANCE_KM:g}km", f"{dist:.1f}"))
        else:
            rows.append((gcity, gcountry, iso3, "", "rejected_too_far", f"{dist:.1f}"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        # No measured value in this file, by design — see the module docstring on ShareAlike.
        w.writerow(["greenspace_city", "greenspace_country", "who_iso3", "who_city", "method", "distance_km"])
        w.writerows(sorted(rows))

    method = Counter(r[4] for r in rows)
    joined = method[f"gazetteer_{TOLERANCE_KM:g}km"]
    print(f"wrote {os.path.relpath(OUT, ROOT)} — {len(rows):,} rows, every greenspace city accounted for")
    for m, n in method.most_common():
        print(f"  {m:<28} {n:>5}")
    print()
    for tol in (10, 25, 50):
        n = sum(1 for d in distances if d <= tol)
        mark = "  <-- declared tolerance" if tol == TOLERANCE_KM else ""
        print(f"  within {tol:>3} km of a measured settlement: {n:>4}{mark}")

    print(f"\nVERDICT: {joined} of {len(green):,} greenspace cities join to a settlement with a measured "
          f"PM2.5 figure at {TOLERANCE_KM:g} km. Air alone covers {len(latest):,} settlements in "
          f"{len({k[0] for k in latest}):,} countries. The design's threshold for treating greenness as "
          f"decoration rather than a layer was about 180.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
