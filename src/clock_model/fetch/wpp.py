"""Fetch UN World Population Prospects 2024: national life tables for every country or area.

Eurostat's `demo_mlifetable` is a better measurement where it reaches — registered deaths, one year
fresher — but it reaches 30 EU/EEA countries, so no world map can ever be drawn from it, and a map drawn
from a second agency contradicts the clock (WHO's 2021 estimates put Romanian men at 69.2 where the
shipped clock says 72.8). WPP is the only source that covers both at the granularity the model needs:
single year of age, by sex, 237 countries or areas. Measured cost of the switch across the 30 countries
we already ship: median |Δe₀| 0.27 years, max 1.73, measured by W-A1's
`experiments/exp15_wpp_vs_eurostat.py` (a sibling PR; this module does not depend on it).

`fetch/eurostat.py` stays, and stops being a source: it becomes the independent witness the WPP tables
are cross-checked against at release time. Two agencies that must agree within a declared band is a
stronger claim than one agency asserting itself.

Source, verified 2026-09-10 from the rendered download page (which is client-side rendered — a plain
fetch of the HTML shows none of this):

    Copyright © 2024 by United Nations, made available under a Creative Commons license CC BY 3.0 IGO:
    http://creativecommons.org/licenses/by/3.0/igo/

    Suggested citation: United Nations, Department of Economic and Social Affairs, Population Division
    (2024). World Population Prospects 2024, Online Edition.

Attribution only — commercial use permitted, unlike WHO's CC BY-NC-SA data.
"""
from __future__ import annotations

import csv
import datetime
import gzip
import hashlib
import io
import json
import os
import urllib.request

#: Per-operation read timeout, passed to urlopen rather than set globally: socket.setdefaulttimeout
#: is process-wide, and eurostat.py sets 30 for itself at import — whichever module imported last
#: would decide the timeout for both, and for every other socket in the process.
TIMEOUT = 180
#: A newline-free gzip stream expands without limit inside readline() — measured at 2841x wire bytes
#: before csv ever sees a field. The real files are ~4 GB decompressed.
MAX_DECOMPRESSED = 8 * 1024 ** 3
#: Bumped whenever the filter, the age grid, a column name OR the cached payload's shape changes, so a
#: stale extract built under different rules cannot be served as if it were this one. It has already
#: earned its keep once: adding `published` to the payload without bumping it produced a cache hit with
#: no published figures, which silently emptied the gate that checks the tables against the publisher.
CACHE_SCHEMA = 3
#: The whole reason for the switch is that WPP covers the world. A partial upstream publish that was
#: consistent across all three files would otherwise yield a structurally perfect 50-country atlas.
MIN_COUNTRIES = 200

BASE = "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES"
PORTAL = "https://population.un.org/wpp/"
LICENCE = "CC BY 3.0 IGO"
LICENCE_URL = "http://creativecommons.org/licenses/by/3.0/igo/"
#: The exact page the licence and citation were read from — the portal root does not carry them, and
#: a citation whose source cannot be re-opened is not a citation (this repo's own convention).
LICENCE_SOURCE_URL = "https://population.un.org/wpp/downloads"
CITATION = ("United Nations, Department of Economic and Social Affairs, Population Division (2024). "
            "World Population Prospects 2024, Online Edition.")

#: The last ESTIMATE year of the 2024 revision. Everything after it is a projection scenario, which
#: would look exactly like an estimate by the time it reached a life table.
LATEST_ESTIMATE_YEAR = 2023

LIFETABLE_FILES = {
    "M": "WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv.gz",
    "F": "WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv.gz",
    # Both sexes together is a separate file for a real reason: a both-sex life table needs the
    # population's sex structure, so it cannot be recovered by averaging the male and female tables.
    "B": "WPP2024_Life_Table_Complete_Medium_Both_1950-2023.csv.gz",
}
INDICATORS_FILE = "WPP2024_Demographic_Indicators_Medium.csv.gz"

#: Each life-table file holds exactly one sex, so the file we opened decides the sex we record. That is
#: an assumption about someone else's filenames, and the failure it hides is the classic one here — a
#: swapped sex dimension reads as a plausible country with the wrong people in it — so it is checked
#: against the column rather than trusted.
EXPECTED_SEX = {"M": "Male", "F": "Female", "B": "Total"}

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")

#: Eurostat calls Greece EL; ISO 3166-1 — and therefore WPP — calls it GR. Nothing else in the 30
#: countries we already ship differs, but stored calculations and deployed clients still say EL.
ALIASES = {"EL": "GR"}

MAX_AGE = 100


class _Fingerprinted:
    """File-like wrapper that fingerprints bytes as they stream past.

    The provenance the bundle records has to name WHICH file was read, not just the URL it came from:
    a 200 MB CSV silently revised upstream would otherwise move the shipped estimate with nothing to
    point at afterwards.
    """

    def __init__(self, raw, limit: int = MAX_DECOMPRESSED):
        self._raw = raw
        self._hash = hashlib.sha256()
        self._limit, self._seen = limit, 0

    def read(self, n=-1):
        chunk = self._raw.read(n)
        self._hash.update(chunk)
        self._seen += len(chunk)
        if self._seen > self._limit:
            raise ValueError(f"read past {self._limit} bytes — refusing to continue")
        return chunk

    @property
    def sha256(self) -> str:
        return self._hash.hexdigest()


def _stream_csv_gz(filename: str, on_row) -> str:
    """Stream one gzipped CSV straight off the socket, row by row; return its SHA-256.

    These files decompress to several gigabytes and hold 1950-2023 for every location and every single
    year of age, of which one year of one column is wanted. So nothing is written to disk whole, and
    `csv.reader` with column indices is used rather than `DictReader` — building ten million dicts to
    discard all but a few thousand is most of the runtime.

    `on_row(index_map, row)` is called for every data row.
    """
    with urllib.request.urlopen(f"{BASE}/{filename}", timeout=TIMEOUT) as resp:
        # urllib follows a 302 from https to http without comment. The digest below would then be a
        # faithful fingerprint of somebody else's file, filed as WPP provenance.
        if resp.url.split(":", 1)[0] != "https":
            raise ValueError(f"refusing a redirect off https: {resp.url}")
        fp = _Fingerprinted(resp)
        # newline="" as the csv module requires: without it a quoted field containing CRLF is split.
        with io.TextIOWrapper(gzip.GzipFile(fileobj=fp), encoding="utf-8-sig", newline="") as text:
            reader = csv.reader(text)
            head = next(reader)
            idx = {name: pos for pos, name in enumerate(head)}
            for row in reader:
                on_row(idx, row)
        return fp.sha256


def _cache_path(kind: str, year: int) -> str:
    """Cache paths are built from an int and a schema number, never from caller-supplied text."""
    return os.path.join(CACHE, f"wpp2024_{kind}_{int(year)}_v{CACHE_SCHEMA}.json")


def _today() -> str:
    return datetime.date.today().isoformat()


def _write_cache(path: str, payload: dict) -> None:
    """Write via a temp file and rename: an interrupted write must not leave a file the next run trusts."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def _require_estimate_year(year: int) -> int:
    """The single line that keeps forecasts out.

    `Variant == "Medium"` reads like a guard against projections and is not one: "Medium" IS the name
    of WPP's central projection scenario, and the indicators file carries Time 1950-2101 with every one
    of those 78 post-2023 forecast years labelled Medium. So the year is the only thing separating an
    estimate from a forecast, and it has to be enforced rather than defaulted — `fetch_locations(2024)`
    would otherwise return a complete, structurally perfect 237-country set of PROJECTED life
    expectancy and cache it under a provenance block that says "estimates only".
    """
    year = int(year)
    if year > LATEST_ESTIMATE_YEAR:
        raise ValueError(f"WPP {year} is a projection, not an estimate: the 2024 revision's last "
                         f"estimate year is {LATEST_ESTIMATE_YEAR}. Refusing to build a life table "
                         f"from a forecast.")
    return year


def _keep(idx: dict, row: list, year: int) -> bool:
    """Four refusals, each guarding a way the wrong number could arrive looking right."""
    if row[idx["Time"]] != str(year):
        return False
    # Trusting the filename's word "Complete" for single-year-of-age would be the same mistake the
    # Sex check below exists to avoid; an abridged row would overwrite a single-year one silently.
    if "AgeGrpSpan" in idx and row[idx["AgeGrpSpan"]] not in ("1", "-1"):
        return False
    # The files open with regional aggregates ("ADB region: Central and West Asia") that carry no ISO
    # code. A country baseline built from one of those would be nonsense that still validates.
    if row[idx["LocTypeName"]] != "Country/Area" or not row[idx["ISO2_code"]]:
        return False
    # Kept for shape, not for safety: inside these files every row is "Medium". What actually excludes
    # a forecast is _require_estimate_year, at the entry points.
    return row[idx["Variant"]] == "Medium"


def fetch_lifetables(year: int = LATEST_ESTIMATE_YEAR) -> dict[str, dict[str, dict[int, float]]]:
    """`{iso2: {'M'|'F'|'B': {age: qx}}}` for every country or area, ages 0-100.

    Cached under data/cache/ (gitignored), keyed by year and by CACHE_SCHEMA — so changing the filter
    or the age grid busts it. It is NOT keyed on the upstream files' fingerprints: those are recorded
    in the cache and in the bundle's provenance, but verifying them would mean re-downloading 600 MB
    to discover a hit, which defeats the cache. A WPP revision therefore has to be picked up by
    bumping the year or clearing the cache, not automatically.
    """
    year = _require_estimate_year(year)
    path = _cache_path("lifetables", year)
    if os.path.exists(path):
        with open(path) as fh:
            cached = json.load(fh)
        tables = {iso: {sex: {int(a): q for a, q in ages.items()} for sex, ages in sexes.items()}
                  for iso, sexes in cached["countries"].items()}
        # Validated on the way OUT as well as in. A cache is a file on a disk someone else can fill,
        # truncate or interrupt, and the failure it produces is not an exception — it is a life table
        # that still integrates and comes back years wrong.
        _validate(tables, year)
        return tables

    os.makedirs(CACHE, exist_ok=True)
    out: dict[str, dict[str, dict[int, float]]] = {}
    published: dict[str, dict[str, dict[str, float]]] = {}
    digests: dict[str, str] = {}
    for sex, filename in LIFETABLE_FILES.items():
        def on_row(idx, row, _sex=sex, _file=filename):
            if not _keep(idx, row, year):
                return
            if row[idx["Sex"]] != EXPECTED_SEX[_sex]:
                raise ValueError(f"{_file} holds {row[idx['Sex']]!r} rows, not {EXPECTED_SEX[_sex]!r} — "
                                 f"refusing to file them under sex {_sex!r}")
            iso2 = row[idx["ISO2_code"]]
            ages = out.setdefault(iso2, {}).setdefault(_sex, {})
            age = int(row[idx["AgeGrpStart"]])
            if age in ages:
                # Two Country/Area rows sharing an ISO2 would last-writer-win into one complete,
                # in-range, correctly-closing table that no structural check can see through.
                raise ValueError(f"{iso2}/{_sex}: age {age} appears twice in {_file}")
            ages[age] = float(row[idx["qx"]])
            # The table's own published life expectancy at birth, and the years it prices its open
            # interval at. Both ride along on the same stream and turn the e0 gate from "catches a
            # sex swap" into "catches a one-year time slip" — see the note in the test.
            if age == 0:
                published.setdefault(iso2, {}).setdefault(_sex, {})["ex0"] = float(row[idx["ex"]])
            elif age == MAX_AGE:
                published.setdefault(iso2, {}).setdefault(_sex, {})["ax_last"] = float(row[idx["ax"]])
        digests[filename] = _stream_csv_gz(filename, on_row)

    _validate(out, year)
    _write_cache(path, {"year": year, "sha256": digests, "retrieved": _today(),
                        "published": published,
                        "countries": {iso: {sex: {str(a): q for a, q in ages.items()}
                                            for sex, ages in sexes.items()}
                                      for iso, sexes in out.items()}})
    return out


def _validate(tables: dict[str, dict[str, dict[int, float]]], year: int) -> None:
    """Refuse the whole download rather than let one malformed country through.

    A life table that is subtly wrong does not fail — it produces a plausible number for a real person,
    which is the failure this project is least able to detect after the fact.
    """
    if len(tables) < MIN_COUNTRIES:
        raise ValueError(f"WPP {year}: only {len(tables)} countries survived the filter; expected at "
                         f"least {MIN_COUNTRIES} — the coverage this source was chosen for")
    expected = set(range(0, MAX_AGE + 1))
    for iso, sexes in sorted(tables.items()):
        missing = set(LIFETABLE_FILES) - set(sexes)
        if missing:
            raise ValueError(f"WPP {year}: {iso} has no table for {sorted(missing)}")
        for sex, qx in sexes.items():
            if set(qx) != expected:
                gaps = sorted(expected - set(qx))[:5]
                raise ValueError(f"WPP {year}: {iso}/{sex} is not a complete 0-{MAX_AGE} table "
                                 f"(missing {gaps}{'…' if len(gaps) == 5 else ''})")
            bad = [(a, q) for a, q in qx.items() if not 0.0 <= q <= 1.0]
            if bad:
                raise ValueError(f"WPP {year}: {iso}/{sex} has qx outside [0,1] at {bad[:3]}")
            if qx[MAX_AGE] != 1.0:
                raise ValueError(f"WPP {year}: {iso}/{sex} does not close at age {MAX_AGE} "
                                 f"(qx={qx[MAX_AGE]}) — the table would run past its own data")


def _validate_locations(places: dict[str, dict], year: int) -> None:
    """The published figures here are what the life tables are checked AGAINST.

    Every comparison in the gate is written `if published is not None`, which is correct — WPP does not
    publish every column for every place — but it means a half-empty locations table does not fail the
    gate, it quietly shrinks it, and the run still reports a healthy median over whatever survived. A
    witness that can go missing without anyone noticing is not a witness.
    """
    if len(places) < MIN_COUNTRIES:
        raise ValueError(f"WPP {year}: only {len(places)} locations; expected at least {MIN_COUNTRIES}")
    for iso2, rec in sorted(places.items()):
        if not (isinstance(rec.get("iso3"), str) and len(rec["iso3"]) == 3 and rec.get("name")):
            raise ValueError(f"WPP {year}: {iso2} has no usable identity ({rec.get('iso3')!r})")
    with_e0 = sum(1 for r in places.values()
                  if all((r.get("published", {}).get("e0") or {}).get(s) is not None for s in ("M", "F", "B")))
    if with_e0 < MIN_COUNTRIES:
        raise ValueError(f"WPP {year}: only {with_e0} locations carry a published e0 for all three "
                         f"sexes; the cross-check would silently cover {with_e0} of {len(places)}")


def fetch_published(year: int = LATEST_ESTIMATE_YEAR) -> dict[str, dict[str, dict[str, float]]]:
    """`{iso2: {sex: {"ex0", "ax_last"}}}` — WPP's OWN answer for each table, off the same stream.

    `ex0` is the life expectancy at birth WPP publishes for exactly this table, and `ax_last` is the
    years it prices its own open interval at. Together they let the gate check the adapter against the
    publisher without going near the indicators file, and they explain the residual: `remaining_le`
    prices the 100+ interval at half a year where WPP prices it at ~2.8, which is 91% of the gap.
    """
    fetch_lifetables(year)                                  # populates the cache if it is cold
    with open(_cache_path("lifetables", year)) as fh:
        published = json.load(fh).get("published")
    # Refuse rather than return {}. Every comparison downstream is written `if published is not None`,
    # so an empty witness does not fail the gate — it quietly shrinks it to nothing while the run still
    # reports a healthy-looking median over whatever survived.
    if not published or len(published) < MIN_COUNTRIES:
        raise ValueError(f"WPP {year}: the cached extract carries no published figures to check "
                         f"against (bump CACHE_SCHEMA or clear data/cache/)")
    return published


def resolve(iso2: str) -> str:
    """Eurostat's code for a country to WPP's. Applied at EVERY lookup, not just the obvious one —
    `fetch_locations()["EL"]` raising KeyError while `fetch_lifetable("EL")` worked is the kind of
    asymmetry that surfaces as a missing country three layers downstream."""
    return ALIASES.get(iso2, iso2)


def fetch_lifetable(iso2: str, year: int = LATEST_ESTIMATE_YEAR) -> dict[str, dict[int, float]]:
    """One country, in the shape `fetch/eurostat.py` returns so the two adapters are interchangeable."""
    return fetch_lifetables(year)[resolve(iso2)]


def location(iso2: str, year: int = LATEST_ESTIMATE_YEAR) -> dict:
    """One country's identity and published figures, alias-resolved."""
    return fetch_locations(year)[resolve(iso2)]


def fetch_locations(year: int = LATEST_ESTIMATE_YEAR) -> dict[str, dict]:
    """`{iso2: {iso3, name, region, published}}` from the 16 MB indicators companion file.

    `published` carries WPP's own life expectancy and 15-60 mortality for the same country and year.
    Nothing scores off it: it is the witness the model's own integrator is checked against, so that a
    misread column or a shifted age grid cannot pass as a plausible country.
    """
    year = _require_estimate_year(year)
    path = _cache_path("locations", year)
    if os.path.exists(path):
        with open(path) as fh:
            places = json.load(fh)["countries"]
        _validate_locations(places, year)
        return places

    os.makedirs(CACHE, exist_ok=True)
    names: dict[str, str] = {}          # LocID -> name, for resolving each country's parent region
    parents: dict[str, str] = {}        # iso2 -> ParentID
    countries: dict[str, dict] = {}

    def on_row(idx, row):
        names[row[idx["LocID"]]] = row[idx["Location"]]
        if not _keep(idx, row, year):
            return
        iso2 = row[idx["ISO2_code"]]
        parents[iso2] = row[idx["ParentID"]]
        num = lambda col: float(row[idx[col]]) if row[idx[col]] else None      # noqa: E731
        countries[iso2] = {
            "iso3": row[idx["ISO3_code"]],
            "name": row[idx["Location"]],
            "region": None,                                                     # filled in below
            "published": {"e0": {"M": num("LExMale"), "F": num("LExFemale"), "B": num("LEx")},
                          "am": {"M": num("Q1560Male"), "F": num("Q1560Female"), "B": num("Q1560")}},
        }

    sha = _stream_csv_gz(INDICATORS_FILE, on_row)
    for iso2, rec in countries.items():
        rec["region"] = names.get(parents[iso2])
    _validate_locations(countries, year)

    _write_cache(path, {"year": year, "sha256": {INDICATORS_FILE: sha}, "retrieved": _today(),
                        "countries": countries})
    return countries


def source_metadata(year: int = LATEST_ESTIMATE_YEAR) -> dict:
    """The provenance block written into every baseline — what was read, from where, when, under what.

    `retrieved` is the date the download happened, recorded INTO the cache when it was written rather
    than read back off the file's mtime — a copy, a restore or a stray `touch` would otherwise relabel
    someone else's data with today's date, and a retrieval date that can drift is not provenance. It
    is recorded PER FILE: the life tables and the indicators are separate downloads and can come from
    different days, and collapsing them to one date would assert something nobody checked.
    """
    files, retrieved = {}, {}
    for kind in ("lifetables", "locations"):
        path = _cache_path(kind, year)
        if os.path.exists(path):
            with open(path) as fh:
                cached = json.load(fh)
            for name, digest in cached.get("sha256", {}).items():
                files[name] = {"sha256": digest, "retrieved": cached.get("retrieved")}
                retrieved[name] = cached.get("retrieved")
    return {
        "dataset": "UN World Population Prospects 2024",
        "publisher": "United Nations, Department of Economic and Social Affairs, Population Division",
        "variant": "Medium — estimates only (1950-2023); never a projection",
        "year": year,
        "url": PORTAL,
        "files": files,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "licence_read_from": LICENCE_SOURCE_URL,
        "citation": CITATION,
        # The newest of the per-file dates, for a caller that wants one; `files` carries them all.
        "retrieved": max(retrieved.values()) if retrieved else None,
    }
