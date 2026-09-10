"""Retrainable training entrypoint: fetch → (ingest) → fit → baselines → evaluate → export.

  python -m clock_model.train --countries all --version 2.0.0
  python -m clock_model.train --countries RO,DE,FR
  python -m clock_model.train --check-updates
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import pandas as pd

from clock_model.config import countries as C
from clock_model.config.countries import LIFETABLE_YEAR
from clock_model.model import cox, centring, baselines
from clock_model.model import fit as FIT
from clock_model.evaluate import gates as G
from clock_model.evaluate import baseline_gates as BG
from clock_model.evaluate import ontology_gates as OG
from clock_model.export import bundle
from clock_model.fetch import air_quality, eurostat, greenspace, wpp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def load_cohort(from_raw: bool) -> pd.DataFrame:
    if from_raw:
        from clock_model.ingest import harmonize
        return harmonize.build_cohort()
    return pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))


def build_env_reference() -> tuple[dict, dict, dict]:
    """The measured exposure reference per country, plus the real settlements behind it.

    Returns `(by_iso3, places, sources)`:
      * `by_iso3` — what an average person in that country is exposed to. This is the CENTRING
        reference the scoring path needs: `env_term` prices a deviation from it, so a wrong reference
        is a wrong number for every reader who has a home location. The service hardcodes
        `RO_PM25_REF = 14.0` and `RO_NDVI_REF = 0.5` today against measured 10.412 and 0.2539.
      * `places` — every settlement with a PM2.5 reading inside the comparison window, with its own
        greenness where one exists and the country's where it does not. This is what replaces the
        seven invented Romanian rows.
      * `sources` — provenance for all three upstream files.

    The air and greenness halves are NOT symmetrical and the asymmetry is carried rather than hidden.
    Air is genuinely per-city: 3,522 settlements. Greenness exists for 426 of them, so the rest take
    their country's derived figure and every value records which of the two it is — `ndvi_basis` is
    `"city"` or `"country"`. A reader must be able to see that Bacau's greenness is Bucharest's.
    """
    country_pm = air_quality.fetch_country_pm25()
    pm_source = country_pm.pop("_source")
    cities = air_quality.fetch_city_pm25()
    city_source = cities.pop("_source")
    joined, report = greenspace.join_to_settlements(cities)
    _, green_source = greenspace.fetch_city_ndvi()
    derived = greenspace.country_ndvi(joined, cities)

    by_iso3 = {}
    for iso3, pm in country_pm.items():
        green = derived.get(iso3)
        by_iso3[iso3] = {
            "pm25": pm["pm25"],
            "pm25_low": pm.get("low"),
            "pm25_high": pm.get("high"),
            "pm25_year": pm["year"],
            "pm25_by_area": pm["by_area"],
            # None, not 0.0, where there is no measurement. A zero would be priced as "this country is
            # barren"; the absence has to stay an absence all the way to the screen.
            "ndvi": None if green is None else green["ndvi"],
            "ndvi_year": None if green is None else green["year_max"],
            "ndvi_cities": 0 if green is None else green["cities"],
            "ndvi_weighted": None if green is None else green["weighted"],
            "ndvi_derived_from": None if green is None else green["derived_from"],
        }

    places = []
    for key, rec in cities.items():
        g = joined.get(key)
        fallback = derived.get(rec["iso3"])
        places.append({
            "iso3": rec["iso3"],
            "city": rec["city"],
            "lat": rec["lat"],
            "lon": rec["lon"],
            "population": rec.get("population"),
            "pm25": rec["pm25"],
            "pm25_year": rec["year"],
            "pm25_stations": rec.get("stations"),
            "pm25_temporal_coverage": rec.get("temporal_coverage"),
            "ndvi": g["ndvi"] if g else (fallback["ndvi"] if fallback else None),
            "ndvi_year": g["ndvi_year"] if g else (fallback["year_max"] if fallback else None),
            # The word the screen has to show. Owner decision 2026-09-10: every value carries where it
            # came from, because a country figure rendered bare reads as a measurement of this city.
            "ndvi_basis": "city" if g else ("country" if fallback else None),
            "ndvi_matched_city": g["matched_city"] if g else None,
            "ndvi_distance_km": g["distance_km"] if g else None,
        })
    places.sort(key=lambda r: (r["iso3"], r["city"]))

    sources = {"air_country": pm_source, "air_city": city_source, "greenspace": green_source,
               "greenspace_join": {"tolerance_km": greenspace.TOLERANCE_KM, "report": report}}
    return by_iso3, places, sources


def build_baselines(coefs: dict, rates: dict, scoreable_geos: list[str], env: dict) -> dict:
    """Every country the UN publishes a life table for, plus centring for the subset we can score.

    Two populations in one directory, and the difference is not cosmetic. A baseline with a
    `reference_lp` can be scored: `/api/meta` derives the country list from it, and the reader gets a
    personal number. A baseline without one is REFERENCE ONLY — it exists so the map can draw it and
    so a reader can see where their country sits, and the service must refuse to score against it.

    The dividing line is prevalence, not life tables. Life tables now cover 237 countries; the Cox
    reference person is centred on national smoking and overweight rates, and those come from Eurostat
    EHIS, which covers Europe. Giving the other 200 a reference person built from US cohort means would
    produce a confident, wrong, personal number for two-thirds of the world.
    """
    tables = wpp.fetch_lifetables()
    places = wpp.fetch_locations()
    source = wpp.source_metadata()
    # Prevalence is fetched with the EUROSTAT code (EL), the baseline is keyed on the ISO one (GR).
    by_iso = {wpp.resolve(geo): geo for geo in scoreable_geos}

    built: dict[str, dict] = {}
    for iso2, qx in tables.items():
        place = places.get(iso2, {})
        built[iso2] = {
            "country": iso2,
            "iso3": place.get("iso3"),
            "name": place.get("name"),
            "region": place.get("region"),
            "lifetable_year": wpp.LATEST_ESTIMATE_YEAR,
            "qx": {sex: {str(a): q for a, q in ages.items()} for sex, ages in qx.items()},
            "national_le_40": {sex: round(baselines.remaining_le(ages, 40, 1.0), 2)
                               for sex, ages in qx.items()},
            "source": source,
        }
        # Keyed by ISO3 upstream, attached by ISO2 here. Absent for a country WHO has never measured —
        # and absent is the right shape: the service must refuse to price an exposure it has no
        # reference for rather than price it against someone else's country.
        if place.get("iso3") in env:
            built[iso2]["env_reference"] = env[place["iso3"]]
        geo = by_iso.get(iso2)
        if geo is None:
            continue
        prevalence = eurostat.fetch_prevalence(geo)
        built[iso2]["prevalence"] = prevalence
        # Switzerland publishes no EHIS figures. Calling what happens next a "cohort-mean fallback"
        # would be false: `cohort_rates` carries no current-smoking rate at all, so the smoking term
        # is simply 0 and the reference person is a NON-SMOKER of cohort-average weight — a healthier
        # reference than the country's average, which inflates every Swiss user's relative risk (by
        # x1.125 as measured). It ships that way today and this PR does not change the number; what
        # it changes is that the artifact now says what the number actually is.
        built[iso2]["prevalence_source"] = (
            f"Eurostat EHIS ({geo})" if prevalence
            else (f"NO PREVALENCE for {geo} — reference person is a non-smoker of cohort-average "
                  f"weight, not this country's average person; relative risk here is overstated"))
        built[iso2]["reference_lp"] = {
            "young": centring.reference_vector(coefs, rates, prevalence, age=40),
            "old": centring.reference_vector(coefs, rates, prevalence, age=60),
        }
    missing = sorted(set(by_iso) - set(built))
    if missing:
        raise SystemExit(f"[train] {len(missing)} scoreable countries have no WPP life table: {missing}")
    return built


def train(country_list, version, from_raw=False):
    print(f"[train] cohort ({'raw NHANES' if from_raw else 'vendored harmonized'}) …")
    df = load_cohort(from_raw)
    fit = FIT.fit_models(df)
    print(f"[train] fitted: n={fit['n']} deaths={fit['deaths']}")

    print("[train] evaluating gates …")
    gates = G.evaluate(fit)
    # The ontology's own gates decide the release too, not just the numeric thresholds. Without this
    # the promise "a violation refuses the release" was untrue: the checks existed but only the test
    # suite ran them (PR#2 F2).
    ont_failed = [f"{n}: {d}" for n, ok, d in OG.check(fit) if not ok]
    if ont_failed:
        for f in ont_failed:
            print(f"[gate] REFUSED — {f}")
        sys.exit(f"[gate] {len(ont_failed)} ontology gate(s) failed — bundle not written.")
    print(f"        C-index={gates['c_index']}  calibration_mae={gates['calibration_mae']}  "
          f"passed={gates['passed']}")
    if not gates["passed"]:
        sys.exit("[train] GATES FAILED — bundle not exported.")

    rates = centring.cohort_rates(fit["cohort"])
    print("[train] fetching measured exposures (WHO air, Stowell greenspace) …")
    env, places, env_sources = build_env_reference()
    print(f"        {len(env)} country references, {len(places):,} settlements, "
          f"{sum(1 for p in places if p['ndvi_basis'] == 'city'):,} with their own greenness")
    print(f"[train] building baselines (life tables for every country, centring for {len(country_list)}) …")
    built = build_baselines(fit["prediction_coefs"], rates, country_list, env)
    scoreable = [iso for iso, b in built.items() if b.get("reference_lp")]
    print(f"        built {len(built)} baselines, {len(scoreable)} of them scoreable")

    # The independent witness: Eurostat is no longer a source, and this is the job it keeps. Fetched
    # per country and allowed to be incomplete — the gate decides whether what it covered is enough.
    witness = {}
    for geo in country_list:
        try:
            witness[geo] = eurostat.fetch_lifetable(geo, LIFETABLE_YEAR)
        except Exception as e:                                    # noqa: BLE001
            print(f"        witness: {geo} unavailable ({e})")
    bl_results = BG.check(built, witness=witness)
    bl_failed = [f"{n}: {d}" for n, ok, d in bl_results if not ok]
    if bl_failed:
        for f in bl_failed:
            print(f"[gate] REFUSED — {f}")
        # Not `continue`, and not a per-country skip: a life table that fails a gate is a wrong number
        # for a real person, and the only safe response is to ship nothing.
        sys.exit(f"[gate] {len(bl_failed)} baseline gate(s) failed — bundle not written.")
    print(f"[gate] baseline gates passed ({len(built)} countries)")

    root = bundle.assemble(ARTIFACTS, version, fit, gates, built, baseline_gates=bl_results,
                           places=places, env_sources=env_sources)
    print(f"[train] bundle → {root}")
    # sanity: RO average ≈ national LE
    if "RO" in built:
        ro = built["RO"]
        print(f"        sanity: RO national LE@40 M={ro['national_le_40']['M']} F={ro['national_le_40']['F']}")
    return root


def check_updates():
    from clock_model.fetch import nhanes
    print("[check-updates] querying upstream sources …\n")

    # Eurostat life tables: is a newer year than we use available?
    latest = eurostat.latest_lifetable_year("RO")
    if latest is None:
        print("  Eurostat: could not read available years (offline?).")
    elif latest > LIFETABLE_YEAR:
        print(f"  Eurostat: newer life table available — {latest} (we use {LIFETABLE_YEAR}). "
              f"Bump LIFETABLE_YEAR and rerun; baselines refresh, coefficients unchanged.")
    else:
        print(f"  Eurostat: up to date (latest {latest}, we use {LIFETABLE_YEAR}).")

    # NHANES cycles beyond the training window
    print("\n  NHANES cycles beyond the 2007-2014 training window:")
    for cyc, suf in [("2015-2016", "I"), ("2017-2018", "J"), ("2021-2022", "L")]:
        avail = nhanes.cycle_available(cyc, suf)
        note = ("exists, but few/no deaths under the 2019 mortality follow-up"
                if cyc <= "2015-2016" else "exists, but NO mortality linkage yet — not trainable")
        print(f"    {cyc}: {'available' if avail else 'not found'}" + (f" — {note}" if avail else ""))

    # NCHS mortality linkage — the binding gate (no machine-readable version endpoint)
    print("\n  NCHS Linked Mortality: this build uses the 2019 follow-up. A newer public linkage is the")
    print("    binding gate for adding training cycles. Verify current release at")
    print("    https://www.cdc.gov/nchs/data-linkage/mortality-public.htm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="RO", help="'all' or comma-separated ISO codes")
    ap.add_argument("--version", default="2.0.0")
    ap.add_argument("--from-raw", action="store_true", help="ingest raw NHANES instead of the vendored cohort")
    ap.add_argument("--check-updates", action="store_true")
    a = ap.parse_args()
    if a.check_updates:
        return check_updates()
    countries = C.EUROSTAT_COUNTRIES if a.countries == "all" else a.countries.split(",")
    train(countries, a.version, from_raw=a.from_raw)


if __name__ == "__main__":
    main()
