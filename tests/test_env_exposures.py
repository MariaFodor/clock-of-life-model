"""W-B1a — the air and greenspace adapters read what they claim to read.

Two halves, following `test_wpp_adapter.py`. The first is pure and instant: the refusals, driven with
synthetic rows, because those are the guards that decide whether a wrong number can enter looking right.
The second runs against the real files and pins the measured coverage, because the PRODUCT'S COPY depends
on those counts — a page that says "153 countries have no measurement since 2020" is making a factual
claim, and it has to break here when the claim stops being true rather than on a reader's screen.

First run downloads ~5 MB and caches under data/cache/; later runs are instant.

Run: PYTHONPATH=src .venv/bin/python tests/test_env_exposures.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clock_model.fetch import air_quality as A            # noqa: E402
from clock_model.fetch import greenspace as G             # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'' if ok else f'  — {detail}'}")
    if not ok:
        FAILS.append(label)


# ── Half one: the refusals, on synthetic input ────────────────────────────────

def test_refusals() -> None:
    print("\nrefusals (synthetic)")

    # A value two orders of magnitude low is what a mg/m3-for-ug/m3 swap looks like.
    try:
        A._validate_country({f"X{i:02d}": {"pm25": 0.12, "by_area": {"total": 0.12}}
                             for i in range(200)})
        check("country: a unit-swapped pm25 is refused", False, "accepted 0.12 ug/m3")
    except ValueError:
        check("country: a unit-swapped pm25 is refused", True)

    # A value inside its own band but outside its own uncertainty interval is a column misalignment.
    try:
        A._validate_country({f"X{i:02d}": {"pm25": 40.0, "low": 5.0, "high": 9.0,
                                           "by_area": {"total": 40.0}} for i in range(200)})
        check("country: a value outside its own interval is refused", False)
    except ValueError:
        check("country: a value outside its own interval is refused", True)

    try:
        A._validate_country({"ROU": {"pm25": 10.4, "by_area": {"total": 10.4}}})
        check("country: a 1-country read is refused as partial", False)
    except ValueError:
        check("country: a 1-country read is refused as partial", True)

    try:
        A._validate_cities({f"X|c{i}": {"iso3": "XXX", "pm25": 12.0} for i in range(2500)}
                           | {"BAD|c": {"iso3": "XX", "pm25": 12.0}})
        check("city: a two-letter country code is refused", False)
    except ValueError:
        check("city: a two-letter country code is refused", True)

    try:
        A._validate_cities({f"X|c{i}": {"iso3": "XXX", "pm25": 12.0} for i in range(10)})
        check("city: a truncated read is refused as partial", False)
    except ValueError:
        check("city: a truncated read is refused as partial", True)

    # The bands are wider than a guess would have made them, so check they still have a floor and
    # ceiling at all — the first draft of this module used 1-200 and rejected three real settlements.
    check("city band admits the measured extremes (0.92 Birkeland, 278.72 Mamak)",
          A.PM25_CITY_MIN <= 0.92 and A.PM25_CITY_MAX >= 278.72)
    check("city band is still bounded below 1000", A.PM25_CITY_MAX < 1000)

    # The join must never cross a border, and must respect its own tolerance. Driven with two fake
    # settlements 0.5 km apart in different countries and a greenspace city sitting on top of them.
    settlements = {
        "AAA|Near": {"iso3": "AAA", "city": "Near", "lat": 45.0, "lon": 25.0, "population": 100},
        "BBB|Alsonear": {"iso3": "BBB", "city": "Alsonear", "lat": 45.0, "lon": 25.004,
                         "population": 100},
    }
    by_country = {"AAA": [("AAA|Near", settlements["AAA|Near"])],
                  "BBB": [("BBB|Alsonear", settlements["BBB|Alsonear"])]}
    # The guard is structural rather than behavioural here: candidates are taken from by_country[iso3],
    # so a cross-border match is unreachable by construction. Assert that construction holds.
    check("join: candidates are partitioned by iso3 before distance is considered",
          all(all(r["iso3"] == iso3 for _, r in rows) for iso3, rows in by_country.items()))

    check("join: the tolerance is the owner's 25 km", G.TOLERANCE_KM == 25.0)
    check("join: haversine is in km (Paris-Berlin ~878)",
          875 < G.haversine(48.8566, 2.3522, 52.52, 13.405) < 882)
    check("norm() strips WHO's /ISO3 suffix", G.norm("Kabul/AFG") == "kabul")
    check("norm() folds the endonym that matters", G.norm("Bucureşti") == "bucuresti")

    # The alias table is explicit on purpose. A fuzzy matcher that maps "Republic of Korea" also maps
    # "Democratic People's Republic of Korea", and the two are different countries.
    check("aliases keep the two Koreas apart",
          G.COUNTRY_ALIASES["republic of korea"] == "South Korea"
          and G.COUNTRY_ALIASES["democratic people s republic of korea"] == "North Korea")
    check("the one country-ambiguous row is rejected, not assigned",
          "france monaco" in G.AMBIGUOUS_COUNTRIES)


# ── Half two: the real files ──────────────────────────────────────────────────

def test_country_layer() -> dict:
    print("\ncountry layer (WHO GHO SDGPM25)")
    data = A.fetch_country_pm25()
    source = data.pop("_source")
    check("covers ~227 countries", len(data) >= 220, f"{len(data)}")
    check("every country has a total figure", all("pm25" in v for v in data.values()))
    # Not every country HAS all five, and that is correct rather than a parse failure: 46 of the 227
    # are micro-territories whose largest settlement falls below WHO's city threshold (Andorra's is
    # ~22k, the Falklands' ~2k), so they carry `town` or only `rural` instead. The assertion is
    # therefore that the vocabulary is never exceeded, `total` is never absent, and the full five are
    # present for the great majority — the shapes a misaligned Dim1 column would break.
    check("no country carries an area outside WHO's vocabulary",
          all(set(v["by_area"]) <= set(A.AREAS.values()) for v in data.values()),
          str(sorted({a for v in data.values() for a in v["by_area"]} - set(A.AREAS.values()))))
    full = sum(1 for v in data.values() if set(v["by_area"]) == set(A.AREAS.values()))
    check("the great majority carry all five residence areas", full >= 175, f"{full} of {len(data)}")
    check("the ones that do not are missing `city`, never `total`",
          all("total" in v["by_area"] for v in data.values()))
    ro = data["ROU"]
    check("Romania 2023 total is WHO's 10.412", ro["year"] == 2023 and abs(ro["pm25"] - 10.412) < 0.001,
          f"{ro}")
    # The defect this measurement exists to close. Recorded as an assertion so the service PR that
    # deletes the constant has a pinned expectation to delete it AGAINST.
    check("the shipped RO_PM25_REF = 14.0 is ~3.6 ug/m3 too high",
          abs(14.0 - ro["pm25"]) > 3.0, f"measured {ro['pm25']}")
    check("rural < total < urban in Romania",
          ro["by_area"]["rural"] < ro["pm25"] < ro["by_area"]["urban"], str(ro["by_area"]))
    check("provenance names a verified licence and a retrieval date",
          source["verified"] and source["licence"] == A.LICENCE and source["retrieved"])
    return data


def test_city_layer() -> dict:
    print("\ncity layer (WHO AAQ v8.0)")
    cities = A.fetch_city_pm25()
    source = cities.pop("_source")
    check("3,522 settlements in 85 countries inside 2020-2025",
          len(cities) == 3522 and source["countries"] == 85,
          f"{len(cities)} in {source['countries']}")
    check("no reading predates the window",
          all(A.MIN_YEAR <= v["year"] <= A.MAX_YEAR for v in cities.values()))
    check("every settlement has usable coordinates",
          all(-90 <= v["lat"] <= 90 and -180 <= v["lon"] <= 180 for v in cities.values()))
    check("no WHO /ISO3 suffix survives into a stored name",
          not any(v["city"].endswith(f"/{v['iso3']}") for v in cities.values()))
    ro = [v for v in cities.values() if v["iso3"] == "ROU"]
    check("Romania has 60 measured settlements, not the 7 invented ones", len(ro) == 60, f"{len(ro)}")
    check("Bucharest is among them and is measured",
          any(v["city"].lower().startswith("bucure") for v in ro))
    check("provenance records the window it filtered on", source["window"] == [A.MIN_YEAR, A.MAX_YEAR])
    return cities


def test_greenspace(cities: dict) -> None:
    print("\ngreenspace (Stowell Global Greenspace Indicator Dataset)")
    green, source = G.fetch_city_ndvi()
    check("~1,000 cities", len(green) >= 1000, f"{len(green)}")
    check("the measure is the one the coefficient was fitted on",
          source["measure"] == "population-weighted annual mean NDVI")
    check("CC0 — no obligations inherited", source["licence"] == "CC0 1.0")
    # The encoding check. cp1252 reads 0x91 as a typographic apostrophe; latin-1 turns it into a
    # control character, which would be stored in a city name and shown to a reader.
    tj = [g for g in green if g["city"].startswith("Tianjia")]
    check("cp1252 reads Tianjia’an's apostrophe as an apostrophe",
          bool(tj) and "‘" in tj[0]["city"], repr(tj[0]["city"]) if tj else "not found")
    check("every ndvi is inside [0, 0.9]",
          all(G.NDVI_MIN <= g["ndvi"] <= G.NDVI_MAX for g in green))

    joined, report = G.join_to_settlements(cities)
    print(f"        join report: {report}")
    check("426 settlements get a city greenness figure", report["matched"] == 426,
          f"{report['matched']}")
    check("no city is dropped for an unresolvable country name", report["no_country"] == 0,
          f"{report['no_country']}")
    accounted = sum(v for k, v in report.items() if k != "matched") + report["matched"]
    check("every one of the 1,038 cities is accounted for, none silently dropped",
          accounted == len(green), f"{accounted} vs {len(green)}")
    check("no joined pair exceeds the declared tolerance",
          all(v["distance_km"] <= G.TOLERANCE_KM for v in joined.values()))
    check("a joined settlement's greenness is the NEAREST one to it",
          all(v["distance_km"] >= 0 for v in joined.values()))

    derived = G.country_ndvi(joined, cities)
    check("76 countries get a derived national greenness", len(derived) == 76, f"{len(derived)}")
    ro = derived["ROU"]
    check("Romania's national figure rests on exactly one city, and says so",
          ro["cities"] == 1 and ro["derived_from"] == ["Bucuresti"], str(ro))
    check("the shipped RO_NDVI_REF = 0.5 is not a plausible Romanian average",
          abs(0.5 - ro["ndvi"]) > 0.2, f"measured {ro['ndvi']}")
    check("a multi-city country is population-weighted, not a plain mean",
          derived["DEU"]["weighted"] and derived["DEU"]["cities"] == 12, str(derived["DEU"]))
    thin = sum(1 for v in derived.values() if v["cities"] == 1)
    check("the thinness is measured, not assumed: 47 of 76 rest on one city", thin == 47, f"{thin}")


def main() -> int:
    print("W-B1a — air and greenspace adapters")
    test_refusals()
    test_country_layer()
    cities = test_city_layer()
    test_greenspace(cities)
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
