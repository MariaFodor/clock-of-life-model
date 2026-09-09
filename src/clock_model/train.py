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
from clock_model.evaluate import ontology_gates as OG
from clock_model.export import bundle
from clock_model.fetch import eurostat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def load_cohort(from_raw: bool) -> pd.DataFrame:
    if from_raw:
        from clock_model.ingest import harmonize
        return harmonize.build_cohort()
    return pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))


def build_country_baseline(iso: str, coefs: dict, rates: dict) -> dict | None:
    try:
        lt = eurostat.fetch_lifetable(iso, LIFETABLE_YEAR)
    except Exception as e:
        print(f"  {iso}: life table unavailable ({e}); skipped")
        return None
    if not lt.get("M") or not lt.get("F"):
        print(f"  {iso}: no life-table data for {LIFETABLE_YEAR}; skipped")
        return None
    prevalence = eurostat.fetch_prevalence(iso)
    lp_young = centring.reference_vector(coefs, rates, prevalence, age=40)
    lp_old = centring.reference_vector(coefs, rates, prevalence, age=60)
    return {
        "country": iso,
        "qx": {"M": {str(a): q for a, q in lt["M"].items()},
               "F": {str(a): q for a, q in lt["F"].items()}},
        "reference_lp": {"young": lp_young, "old": lp_old},
        "national_le_40": {"M": round(baselines.remaining_le(lt["M"], 40, 1.0), 2),
                           "F": round(baselines.remaining_le(lt["F"], 40, 1.0), 2)},
        "prevalence": prevalence,
    }


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
    print(f"[train] building baselines for {len(country_list)} countries …")
    built = {}
    for iso in country_list:
        b = build_country_baseline(iso, fit["prediction_coefs"], rates)
        if b:
            built[iso] = b
    print(f"        built {len(built)} country baselines")

    root = bundle.assemble(ARTIFACTS, version, fit, gates, built)
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
