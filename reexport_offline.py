"""Offline bundle re-export: refit the cohort model and rebuild the versioned bundle WITHOUT network.

Reuses the vendored per-country life-table data (qx, national_le_40, prevalence) from a previous bundle in
artifacts/, recomputing only each country's reference_lp with the new coefficients. Use this when the
cohort model changed (e.g. EXP-14 added BMI + current-smoking dose) but the country life tables did not.

    PYTHONPATH=src .venv/bin/python reexport_offline.py --from-version 2.0.0 --version 2.1.0
"""
from __future__ import annotations
import argparse, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from clock_model.model import cox, centring, fit as FIT
from clock_model.evaluate import gates as G
from clock_model.evaluate import ontology_gates as OG
from clock_model.export import bundle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ARTIFACTS = os.path.join(HERE, "artifacts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-version", default="2.0.0", help="bundle in artifacts/ to reuse country data from")
    ap.add_argument("--version", default="2.1.0", help="new bundle version to write")
    args = ap.parse_args()

    df = pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))
    fit = FIT.fit_models(df)
    gates = G.evaluate(fit)
    print(f"[reexport] n={fit['n']} deaths={fit['deaths']} "
          f"C-index={gates['c_index']} calibration={gates['calibration_mae']} passed={gates['passed']}")
    if not gates["passed"]:
        sys.exit("[reexport] GATES FAILED — bundle not written.")

    # The ontology's own gates decide the release too, not just the numeric thresholds. Without
    # this the module docstring's promise ("a violation refuses the release") was untrue: the checks
    # existed but only the test suite ran them (PR#2 F2).
    ont_results = OG.check(fit)
    ont_failed = [f"{n}: {d}" for n, ok, d in ont_results if not ok]
    if ont_failed:
        for f in ont_failed:
            print(f"[gate] REFUSED — {f}")
        sys.exit(f"[gate] {len(ont_failed)} ontology gate(s) failed — bundle not written.")
    print(f"[gate] {len(ont_results)} ontology gates passed")

    rates = centring.cohort_rates(fit["cohort"])
    src = os.path.join(ARTIFACTS, f"model-v{args.from_version}", "baselines")
    built = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".json"):
            continue
        iso = fn[:-5]
        prev = json.load(open(os.path.join(src, fn)))
        # Carry the whole baseline forward and re-derive ONLY the centring, which is the one field
        # that depends on the coefficients being re-fitted. Enumerating the fields to copy is how this
        # script would quietly emit a bundle labelled v4 with v3-shaped contents — `Bundle::load`
        # accepts it, and the missing identity/provenance would surface as a map with no country names.
        built[iso] = dict(prev)
        if prev.get("reference_lp") is not None:
            prevalence = prev.get("prevalence")
            built[iso]["reference_lp"] = {
                "young": centring.reference_vector(fit["prediction_coefs"], rates, prevalence, age=40),
                "old": centring.reference_vector(fit["prediction_coefs"], rates, prevalence, age=60),
            }
    root = bundle.assemble(ARTIFACTS, args.version, fit, gates, built)
    print(f"[reexport] wrote {root}  ({len(built)} countries, {len(fit['prediction_coefs'])} coefficients)")


if __name__ == "__main__":
    main()
