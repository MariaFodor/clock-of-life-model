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

    rates = centring.cohort_rates(fit["cohort"])
    src = os.path.join(ARTIFACTS, f"model-v{args.from_version}", "baselines")
    built = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".json"):
            continue
        iso = fn[:-5]
        prev = json.load(open(os.path.join(src, fn)))
        prevalence = prev.get("prevalence")
        built[iso] = {
            "country": iso,
            "qx": prev["qx"],
            "reference_lp": {
                "young": centring.reference_vector(fit["prediction_coefs"], rates, prevalence, age=40),
                "old": centring.reference_vector(fit["prediction_coefs"], rates, prevalence, age=60),
            },
            "national_le_40": prev["national_le_40"],
            "prevalence": prevalence,
        }
    root = bundle.assemble(ARTIFACTS, args.version, fit, gates, built)
    print(f"[reexport] wrote {root}  ({len(built)} countries, {len(fit['prediction_coefs'])} coefficients)")


if __name__ == "__main__":
    main()
