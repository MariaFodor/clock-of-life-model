"""Golden-number regression: the lifelines pipeline must reproduce the witnessed experiment results.
Run: PYTHONPATH=src .venv/bin/python tests/test_golden.py  (no network needed — uses vendored cohort)
"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from clock_model.model import cox, centring, baselines
from clock_model.evaluate import gates
from clock_model.fetch import eurostat

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    df = pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))
    fit = cox.fit_models(df)
    g = gates.evaluate(fit)
    checks = []
    checks.append(("n == 18839", fit["n"] == 18839))
    checks.append(("deaths == 2119", fit["deaths"] == 2119))
    # REFIT-01 dropped BMI (owner decision 2026-09-09): 0.820 -> 0.805. The lost 0.015 was bought
    # by a large negative bmi coefficient fighting waist (r ~ 0.9). Band is tight because the fit is
    # deterministic (three runs agree to 10 dp); it brackets the witnessed value, and its upper
    # bound is what would catch a BMI reintroduction.
    checks.append(("C-index in [0.798, 0.812]", 0.798 <= g["c_index"] <= 0.812))
    checks.append(("bmi is not a fitted feature (REFIT-01)", "bmi" not in fit["prediction_coefs"]))
    # The number What-If and Why? deliver must say a bigger waist is worse. This gates the SIGN of
    # the What-If/Why? half only: the headline Life Clock number is ungated on both known
    # inversions (prediction waist, M10) and the magnitudes are inflated because neither fit
    # adjusts for age or sex (M11). Both are registered owner decisions, not silent fixes.
    checks.append(("attribution: waist is correctly signed (bigger waist = worse)",
                   fit["attribution_coefs"]["waist"] > 0))
    checks.append(("attribution: smoking is correctly signed",
                   fit["attribution_coefs"]["smk_current"] > 0))
    checks.append(("calibration MAE <= 0.02", g["calibration_mae"] <= 0.02))

    # centring: average Romanian ≈ national life expectancy at 40
    prev = json.load(open(os.path.join(DATA, "RO_prevalence.json")))
    ro = {"current_smoking": prev["current_smoking_pct"][0] / 100, "overweight_plus": prev["overweight_plus_pct"][0] / 100}
    rates = centring.cohort_rates(fit["cohort"])
    lt = json.load(open(os.path.join(DATA, "RO_2024.json")))
    qxM = baselines.qx_from_eurostat_lifetable(lt["PROBDEATH_M"])
    le_avg = baselines.remaining_le(qxM, 40, 1.0)     # RR=1 by construction for the average
    checks.append(("avg-RO male LE@40 ≈ national 34.5", abs(le_avg - lt["LIFEXP_M"]["Y40"]) < 0.6))

    # Bundle-content checks (LEV-01): the exported v2.2.0 artifact must carry the literature
    # standardizers, centring references, the corrected env role, and a data-vintage stamp.
    broot = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model-v2.2.0")
    if not os.path.isdir(broot):
        sys.exit("GOLDEN: PRECONDITION MISSING — artifacts/model-v2.2.0 not found. Generate it first:\n"
                 "  PYTHONPATH=src .venv/bin/python reexport_offline.py --from-version 2.1.0 --version 2.2.0")
    coefs = json.load(open(os.path.join(broot, "coefficients.json")))
    ev = json.load(open(os.path.join(broot, "evidence.json")))
    man = json.load(open(os.path.join(broot, "manifest.json")))
    for k in ("diet", "sedentary", "stress"):
        checks.append((f"standardizer ships {k}", k in coefs["standardizer"]
                       and coefs["standardizer"][k]["sd"] > 0))
        checks.append((f"literature {k} has a centring reference", "reference" in coefs["literature"][k]))
    checks.append(("alcohol reference = light", coefs["literature"]["alcohol"]["reference"].get("level") == "light"))
    checks.append(("env role is context", ev["env"]["role"] == "context"))
    checks.append(("diet/alcohol/sedentary/stress stay levers",
                   all(ev[k]["role"] == "lever" for k in ("diet", "alcohol", "sedentary", "stress"))))
    checks.append(("data_as_of is the bare data-vintage date (service parses %Y-%m-%d)",
                   man["data_as_of"] == "2019-12-31"))
    checks.append(("data_vintage_note carries the prose context", "Eurostat" in man.get("data_vintage_note", "")))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\nGOLDEN: {'PASS' if ok else 'FAIL'}  (C-index={g['c_index']}, calibration={g['calibration_mae']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
