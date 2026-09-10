"""Golden-number regression: the lifelines pipeline must reproduce the witnessed experiment results.
Run: PYTHONPATH=src .venv/bin/python tests/test_golden.py  (no network needed — uses vendored cohort)
"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from clock_model.model import cox, centring, baselines, fit as FIT
from clock_model.evaluate import ontology_gates as OG
from clock_model.evaluate import gates
from clock_model.fetch import eurostat

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    df = pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))
    fit = FIT.fit_models(df)
    g = gates.evaluate(fit)
    checks = []
    checks.append(("n == 18839", fit["n"] == 18839))
    checks.append(("deaths == 2119", fit["deaths"] == 2119))
    # REFIT-01 dropped BMI (owner decision 2026-09-09): 0.820 -> 0.805. The lost 0.015 was bought
    # by a large negative bmi coefficient fighting waist (r ~ 0.9). Band is tight because the fit is
    # deterministic (three runs agree to 10 dp); it brackets the witnessed value, and its upper
    # bound is what would catch a BMI reintroduction.
    # ONT-02 stratified the fit by age band x sex. The GLOBAL C-index drops (0.805 -> ~0.76)
    # because the model no longer ranks people using age confounding that leaked into the lever
    # coefficients. The number that matters for the product — telling apart two people of the SAME
    # age and sex — went the other way, and is asserted below rather than quoted here.
    checks.append(("C-index in [0.74, 0.78] (stratified; see note)", 0.74 <= g["c_index"] <= 0.78))
    checks.append(("bmi is not a fitted feature (REFIT-01)", "bmi" not in fit["prediction_coefs"]))
    # The number What-If and Why? deliver must say a bigger waist is worse. This gates the SIGN of
    # the What-If/Why? half only: the headline Life Clock number is ungated on both known
    # inversions (prediction waist, M10) and the magnitudes are inflated because neither fit
    # adjusts for age or sex (M11). Both are registered owner decisions, not silent fixes.
    checks.append(("total effect: waist is correctly signed (bigger waist = worse)",
                   fit["total_effect_coefs"]["waist"] > 0))
    checks.append(("total effect: smoking is correctly signed",
                   fit["total_effect_coefs"]["smk_current"] > 0))
    # The defect that shipped: smoking read as protective for the under-55s. It was age confounding,
    # not an interaction, and stratification is what fixes it.
    net_young = fit["prediction_coefs"]["smk_current"] + fit["prediction_coefs"]["smk_current_x_young"]
    checks.append((f"net smoking for the young is harmful (was -0.55)", net_young > 0))

    # Every gate the ontology implies, re-checked on this artifact.
    for name, ok, detail in OG.check(fit):
        checks.append((f"ontology gate — {name}", ok))
    # Measured WITHIN strata since ONT-02: each age x sex stratum has its own baseline, so a pooled
    # Breslow curve would score the model against one it never uses (that read 0.026; the honest
    # within-stratum figure is 0.013).
    checks.append(("calibration MAE <= 0.02 (within strata)", g["calibration_mae"] <= 0.02))

    # centring: average Romanian ≈ national life expectancy at 40
    prev = json.load(open(os.path.join(DATA, "RO_prevalence.json")))
    ro = {"current_smoking": prev["current_smoking_pct"][0] / 100, "overweight_plus": prev["overweight_plus_pct"][0] / 100}
    rates = centring.cohort_rates(fit["cohort"])
    lt = json.load(open(os.path.join(DATA, "RO_2024.json")))
    qxM = baselines.qx_from_eurostat_lifetable(lt["PROBDEATH_M"])
    le_avg = baselines.remaining_le(qxM, 40, 1.0)     # RR=1 by construction for the average
    checks.append(("avg-RO male LE@40 ≈ national 34.5", abs(le_avg - lt["LIFEXP_M"]["Y40"]) < 0.6))

    # Bundle-content checks (LEV-01): the shipped artifact must carry the literature standardizers,
    # centring references, the corrected env role, and a data-vintage stamp.
    # artifacts/ is gitignored, so every fresh checkout takes this branch — the message below is the
    # repo's only documented way to build a bundle, and it has to actually run. It named 3.0.2, which
    # this branch deleted: an instruction left a revision behind the thing it describes, which is the
    # defect the verbatim check further down exists to catch.
    BUNDLE, BUNDLE_FROM = "model-v3.0.3", "3.0.1"
    broot = os.path.join(os.path.dirname(__file__), "..", "artifacts", BUNDLE)
    if not os.path.isdir(broot):
        sys.exit(f"GOLDEN: PRECONDITION MISSING — artifacts/{BUNDLE} not found. Generate it first:\n"
                 f"  PYTHONPATH=src python3 reexport_offline.py "
                 f"--from-version {BUNDLE_FROM} --version {BUNDLE.removeprefix('model-v')}")
    coefs = json.load(open(os.path.join(broot, "coefficients.json")))
    ont_path = os.path.join(broot, "ontology.json")
    checks.append(("bundle ships the ontology", os.path.exists(ont_path)))
    bundle_ont = json.load(open(ont_path)) if os.path.exists(ont_path) else {}
    # The ontology reaches the reader by travelling VERBATIM into the bundle, so that is what gets
    # asserted — the whole file against its source, not a couple of fields.
    #
    # An earlier version of this check compared two DOI strings. It was green while the bundle was a
    # revision behind the source in four other places, because those two strings had not changed:
    # a stale verification date and the uncorrected `design` prose would both have shipped. A test
    # for "copied verbatim" has to compare the copy with the original.
    source_ont = json.load(open(os.path.join(os.path.dirname(__file__), "..", "src", "clock_model",
                                             "ontology", "ontology.json")))
    drift = sorted(k for k in set(source_ont) | set(bundle_ont)
                   if source_ont.get(k) != bundle_ont.get(k))
    checks.append(("bundle's ontology is the source ontology, verbatim"
                   + (f" (drifted: {drift})" if drift else ""), not drift))
    ie = (bundle_ont.get("cigs_day") or {}).get("intervention_evidence") or {}
    checks.append(("bundle ships cigs_day's intervention evidence",
                   {s.get("doi") for s in ie.get("sources", [])}
                   == {"10.1093/aje/kwf150", "10.1136/tc.2005.011932"}))
    # A current smoker who skips the dose question must not be scored as smoking zero a day: since
    # the contrast fix, smk_current no longer absorbs dose, so the bundle has to supply the neutral
    # value rather than leaving the service to invent one.
    cd = coefs.get("conditional_defaults", {})
    checks.append(("bundle ships a smoker-conditional dose default",
                   cd.get("cigs_day_when_current_smoker", 0) > 5))
    ev_any = json.load(open(os.path.join(broot, "evidence.json")))
    # Anything but "na" is a grade shown to the reader, so anything but "na" needs a link. Listing
    # the grades instead let `strong_for_harm` — alcohol's, and a documented value — slip the check
    # silently. A closed list of what does NOT count is the safe direction: a new grade is covered
    # the day it is invented rather than the day someone remembers to add it here.
    ungraded = ("na", None)
    checks.append(("every graded factor carries an openable link",
                   all(v.get("url") for v in ev_any.values() if v.get("grade") not in ungraded)))
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

    # The product compares people of the same age and sex, so that is the discrimination to pin.
    # A quoted figure in a comment drifts; this one is measured every run.
    from lifelines.utils import concordance_index
    import numpy as np
    lp = (fit["design"][list(fit["prediction_coefs"])] * pd.Series(fit["prediction_coefs"])).sum(axis=1)
    strata, T, E = fit["strata_labels"], fit["T"].to_numpy(), fit["E"].to_numpy()
    tot = n = 0.0
    for stratum in np.unique(strata):
        m = strata == stratum
        if E[m].sum() < 20:
            continue
        tot += concordance_index(T[m], -lp[m].to_numpy(), E[m]) * E[m].sum()
        n += E[m].sum()
    within = tot / n
    checks.append((f"within-stratum C-index >= 0.68 (measured {within:.3f})", within >= 0.68))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\nGOLDEN: {'PASS' if ok else 'FAIL'}  (C-index={g['c_index']}, calibration={g['calibration_mae']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
