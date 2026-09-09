"""Literature-sourced feature coefficients — appended, not fitted.

These features are not in the NHANES cohort (no item-level batteries / no environment), so their effects
come from the evidence (clock_dev/analysed_papers/, RES-02/03/04), expressed on the same log-hazard scale
as the fitted coefficients. Magnitudes are deliberately conservative starting values; they are flagged so
the model card and the UI can show them at their stated confidence.

Encoding notes:
- Continuous literature features are expressed per +1 standardized unit (the Rust service standardizes the
  user's index the same way), so they compose with the fitted linear predictor.
- alcohol is categorical & MONOTONIC (no protective dip — RES-02): the BETA anchor is none=0.0
  log-HR; the CENTRING reference (what the average person is assumed to be) is separate — see the
  "reference" entry on each feature.
- ENV is computed by the service from PM2.5 / greenspace (RES-04) — the coefficient is 1.0 because the ENV
  term already carries ln(HR) inside it; see note.
"""

LITERATURE = {
    "diet": {
        "beta": -0.15, "grade": "strong", "kind": "continuous_z",
        "citation": "Trichopoulou 2003 (Mediterranean diet score); PREDIMED",
        "note": "per +1 SD of Mediterranean-style diet score; higher = lower risk",
        # Our score is a 0-5 sum of five binary items (THE_QUESTIONNAIRE.md S6). DECLARED ASSUMPTION:
        # items ~ p=0.5 -> binomial mean 2.5, sd sqrt(5*0.25)=1.12. Reference = mean (z=0 for the
        # average person), pending real RO distribution (MH-PREVALENCE).
        "standardizer": {"mean": 2.5, "sd": 1.12, "source": "assumed binomial(5, 0.5) on the 0-5 item sum"},
        "reference": {"kind": "mean"},
    },
    "sedentary": {
        "beta": 0.08, "grade": "moderate", "kind": "continuous_z",
        "citation": "Chau 2013 sitting-time meta-analysis",
        "note": "per +1 SD sitting hours; modelled jointly with activity (offset is real)",
        # DECLARED ASSUMPTION: adult daily sitting ~ 6.0 h mean, 2.5 h sd (Chau 2013 dose-response
        # range; matches NHANES-era sitting-time distributions). Reference = mean (MH-PREVALENCE).
        "standardizer": {"mean": 6.0, "sd": 2.5, "source": "assumed from Chau 2013 dose-response range"},
        "reference": {"kind": "mean"},
    },
    "stress": {
        "beta": 0.05, "grade": "weak", "kind": "continuous_z",
        "citation": "Cohen 1983 PSS; perceived-stress mortality cohorts",
        "note": "per +1 SD perceived stress; low confidence",
        # PSS-4 population norms: Warttig et al. 2013 (n=1,568 English adults): mean 6.11, sd 3.14.
        "standardizer": {"mean": 6.11, "sd": 3.14, "source": "Warttig 2013 PSS-4 population norms"},
        "reference": {"kind": "mean"},
    },
    "alcohol": {
        "kind": "categorical_monotonic", "grade": "strong_for_harm",
        "citation": "GBD 2018/2020; Mendelian randomization (no safe level)",
        "levels": {"none": 0.0, "light": 0.03, "moderate": 0.12, "heavy": 0.30},
        "note": "monotonic non-decreasing; never protective (J-curve refuted)",
        # Centring reference = "light" (the modal adult category). TRADE (principle 20): until the
        # EHIS-weighted RO distribution lands (MH-PREVALENCE) the anchoring error is bounded by
        # |levels - 0.03| <= 0.27 log-HR for heavy drinkers and 0.03 for abstainers.
        "reference": {"kind": "level", "level": "light"},
    },
    "env": {
        "kind": "precomputed", "beta": 1.0, "grade": "moderate",
        "citation": "WHO AQG (PM2.5 RR 1.095/10µg); Rojas-Rueda 2019 (greenspace)",
        "note": "service computes ENV = ln(1.095)*(PM25-ref)/10 + ln(0.965)*(NDVI-ref)/0.1; coef 1.0",
    },
}
