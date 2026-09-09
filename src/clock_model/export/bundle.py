"""Assemble the immutable, versioned artifact bundle the Rust service loads."""
from __future__ import annotations
import hashlib
import json
import os

from clock_model.config import features as F
from clock_model.config.literature import LITERATURE
from clock_model.config.cycles import MORT_FOLLOWUP_THROUGH
from clock_model.config.countries import LIFETABLE_YEAR
from clock_model.ontology import _PATH as ONT_PATH


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)


def _checksum(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _merged_standardizer(fitted: dict) -> dict:
    lit = {k: {"mean": v["standardizer"]["mean"], "sd": v["standardizer"]["sd"]}
           for k, v in LITERATURE.items() if "standardizer" in v}
    overlap = set(fitted) & set(lit)
    if overlap:
        raise SystemExit(f"[bundle] standardizer key(s) {sorted(overlap)} are both fitted and "
                         "literature-declared — resolve which one ships before exporting")
    return {**fitted, **lit}


def assemble(out_dir: str, version: str, fit: dict, gates: dict, countries: dict) -> str:
    """countries: {ISO: {qx:{M,F}, reference_lp:{young,old}, national_le_40:{M,F}, prevalence}}."""
    root = os.path.join(out_dir, f"model-v{version}")
    # Bundles are immutable (ADR-006): a change means a NEW version, never an in-place edit.
    if os.path.exists(root):
        raise SystemExit(f"[bundle] {root} already exists — bump the version instead of overwriting")

    coefficients = {
        "prediction": fit["prediction_coefs"],
        # Total effects: one per lever, each fitted on the adjustment set the causal graph implies
        # (confounders only, never mediators), then precision-weighted against the literature prior.
        # These are what What-If and the recommendations may use; they are NOT summable with the
        # prediction coefficients, because that would double-count every mediated path.
        "total_effect": fit["total_effect_coefs"],
        "total_effect_sd": fit.get("total_effect_sd", {}),
        "total_effect_source": fit.get("total_effect_source", {}),
        # What the cohort alone said, before any literature prior was applied. Shipped so the
        # disagreement between our data and the published effect sizes stays auditable in the
        # artifact rather than living only in the fitting process (PR#2 F5).
        "total_effect_data_only": fit.get("total_effect_data_only", {}),
        # Coefficients the sign constraint pinned to their bound: the data pointed the other way and
        # the constraint refused. Declared in the ontology with a reason; shipped so a reader can see
        # which numbers are decisions rather than measurements.
        "clipped_at_bound": fit.get("clipped_at_bound", []),
        "adjustment_sets": fit.get("adjustment_sets", {}),
        "strata": fit.get("strata", {}),
        "attribution": fit.get("attribution_coefs", fit["total_effect_coefs"]),
        # Fitted standardizers + the literature features' published/declared ones, in one map, so
        # the service z-scores every continuous input the same way (and can validate coverage).
        # A key can only come from one side: if a future fit gains a column the literature also
        # declares, that's a modelling decision, not a silent override.
        "standardizer": _merged_standardizer(fit["standardizer"]),
        "literature": LITERATURE,
        "young_cutoff": F.YOUNG_CUTOFF,
    }
    _write(os.path.join(root, "coefficients.json"), coefficients)

    # The ontology travels with the model: the service reads roles, signs, grades and the article
    # links straight from it, so there is one source of truth end to end instead of four copies.
    ont_raw = json.load(open(os.path.join(os.path.dirname(os.path.abspath(ONT_PATH)), "ontology.json")))
    _write(os.path.join(root, "ontology.json"), ont_raw)

    evidence = {}
    for key, spec in ((k, v) for k, v in ont_raw.items() if not k.startswith("_")):
        prior = spec.get("prior") or {}
        evidence[key] = {
            "role": spec.get("role", "context"),
            "grade": spec.get("grade", "na"),
            "citation": prior.get("title") or spec.get("study", ""),
            "doi": prior.get("doi"),
            "url": prior.get("url") or (f"https://doi.org/{prior['doi']}" if prior.get("doi") else None),
            "first_author": prior.get("first_author"),
            "year": prior.get("year"),
            "study_slug": spec.get("study"),
        }
    # The literature block used to write its own evidence entries here, in a prose-only form with no
    # link. The ontology now covers those same factors with verified DOIs, so overwriting them would
    # reintroduce exactly the duplication (and the unlinkable citations) this work removes.
    _write(os.path.join(root, "evidence.json"), evidence)

    for iso, b in countries.items():
        _write(os.path.join(root, "baselines", f"{iso}.json"), b)

    card = f"""# Model card — The Clock of Life v{version}

**Algorithm:** Cox proportional hazards (interpretable), lifestyle + pathology predictors; age & sex to
the national life-table baseline. **Attribution/What-If** uses a separate total-effect model.

**Training:** NHANES 2007-2014 linked to NCHS mortality (through {MORT_FOLLOWUP_THROUGH}); n={fit['n']},
deaths={fit['deaths']}. **Discrimination** C-index {gates['c_index']}; **calibration** MAE
{gates['calibration_mae']}.

**Fitting (v3.0.0+):** stratified by age band x sex — each stratum has its own baseline hazard, so
coefficients compare people of the same age and sex, and age stays out of the linear predictor.
The C-index above is therefore GLOBAL and not comparable to earlier unstratified versions (v2.2.0
read 0.805): the old figure was inflated by age confounding leaking into the lever coefficients.
Discrimination between people of the same age and sex, which is what the product actually does,
improved. Sign constraints from the ontology are enforced as optimizer bounds; coefficients listed
in `coefficients.clipped_at_bound` were pinned there by a declared decision, not measured.
Per-lever TOTAL effects are fitted on the adjustment set the causal graph implies and, where the
prior is on the same scale, precision-weighted against it; `total_effect_data_only` records what
the cohort alone said.

**Baselines:** {len(countries)} countries (Eurostat life tables); relative risk centred on each country's
average person (smoking & weight from EHIS, cohort-mean fallback otherwise).

**Literature features** (diet, alcohol, sedentary, stress, environment) are appended from the evidence
base at stated confidence, not fitted on the cohort.

**Intended use:** wellness/education only — a statistical estimate, never a prediction or diagnosis
(ADR-001). Recommendations target modifiable/manageable factors only.
"""
    with open(os.path.join(root, "model-card.md"), "w") as f:
        f.write(card)

    # checksum every file in the bundle (except the manifest itself), then write the manifest last
    checksums = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name == "manifest.json":
                continue
            p = os.path.join(dirpath, name)
            checksums[os.path.relpath(p, root)] = _checksum(p)
    manifest = {
        "version": version,
        "algorithm": "cox_ph",
        "reference_population": "per-country (Eurostat)",
        "training_cohort": "NHANES 2007-2014 + NCHS Linked Mortality",
        "mortality_followup_through": MORT_FOLLOWUP_THROUGH,
        # The DATA vintage, not the build date. Kept a bare ISO date — the service parses this
        # into a DATE column (seed.rs %Y-%m-%d); the prose context goes in data_vintage_note.
        "data_as_of": MORT_FOLLOWUP_THROUGH,
        "data_vintage_note": f"mortality linkage through {MORT_FOLLOWUP_THROUGH}; "
                             f"Eurostat life tables {LIFETABLE_YEAR}",
        "n": fit["n"], "deaths": fit["deaths"],
        "gates": gates,
        "countries": sorted(countries.keys()),
        "checksums": checksums,
    }
    _write(os.path.join(root, "manifest.json"), manifest)
    return root
