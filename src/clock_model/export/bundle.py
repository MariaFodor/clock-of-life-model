"""Assemble the immutable, versioned artifact bundle the Rust service loads."""
from __future__ import annotations
import hashlib
import json
import os

from clock_model.config import features as F
from clock_model.config.literature import LITERATURE
from clock_model.config.cycles import MORT_FOLLOWUP_THROUGH
from clock_model.config.countries import LIFETABLE_YEAR
from clock_model.fetch.wpp import ALIASES, LATEST_ESTIMATE_YEAR as WPP_YEAR
from clock_model.ontology import _PATH as ONT_PATH


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)


def _checksum(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _licences(env_sources: dict) -> list[dict]:
    """Every licence the bundle's contents oblige, deduplicated, with the share-alike ones flagged.

    `share_alike` is derived from the licence string rather than hand-marked: a new source added later
    with an -SA licence must light this up without anyone remembering to.
    """
    seen = {}
    for where, src in (env_sources or {}).items():
        licence = (src or {}).get("licence")
        if not licence:
            continue
        seen.setdefault(licence, {"licence": licence, "url": src.get("licence_url"),
                                  "applies_to": [], "share_alike": "-sa" in licence.lower().replace(" ", "-"),
                                  "non_commercial": "-nc" in licence.lower().replace(" ", "-")})
        seen[licence]["applies_to"].append(where)
    for v in seen.values():
        v["applies_to"].sort()
    return sorted(seen.values(), key=lambda v: v["licence"])


def _validate_places(places: list, countries: dict) -> None:
    """Refuse a places artifact that would put a fiction, or a silent zero, in front of a reader."""
    if len(places) < 2000:
        raise SystemExit(f"[bundle] only {len(places)} places — refusing a partial artifact")
    iso3s = {b.get("iso3") for b in countries.values() if b.get("iso3")}
    # The service stores these in a table with UNIQUE (iso3-derived country, name). A duplicate pair is
    # not a cosmetic flaw there: Postgres refuses the whole batched insert, so the seeder fails and the
    # previous — invented — rows stay. One pair in 3,522 did exactly that.
    seen = set()
    for rec in places:
        pair = (rec.get("iso3"), rec.get("city"))
        if pair in seen:
            raise SystemExit(f"[bundle] places has two rows for {pair} — the service keys locations on "
                             "this pair and would refuse the batch, leaving the old rows in place")
        seen.add(pair)
    for rec in places:
        where = f"{rec.get('iso3')}/{rec.get('city')}"
        if rec["iso3"] not in iso3s:
            raise SystemExit(f"[bundle] place {where} is in no country this bundle has a life table "
                             "for — it could be drawn on the map with nothing behind it")
        for field in ("city", "lat", "lon", "pm25", "pm25_year"):
            if rec.get(field) in (None, ""):
                raise SystemExit(f"[bundle] place {where} has no {field} — refusing")
        # The failure this exists to stop: a greenness value present with no word for where it came
        # from renders as a measurement of this city when it is the country's figure.
        if (rec.get("ndvi") is None) != (rec.get("ndvi_basis") is None):
            raise SystemExit(f"[bundle] place {where} has ndvi={rec.get('ndvi')} and basis="
                             f"{rec.get('ndvi_basis')!r} — a value without its provenance, or the "
                             "reverse; the screen cannot label this honestly")
        if rec.get("ndvi_basis") not in (None, "city", "country"):
            raise SystemExit(f"[bundle] place {where} has an unknown ndvi_basis "
                             f"{rec['ndvi_basis']!r}")
        if "illustrative" in str(rec.get("city", "")).lower():
            raise SystemExit(f"[bundle] place {where} is still labelled illustrative")


def _merged_standardizer(fitted: dict) -> dict:
    lit = {k: {"mean": v["standardizer"]["mean"], "sd": v["standardizer"]["sd"]}
           for k, v in LITERATURE.items() if "standardizer" in v}
    overlap = set(fitted) & set(lit)
    if overlap:
        raise SystemExit(f"[bundle] standardizer key(s) {sorted(overlap)} are both fitted and "
                         "literature-declared — resolve which one ships before exporting")
    return {**fitted, **lit}


def assemble(out_dir: str, version: str, fit: dict, gates: dict, countries: dict,
             baseline_gates: list | None = None, places: list | None = None,
             env_sources: dict | None = None) -> str:
    """countries: {ISO2: {qx:{M,F,B}, national_le_40, iso3, name, region, lifetable_year, source,
    and — for the subset that can be scored — reference_lp + prevalence}}.

    Two country lists come out of this, and keeping them apart is the point. `countries` in the
    manifest stays what it has always been: the set the service may SCORE, which the service turns
    into `/api/meta`'s country list. `reference_countries` is everything there is a life table for.
    Widening the first to match the second would make every country scoreable against a US
    cohort-mean reference person."""
    root = os.path.join(out_dir, f"model-v{version}")
    # Bundles are immutable (ADR-006): a change means a NEW version, never an in-place edit.
    if os.path.exists(root):
        raise SystemExit(f"[bundle] {root} already exists — bump the version instead of overwriting")

    scoreable = sorted(iso for iso, b in countries.items() if b.get("reference_lp"))

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
        # Neutral values for optional inputs, so the service never has to invent one. A current
        # smoker who skips the dose question is scored at the smokers' mean, not at zero.
        "conditional_defaults": fit.get("conditional_defaults", {}),
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

    # The real settlements, as ONE artifact rather than one file per country. The service seeds its
    # `location` table from this, and the picker reads it; splitting it per country would mean 85 files
    # to checksum and a partial read that looks like a small country rather than a broken bundle.
    if places is not None:
        _validate_places(places, countries)
        _write(os.path.join(root, "places.json"), places)

    places_count = f"{len(places):,}" if places else "no"
    places_countries = len({p["iso3"] for p in places}) if places else 0
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

**Baselines:** life tables for {len(countries)} countries or areas (UN World Population Prospects 2024,
{WPP_YEAR} estimates, single year of age 0-100 by sex; CC BY 3.0 IGO). Of those, **{len(scoreable)} can be
SCORED** — relative risk is centred on the country's average person using smoking and weight from
Eurostat EHIS, and only those countries have it. The rest carry a life table so the atlas can draw
them and are refused a personal estimate. Eurostat life tables ({LIFETABLE_YEAR}) are retained as the
independent cross-source witness the release gates check against, not as a source.

**Literature features** (diet, alcohol, sedentary, stress, environment) are appended from the evidence
base at stated confidence, not fitted on the cohort.

**Exposure reference (v4.1.0+):** each baseline carries `env_reference` — the measured air and greenness
an average person in that country is exposed to, which is what the environment term is CENTRED on. Air:
WHO Global Health Observatory `SDGPM25`, population-weighted and split by residence area (total / urban /
rural / city / town), so a reader in a village is not centred on a capital-city average. Greenness:
population-weighted annual mean NDVI from Stowell et al. 2023 (CC0), derived from that country's own
measured cities, with `ndvi_cities` recording how many — 22 of the 30 scoreable countries rest on ONE
city, and the figure must be labelled as that rather than presented as a measurement of the country.
`places.json` carries {places_count} real settlements in {places_countries} countries, each with its own
PM2.5 reading and `ndvi_basis` saying whether its greenness is its own or its country's.

**Inherited licence:** WHO's air data is CC BY-NC-SA 3.0 IGO — non-commercial and SHARE-ALIKE, and that
obligation attaches to this bundle and to anything distributed containing it. `manifest.licences` states
it so it travels with the artifact.

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
    sources = sorted({json.dumps(b["source"], sort_keys=True) for b in countries.values()
                      if b.get("source")})
    # The manifest below states UN WPP provenance as a literal. That is a claim about the DATA, so it
    # has to be refused when the data cannot support it: re-exporting from a v3 bundle carries
    # Eurostat baselines with no `source` at all, and would have shipped an EL-keyed, age-95 table
    # under a manifest reading "UN World Population Prospects 2024".
    sourceless = sorted(iso for iso, b in countries.items() if not b.get("source"))
    if sourceless:
        raise SystemExit(
            f"[bundle] {len(sourceless)} baselines carry no source block ({sourceless[:5]}) — refusing "
            f"to stamp this bundle with a provenance its data does not have. Rebuild with "
            f"`train.py`, or re-export from a bundle that has one.")
    manifest = {
        "version": version,
        "algorithm": "cox_ph",
        "reference_population": "per-country (UN World Population Prospects 2024)",
        "training_cohort": "NHANES 2007-2014 + NCHS Linked Mortality",
        "mortality_followup_through": MORT_FOLLOWUP_THROUGH,
        # The DATA vintage, not the build date. Kept a bare ISO date — the service parses this
        # into a DATE column (seed.rs %Y-%m-%d); the prose context goes in data_vintage_note.
        "data_as_of": MORT_FOLLOWUP_THROUGH,
        "data_vintage_note": f"mortality linkage through {MORT_FOLLOWUP_THROUGH}; "
                             f"UN WPP 2024 life tables (estimates for {WPP_YEAR}); "
                             f"Eurostat {LIFETABLE_YEAR} retained as the cross-source witness",
        "n": fit["n"], "deaths": fit["deaths"],
        "gates": gates,
        # What the baseline gates actually checked, recorded rather than printed and discarded. Without
        # this a reader cannot tell whether the cross-source witness covered sixty tables, ten, or none
        # — and "Eurostat retained as the witness" in the vintage note would be unfalsifiable prose.
        "baseline_gates": [{"name": n, "passed": ok, "detail": d}
                           for n, ok, d in (baseline_gates or [])] or None,
        # Scoreable. The service derives /api/meta from this, and an entry here is a promise that a
        # person from that country can be given a number centred on their own population.
        "countries": scoreable,
        # Everything with a life table — what the atlas may draw. A superset of the above.
        "reference_countries": sorted(countries.keys()),
        # Eurostat calls Greece EL and ISO calls it GR. Stored calculations and deployed clients still
        # say EL, so the service normalises through this rather than 400ing on a code it used to accept.
        "country_aliases": dict(sorted(ALIASES.items())),
        "sources": [json.loads(s) for s in sources],
        # Where the exposure VALUES came from, beside where the exposure RESPONSE came from. The
        # ontology could already cite Burnett 2014 for the coefficient; nothing in the artifact said
        # what PM2.5 number that coefficient was being applied to, or when it was measured.
        "env_sources": env_sources,
        # The obligation this bundle INHERITS, stated rather than discovered. WHO's air data is
        # CC BY-NC-SA 3.0 IGO: non-commercial (approved by the owner) and SHARE-ALIKE, which attaches
        # to anything distributed containing it — this bundle, and the service that vendors it. Written
        # into the manifest so the obligation travels with the artifact instead of living in a commit
        # message nobody reads at distribution time.
        "licences": _licences(env_sources) if env_sources else None,
        "places": None if places is None else {
            "count": len(places),
            "countries": len({p["iso3"] for p in places}),
            "with_city_greenness": sum(1 for p in places if p["ndvi_basis"] == "city"),
            "with_country_greenness": sum(1 for p in places if p["ndvi_basis"] == "country"),
            "without_greenness": sum(1 for p in places if p["ndvi_basis"] is None),
        },
        "checksums": checksums,
    }
    _write(os.path.join(root, "manifest.json"), manifest)
    return root
