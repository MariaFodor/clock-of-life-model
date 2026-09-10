"""W-A2 — the UN WPP adapter reads what it claims to read.

Two halves. The first is pure and instant: the refusals in `_require_estimate_year`, `_keep` and
`_validate`, driven with synthetic rows, because those are the guards that decide whether a wrong
number can enter looking right. The second runs against the real 237-country tables and checks them
against WPP's OWN published life expectancy — the strongest gate available, since it proves the adapter
read the right file, year, sex and age column rather than merely something plausible.

First run downloads ~600 MB (three per-sex life-table files) and caches the filtered extract under
data/cache/; later runs read the cache and take seconds.

Run: PYTHONPATH=src .venv/bin/python tests/test_wpp_adapter.py
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clock_model.fetch import wpp                      # noqa: E402
from clock_model.model.baselines import remaining_le   # noqa: E402

HEAD = ["ISO2_code", "ISO3_code", "LocTypeName", "Time", "Variant", "AgeGrpStart", "AgeGrpSpan",
        "qx", "ex", "ax", "Sex"]
IDX = {name: pos for pos, name in enumerate(HEAD)}


def row(iso2="RO", loctype="Country/Area", time="2023", variant="Medium", age="0", span="1",
        qx="0.005", sex="Male"):
    return [iso2, "ROU", loctype, time, variant, age, span, qx, "72.4", "0.5", sex]


def refuses(fn, *args) -> bool:
    """Narrow to ValueError on purpose: catching bare Exception would report PASS against a validator
    that raised TypeError on everything, which is a gate that has stopped reading its input."""
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def main():
    checks = []

    # ── the one guard that actually keeps forecasts out ────────────────────────
    # `Variant == "Medium"` reads like a projection guard and is not one: Medium IS the name of WPP's
    # central projection scenario, and the indicators file carries 78 post-2023 forecast years all
    # labelled Medium. The year is the only thing separating an estimate from a forecast.
    checks.append(("the year after the last estimate is refused",
                   refuses(wpp._require_estimate_year, wpp.LATEST_ESTIMATE_YEAR + 1)))
    checks.append(("2024 — the year the model's own config asks for — is refused",
                   refuses(wpp._require_estimate_year, 2024)))
    checks.append(("the last estimate year is accepted",
                   wpp._require_estimate_year(wpp.LATEST_ESTIMATE_YEAR) == wpp.LATEST_ESTIMATE_YEAR))

    # ── the refusals in _keep ──────────────────────────────────────────────────
    checks.append(("_keep accepts a plain country row", wpp._keep(IDX, row(), 2023)))
    checks.append(("_keep refuses another year", not wpp._keep(IDX, row(time="2022"), 2023)))
    # The files open with regional aggregates ("ADB region: Central and West Asia") carrying no ISO
    # code — a baseline built from one of those is nonsense that still validates.
    checks.append(("_keep refuses an aggregate region",
                   not wpp._keep(IDX, row(iso2="", loctype="Region"), 2023)))
    checks.append(("_keep refuses a country row with no ISO2",
                   not wpp._keep(IDX, row(iso2=""), 2023)))
    # Trusting the filename's word "Complete" for single-year-of-age is the same mistake the Sex check
    # exists to avoid: an abridged row would overwrite a single-year one and pass every later check.
    checks.append(("_keep refuses an abridged five-year row",
                   not wpp._keep(IDX, row(span="5"), 2023)))

    # ── the refusals in _validate ──────────────────────────────────────────────
    full = {a: (1.0 if a == wpp.MAX_AGE else 0.005) for a in range(0, wpp.MAX_AGE + 1)}
    world = {f"C{i:02d}": {s: dict(full) for s in ("M", "F", "B")} for i in range(wpp.MIN_COUNTRIES)}
    checks.append(("_validate accepts a complete world", not refuses(wpp._validate, world, 2023)))
    checks.append(("_validate refuses an empty download", refuses(wpp._validate, {}, 2023)))
    # A partial upstream publish, consistent across all three files, would otherwise yield a
    # structurally perfect 50-country atlas — and coverage is the whole reason for this source.
    checks.append(("_validate refuses a world too small to be the world",
                   refuses(wpp._validate, {"RO": {s: dict(full) for s in ("M", "F", "B")}}, 2023)))
    missing_sex = {**world, "C00": {"M": dict(full), "F": dict(full)}}
    checks.append(("_validate refuses a missing sex", refuses(wpp._validate, missing_sex, 2023)))
    gappy = {**world, "C00": {s: {a: q for a, q in full.items() if a != 40} for s in ("M", "F", "B")}}
    checks.append(("_validate refuses a hole in the age grid", refuses(wpp._validate, gappy, 2023)))
    oob = {**world, "C00": {s: {**full, 40: 1.4} for s in ("M", "F", "B")}}
    checks.append(("_validate refuses qx outside [0,1]", refuses(wpp._validate, oob, 2023)))
    open_end = {**world, "C00": {s: {**full, wpp.MAX_AGE: 0.4} for s in ("M", "F", "B")}}
    checks.append(("_validate refuses a table that never closes", refuses(wpp._validate, open_end, 2023)))

    # ── against the real data, checked against WPP's own published figures ─────
    tables = wpp.fetch_lifetables()
    published = wpp.fetch_published()
    places = wpp.fetch_locations()

    checks.append((f"covers exactly 237 countries or areas (got {len(tables)})", len(tables) == 237))
    checks.append(("every table has a place, and every place has a table",
                   set(tables) == set(places)))
    checks.append(("Greece is GR upstream, and EL resolves to it everywhere",
                   "GR" in tables and "EL" not in tables
                   and wpp.fetch_lifetable("EL") == tables["GR"]
                   and wpp.location("EL")["iso3"] == "GRC"))

    raw_worst = worst = worst_am = 0.0
    worst_at = worst_am_at = None
    residuals, compared = [], 0
    bracket_fail, closed_early, tail_fail = [], [], []
    for iso, sexes in tables.items():
        e0 = {}
        for sex, qx in sexes.items():
            e0[sex] = remaining_le(qx, 0, 1.0)
            pub = published.get(iso, {}).get(sex, {})
            if pub.get("ex0") is not None and pub.get("ax_last") is not None:
                compared += 1
                # Correcting the ONE thing that separates this integrator from the publisher's. At age
                # 100 qx is 1.0, so `remaining_le` prices the whole open interval at half a year
                # (`le += S * (1 - qa/2)`), while WPP prices it at its own ax ~ 2.8. That single
                # constant explains 91% of the squared residual — NOT the infant-ax effect an earlier
                # version of this comment blamed, which is worth 0.0002-0.002 yr, two to three orders
                # of magnitude too small. The signature gave it away: of the 20 worst tables, 15 were
                # female and 0 male, and the worst six are the six highest survivorships to age 100.
                survivors = 1.0
                for age in range(0, wpp.MAX_AGE):
                    survivors *= 1.0 - qx[age]
                corrected = e0[sex] + (pub["ax_last"] - 0.5) * survivors
                residuals.append(d := abs(corrected - pub["ex0"]))
                raw_worst = max(raw_worst, abs(e0[sex] - pub["ex0"]))
                if d > worst:
                    worst, worst_at = d, f"{iso}/{sex}"
            # 15-60 mortality per 1,000, derived from qx alone rather than read from a published
            # column: the atlas will serve a derived number, so it is the derivation that has to be
            # right, and the published column is what proves it.
            surv = 1.0
            for age in range(15, 60):
                surv *= 1.0 - qx[age]
            p = (places.get(iso, {}).get("published", {}).get("am") or {}).get(sex)
            if p is not None and (d := abs(1000.0 * (1.0 - surv) - p)) > worst_am:
                worst_am, worst_am_at = d, f"{iso}/{sex}"
            # Eurostat's tables close at 95 with qx = 1.0, which prices every 95-year-old in every
            # country at exactly half a year. The invariant is "nothing closes before the last age",
            # not "age 95 specifically" — a table closing at 97 would fail the promise and pass a
            # check that only looked at 95.
            if any(qx[a] >= 1.0 for a in range(wpp.MAX_AGE)):
                closed_early.append(f"{iso}/{sex}")
            if remaining_le(qx, 95, 1.0) <= 1.0:
                tail_fail.append(f"{iso}/{sex}")
        # NOT a swapped-sex detector, despite the obvious reading: min/max are symmetric, so swapping
        # M and F leaves this expression unchanged. (A swap is caught by the e0 gate instead — it
        # would show up as the whole sex gap, ~5 years.) What this buys is that B is a real both-sex
        # table rather than independently wrong.
        if not min(e0["M"], e0["F"]) <= e0["B"] <= max(e0["M"], e0["F"]):
            bracket_fail.append(iso)

    # Every comparison above is opt-in (`if published is not None`), which is correct — but it means a
    # half-empty witness silently shrinks the gate instead of failing it, and the run still reports a
    # healthy median over whatever survived. So count them.
    checks.append((f"every table was actually compared ({compared} of {3 * len(tables)})",
                   compared == 3 * len(tables)))
    median_r = statistics.median(residuals)
    checks.append((f"e0 reproduces WPP's own published ex to 0.02 yr once the open interval is priced "
                   f"the same way (median {median_r:.4f}, worst {worst:.4f} at {worst_at}; "
                   f"uncorrected worst was {raw_worst:.3f})", worst <= 0.02))
    checks.append((f"derived 45q15 matches WPP's published Q1560 within 0.5 per 1,000 "
                   f"(worst {worst_am:.3f} at {worst_am_at})", worst_am <= 0.5))
    checks.append((f"both-sex e0 sits between the sexes everywhere ({len(bracket_fail)} failures)",
                   not bracket_fail))
    checks.append((f"no table closes before {wpp.MAX_AGE}, so no country prices its 95-year-olds at "
                   f"the closeout ({len(closed_early)} failures)", not closed_early))
    checks.append((f"remaining years at 95 exceeds 1.0 everywhere ({len(tail_fail)} failures)",
                   not tail_fail))

    meta = wpp.source_metadata()
    checks.append(("provenance records a verified licence, its source page, and per-file fingerprints",
                   meta["licence"] == "CC BY 3.0 IGO" and meta["licence_read_from"]
                   and len(meta["files"]) >= 4
                   and all(f.get("sha256") and f.get("retrieved") for f in meta["files"].values())))

    ok = all(p for _, p in checks)
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nWPP ADAPTER: {'PASS' if ok else 'FAIL'}  ({len(tables)} countries, {wpp.LATEST_ESTIMATE_YEAR})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
