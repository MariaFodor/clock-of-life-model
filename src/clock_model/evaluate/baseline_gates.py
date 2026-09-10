"""Release gates for the per-country life tables.

The ontology gates check what the model CLAIMS about its factors. These check what its baselines ARE —
and they exist because a life table cannot fail loudly. A misread column, a shifted age grid, a swapped
sex, a truncated download: none of them raise. They produce a plausible number for a real person, which
is the failure this product is least able to notice afterwards.

So every gate here compares against something that was not produced by the same code path: WPP's own
published life expectancy, Eurostat's independent measurement of the same countries, or an arithmetic
identity the table must satisfy whatever it contains.
"""
from __future__ import annotations

import statistics

from clock_model.fetch import wpp
from clock_model.model.baselines import remaining_le

#: The adapter reproduces WPP's published `ex` to 0.018 yr once the open interval is priced the way WPP
#: prices it (measured over all 711 tables; median 0.004). At 0.05 this catches a one-year time slip.
MAX_PUBLISHED_GAP = 0.05
#: Eurostat vs WPP across the 30 countries we already ship: median 0.27, max 1.73 (W-A1). The bands are
#: set above the measurement, not at it — this gate is looking for a different SOURCE having been read,
#: not for the two agencies to agree exactly.
MAX_WITNESS_MEDIAN = 0.5
MAX_WITNESS_WORST = 2.0
MIN_WITNESS_RANK_CORR = 0.85
#: A world that stopped being the world.
MIN_COUNTRIES = 230
#: The scoreable set is 30 hand-maintained European countries. Gating it at "non-empty" let
#: `--countries RO` produce a fully green bundle that drops 29 countries out of /api/meta.
MIN_SCOREABLE = 25
#: The countries known to have no EHIS prevalence. An ALLOWLIST, not a predicate: the previous gate
#: asked whether the fallback was declared, and `train.py` declares it in the same branch that
#: creates it — so the gate was tautological with its only producer and could not fail. An EHIS
#: outage would have shipped all 30 European countries centred on a fabricated reference person with
#: every gate green.
EXPECTED_FALLBACKS = {"CH"}
#: 30 countries x 2 sexes. A partial Eurostat outage leaving 10 pairs used to pass, reporting a
#: healthy median over a sixth of the intended coverage.
MIN_WITNESS_PAIRS = 48


def _e(qx: dict, age: int) -> float:
    return remaining_le({int(a): q for a, q in qx.items()}, age, 1.0)


def check(baselines: dict, *, witness: dict | None = None) -> list[tuple[str, bool, str]]:
    """`(name, passed, detail)` per gate.

    `baselines` is what would be written to the bundle. `witness` is `{iso2: {'M'|'F': {age: qx}}}` from
    an INDEPENDENT source (Eurostat) for whatever subset it covers; when absent, G2 reports as skipped
    rather than passing, because a gate that quietly stops gating is worse than one that fails.
    """
    out: list[tuple[str, bool, str]] = []
    published = wpp.fetch_published()

    # ── G7 coverage ───────────────────────────────────────────────────────────
    out.append((f"covers at least {MIN_COUNTRIES} countries or areas",
                len(baselines) >= MIN_COUNTRIES, f"{len(baselines)} baselines"))

    # ── G1 the integrator against the publisher ───────────────────────────────
    # The strongest check available: it proves the adapter read the right file, year, sex and age
    # column, because a misread lands years out and this band is hundredths of a year. The one
    # correction is the open interval — `remaining_le` prices 100+ at half a year, WPP at its own ax.
    gaps = []
    for iso, b in baselines.items():
        for sex, qx in b["qx"].items():
            pub = published.get(iso, {}).get(sex, {})
            if pub.get("ex0") is None or pub.get("ax_last") is None:
                gaps.append((99.0, f"{iso}/{sex} has no published figure to check against"))
                continue
            survivors = 1.0
            for age in range(0, wpp.MAX_AGE):
                survivors *= 1.0 - qx[str(age)]
            corrected = _e(qx, 0) + (pub["ax_last"] - 0.5) * survivors
            gaps.append((abs(corrected - pub["ex0"]), f"{iso}/{sex}"))
    worst, worst_at = max(gaps) if gaps else (0.0, "-")
    out.append(("every life table reproduces its publisher's own life expectancy",
                bool(gaps) and worst <= MAX_PUBLISHED_GAP,
                f"worst {worst:.4f} yr at {worst_at}, median {statistics.median(g for g, _ in gaps):.4f}"))

    # ── G3 the both-sex table is a both-sex table ─────────────────────────────
    # Not a swapped-sex detector — min/max are symmetric, so a swap leaves this unchanged (G1 catches
    # that, as the whole sex gap). What it buys is that B was not filled from somewhere else entirely.
    unbracketed = [iso for iso, b in baselines.items()
                   if "B" in b["qx"] and not (min(_e(b["qx"]["M"], 0), _e(b["qx"]["F"], 0))
                                              <= _e(b["qx"]["B"], 0)
                                              <= max(_e(b["qx"]["M"], 0), _e(b["qx"]["F"], 0)))]
    out.append(("both-sex life expectancy falls between the sexes", not unbracketed,
                f"{len(unbracketed)} outside: {unbracketed[:5]}"))

    # ── G4 nothing closes early ───────────────────────────────────────────────
    # The defect this whole switch inherits from: Eurostat's tables end at 95 with qx = 1.0, so
    # `remaining_le` returned exactly 0.50 years for every 95-year-old in every country — arithmetic,
    # not a claim about old age. The property that fixes it is structural.
    early = [f"{iso}/{sex}" for iso, b in baselines.items() for sex, qx in b["qx"].items()
             if any(float(qx[str(a)]) >= 1.0 for a in range(0, wpp.MAX_AGE))]
    out.append((f"no table closes before age {wpp.MAX_AGE}", not early,
                f"{len(early)} close early: {early[:5]}"))
    thin = [f"{iso}/{sex}" for iso, b in baselines.items() for sex, qx in b["qx"].items()
            if _e(qx, 95) <= 1.0]
    out.append(("a 95-year-old is given more than one year anywhere", not thin,
                f"{len(thin)} under a year: {thin[:5]}"))

    # ── G6 scoreable means centred ────────────────────────────────────────────
    # `/api/meta` derives the country list from whatever has a baseline. If a country without
    # prevalence carried a reference_lp, it would be scored against a US cohort-mean reference person
    # and the reader would get a confident, wrong, PERSONAL number — the worst outcome available here.
    # A country may fall back to the cohort mean — Switzerland does today, because EHIS publishes
    # nothing usable for it — but it must SAY so. An undeclared fallback is a reference person from
    # another continent wearing this country's name.
    fallbacks = {iso for iso, b in baselines.items()
                 if b.get("reference_lp") is not None and not b.get("prevalence")}
    out.append(("only the countries known to lack prevalence fall back to a fabricated reference",
                fallbacks <= EXPECTED_FALLBACKS,
                f"expected {sorted(EXPECTED_FALLBACKS)}, got {sorted(fallbacks)}"))
    # A partial dict is truthy, so a country whose smoking dataset dropped out while its weight one
    # held would be labelled "Eurostat EHIS" while being centred on a non-smoker.
    partial = sorted(iso for iso, b in baselines.items()
                     if b.get("prevalence") and
                     not {"current_smoking", "overweight_plus"} <= set(b["prevalence"]))
    out.append(("no country is centred on half a prevalence record", not partial, f"{partial[:5]}"))
    # The structural tell for a degenerate reference person: with no prevalence, every _x_young
    # interaction is zero, so the young and old centrings collapse to the same number. It is not
    # derived from the same predicate the declaration is, which is the point.
    collapsed = sorted(iso for iso, b in baselines.items()
                       if (lp := b.get("reference_lp")) and lp["young"] == lp["old"])
    out.append(("no country's young and old centrings have collapsed together",
                set(collapsed) <= EXPECTED_FALLBACKS, f"collapsed: {collapsed}"))
    scoreable = [iso for iso, b in baselines.items() if b.get("reference_lp") is not None]
    out.append((f"at least {MIN_SCOREABLE} countries can actually be scored",
                len(scoreable) >= MIN_SCOREABLE, f"{len(scoreable)} scoreable of {len(baselines)}"))

    # ── G5 aliases resolve ────────────────────────────────────────────────────
    # Against the SCOREABLE set, not merely the drawable one: the alias exists so a stored `EL`
    # still gets a number. If GR were reference-only the alias would resolve, the gate would pass,
    # and every Greek user would get a 400 with the release green.
    scoreable_set = {iso for iso, b in baselines.items() if b.get("reference_lp") is not None}
    bad_alias = [f"{k}->{v}" for k, v in wpp.ALIASES.items()
                 if v not in scoreable_set or k in baselines]
    out.append(("every country alias resolves to a country that can be scored",
                not bad_alias, f"{bad_alias}"))

    # ── G2 the independent witness ────────────────────────────────────────────
    if not witness:
        out.append(("an independent source agrees within the declared band", False,
                    "no witness supplied — refusing to record a cross-check that did not happen"))
    else:
        deltas, pairs = [], []
        for geo, tables in witness.items():
            iso = wpp.resolve(geo)
            b = baselines.get(iso)
            if not b:
                continue
            for sex in ("M", "F"):
                if sex not in tables or sex not in b["qx"]:
                    continue
                ours, theirs = _e(b["qx"][sex], 0), remaining_le(tables[sex], 0, 1.0)
                deltas.append((abs(ours - theirs), f"{iso}/{sex}"))
                pairs.append((ours, theirs))
        if len(pairs) < MIN_WITNESS_PAIRS:
            out.append(("an independent source agrees within the declared band", False,
                        f"witness covered only {len(pairs)} of the {MIN_WITNESS_PAIRS} tables "
                        f"required — a cross-check over a sixth of the countries is not one"))
        else:
            med = statistics.median(d for d, _ in deltas)
            worst_d, worst_iso = max(deltas)
            # Rank agreement without scipy: Pearson on ranks is Spearman when there are no ties.
            rank = lambda xs: {v: i for i, v in enumerate(sorted(xs))}          # noqa: E731
            ra, rb = rank([a for a, _ in pairs]), rank([b for _, b in pairs])
            xs = [ra[a] for a, _ in pairs]
            ys = [rb[b] for _, b in pairs]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
            corr = num / den if den else 0.0
            ok = (med <= MAX_WITNESS_MEDIAN and worst_d <= MAX_WITNESS_WORST
                  and corr >= MIN_WITNESS_RANK_CORR)
            out.append(("an independent source agrees within the declared band", ok,
                        f"{len(pairs)} tables: median {med:.2f} yr, worst {worst_d:.2f} at "
                        f"{worst_iso}, rank corr {corr:.3f}"))
    return out
