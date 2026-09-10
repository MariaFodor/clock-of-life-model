"""Release gates GENERATED from the ontology, not hand-written.

Every claim the ontology makes about a factor is re-checked against the fitted artifact, and a
violation refuses the release. The point is that the gate list cannot drift from the knowledge it
is supposed to enforce: adding a factor to the ontology adds its checks automatically.
"""
from __future__ import annotations

from clock_model import ontology as O

SIGN_TOLERANCE = 1e-6  # a bound-constrained optimum sits exactly on 0.0, not slightly past it


def check(fit: dict) -> list[tuple[str, bool, str]]:
    """Return (name, passed, detail) for every gate the ontology implies."""
    ont = O.load()
    out: list[tuple[str, bool, str]] = []

    # 1. Every citation the model makes — priors, and the sources behind a lever's
    #    intervention advice — must be openable and verified. A reader cannot tell from the
    #    screen which internal field a claim came from, so neither should the gate.
    problems = O.citation_problems(ont)
    out.append(("every citation the model makes is openable and verified", not problems,
                "; ".join(problems) if problems else "all verified"))

    # 2. Declared signs must hold in both estimands. This is the class of defect that shipped an
    #    inverted BMI coefficient and a protective smoking effect for the young.
    for name, coefs in (("prediction", fit["prediction_coefs"]),
                        ("total effect", fit["total_effect_coefs"])):
        for key, value in coefs.items():
            base = key.replace("_x_young", "")
            if key.endswith("_x_young"):
                continue  # interactions are differences, not claims about direction
            sign = ont.get(base, {}).get("sign", "free")
            if sign == "positive":
                ok = value >= -SIGN_TOLERANCE
            elif sign == "negative":
                ok = value <= SIGN_TOLERANCE
            else:
                continue
            out.append((f"{name}: {key} is {sign}", ok, f"{value:+.4f}"))

    # 3. Net lever direction per age stratum — a main effect can be correctly signed while its age
    #    interaction flips the total for one group, which is exactly how M2 hid.
    pred = fit["prediction_coefs"]
    for key, value in pred.items():
        if not key.endswith("_x_young"):
            continue
        base = key[: -len("_x_young")]
        sign = ont.get(base, {}).get("sign", "free")
        if sign == "free" or base not in pred:
            continue
        net = pred[base] + value
        ok = net >= -SIGN_TOLERANCE if sign == "positive" else net <= SIGN_TOLERANCE
        out.append((f"net {base} for the young is {sign}", ok, f"{net:+.4f}"))

    # 4. A total-effect model may never condition on its target's mediators.
    for lever, adj in fit.get("adjustment_sets", {}).items():
        med = O.descendants(lever, ont)
        leak = sorted(set(adj) & med)
        out.append((f"{lever}: total effect adjusts for no mediator", not leak,
                    f"leaked: {leak}" if leak else "clean"))

    # 5. Factors whose role was decided on evidence must keep it. Reading `levers()` back and
    #    checking it contains only levers is a tautology — the real claim is that the specific
    #    demotions made in ONT-01 have not been quietly undone (PR#2 F3).
    for key, expected in (("sleep_long", "marker"), ("mobility", "marker"), ("env", "context")):
        actual = ont.get(key, {}).get("role")
        out.append((f"{key} is still classified as {expected}", actual == expected, str(actual)))

    # 6. A coefficient pinned to its bound means the data pointed the other way and the constraint
    #    refused. That is a finding, not a pass: without this the gate reads 0.0000 as "correctly
    #    signed" and the clipping vanishes from the record (PR#2 F4).
    clipped = fit.get("clipped_at_bound", [])
    unexpected = [c for c in clipped if not ont.get(c.replace("_x_young", ""), {}).get("clip_expected")]
    out.append(("every clipped coefficient is a declared decision, not a surprise", not unexpected,
                f"undeclared clips: {unexpected}" if unexpected
                else (f"declared: {clipped}" if clipped else "none clipped")))

    # 7. A total effect is only causal if the confounders the graph declares were actually
    #    available. Silently dropping the ones that are not cohort columns would present a
    #    partially-adjusted estimate as a total effect (PR#2 F9).
    for lever, adj in fit.get("adjustment_sets", {}).items():
        declared = set(ont.get(lever, {}).get("confounded_by", [])) - {"age", "sex"}
        med = O.descendants(lever, ont)
        expected = declared - med
        missing = sorted(expected - set(adj) - set(ont.get(lever, {}).get("waived_confounders", [])))
        out.append((f"{lever}: every declared confounder adjusted or waived", not missing,
                    f"unadjusted: {missing}" if missing else "complete"))

    # 8. A lever whose prior is on the binary scale must be fitted alongside its companion
    #    indicators, or the contrast it estimates is not the contrast the prior measured — the
    #    reference group silently becomes "everyone else" instead of "never" (measured: it halved
    #    the former-smoker effect).
    for lever, adj in fit.get("adjustment_sets", {}).items():
        companions = ont.get(lever, {}).get("companions", [])
        missing = [c for c in companions if c not in adj]
        out.append((f"{lever}: fitted with its companion indicators", not missing,
                    f"missing: {missing}" if missing else ("n/a" if not companions else "complete")))

    #    Companions must be declared symmetrically. Without this the gate is circular: it compares
    #    the fit against the same declaration the fit was built from, so deleting the declaration
    #    moves both together and the contrast silently reverts.
    for key, spec in ont.items():
        for companion in spec.get("companions", []):
            back = ont.get(companion, {}).get("companions", [])
            out.append((f"{key}/{companion}: companion declaration is symmetric", key in back,
                        f"{companion} does not list {key}"))

    # 9. A declared clip or waiver must carry its reason. "Declared" without a reason is just a
    #    switch that silences the gate.
    for key, spec in ont.items():
        if spec.get("clip_expected") and not spec.get("clip_reason"):
            out.append((f"{key}: declared clip states its reason", False, "clip_reason missing"))
        if spec.get("waived_confounders") and not spec.get("waiver_reason"):
            out.append((f"{key}: waived confounders state their reason", False, "waiver_reason missing"))

    # 10. Every design column must be declared, or it is both unconstrained and ungated.
    undeclared = sorted(k.replace("_x_young", "") for k in fit["prediction_coefs"]
                        if k.replace("_x_young", "") not in ont)
    out.append(("every fitted coefficient is declared in the ontology", not undeclared,
                str(undeclared) if undeclared else "ok"))
    return out


def passed(fit: dict) -> bool:
    return all(ok for _, ok, _ in check(fit))
