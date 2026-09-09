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

    # 1. Every literature prior must carry a verified, openable citation.
    problems = O.citation_problems(ont)
    out.append(("every literature prior carries a verified DOI/URL", not problems,
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

    # 5. Markers and context must never be advertised as recommendable.
    levers = set(O.levers(ont))
    bad = [k for k in levers if ont[k].get("role") != "lever"]
    out.append(("only levers are recommendable", not bad, str(bad) if bad else "ok"))
    return out


def passed(fit: dict) -> bool:
    return all(ok for _, ok, _ in check(fit))
