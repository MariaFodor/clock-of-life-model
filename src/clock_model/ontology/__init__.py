"""The ontology: declarative knowledge that constrains the fit, instead of documenting it afterwards.

`ontology.json` is the single source of truth for roles, sign/shape constraints, literature priors
(each with a Crossref-verified DOI) and the causal graph. This module loads it and derives the two
things the fitting layer needs:

  * `adjustment_set(target)` — which covariates a TOTAL-EFFECT estimate for `target` may condition
    on. Mediators (anything downstream of the target in the causal graph) are excluded by
    construction, which is what stops the model from reporting that a larger waist is protective.
  * `bounds(columns)` — sign constraints as optimizer bounds, making a wrong-signed lever impossible
    rather than merely detectable after the fact.
"""
from __future__ import annotations
import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontology.json")


def load() -> dict:
    """The ontology as {factor: spec}, with the leading `_meta`/`_comment` keys removed."""
    raw = json.load(open(_PATH))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def meta() -> dict:
    return json.load(open(_PATH))["_meta"]


ONT = load()


def descendants(target: str, ont: dict | None = None) -> set[str]:
    """Everything causally downstream of `target` — its mediators, transitively.

    A mediator of a mediator is still a mediator: conditioning on it blocks part of the causal path
    just the same, so the search has to be transitive rather than one hop deep.
    """
    ont = ont or ONT
    seen: set[str] = set()
    stack = list(ont.get(target, {}).get("causes", []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(ont.get(node, {}).get("causes", []))
    return seen


def adjustment_set(target: str, available: list[str], ont: dict | None = None) -> list[str]:
    """Covariates a TOTAL-EFFECT estimate for `target` may condition on.

    Declared confounders, minus anything downstream of the target. `age`/`sex` are excluded because
    they are handled by the strata, not by coefficients.
    """
    ont = ont or ONT
    med = descendants(target, ont)
    conf = set(ont.get(target, {}).get("confounded_by", []))
    # Companion indicators of the same categorical exposure are not confounders — they are part of
    # the same variable. Leaving one out silently redefines the reference group (former vs "everyone
    # else" instead of former vs never), which then gets blended with a prior measured against a
    # different contrast.
    conf |= set(ont.get(target, {}).get("companions", []))
    return [c for c in available
            if c != target and c in conf and c not in med and c not in ("age", "sex")]


def bounds(columns: list[str], ont: dict | None = None) -> list[tuple]:
    """Sign constraints as (low, high) optimizer bounds, one per design column.

    Age-interaction columns (`*_x_young`) are deliberately unconstrained: they are differences from
    the main effect, so their sign carries no evidential claim of its own.
    """
    ont = ont or ONT
    out = []
    for c in columns:
        if c.endswith("_x_young"):
            out.append((None, None))
            continue
        sign = ont.get(c, {}).get("sign", "free")
        out.append({"positive": (0.0, None), "negative": (None, 0.0)}.get(sign, (None, None)))
    return out


def levers(ont: dict | None = None) -> list[str]:
    """Factors the product may recommend changing. Markers and context are explained, never advised."""
    return [k for k, v in (ont or ONT).items() if v.get("role") == "lever"]


def citation_problems(ont: dict | None = None) -> list[str]:
    """Every literature prior must carry a verified DOI — an unopenable citation is not a citation."""
    problems = []
    for key, spec in (ont or ONT).items():
        for field in ("prior", "prior_secondary"):
            prior = spec.get(field)
            if prior is None:
                continue
            if not prior.get("doi") and not prior.get("url"):
                problems.append(f"{key}.{field}: no doi/url")
            elif not prior.get("verified"):
                problems.append(f"{key}.{field}: doi present but never verified")
        # Evidence that qualifies an INTERVENTION rather than the coefficient itself — the sources
        # behind "cutting down is not quitting", say. It is held to the same bar: this is the
        # evidence a user reads on screen, and it lived in the service's source code until the
        # model claimed it. Something the model does not declare cannot be cited as the model's.
        for i, src in enumerate(spec.get("intervention_evidence", {}).get("sources", [])):
            where = f"{key}.intervention_evidence.sources[{i}]"
            if not src.get("doi") and not src.get("url"):
                problems.append(f"{where}: no doi/url")
            elif not src.get("verified"):
                problems.append(f"{where}: doi present but never verified")
            if not src.get("supports"):
                problems.append(f"{where}: does not say WHICH claim it supports")
    return problems
