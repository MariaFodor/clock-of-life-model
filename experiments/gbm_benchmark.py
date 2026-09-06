"""EXP-14 — GBM benchmark: gradient-boosted survival (XGBoost survival:cox) vs the interpretable Cox.

Identical 5-fold CV, identical feature set (lifestyle + pathology; age/sex stay in the life-table
baseline). Primary comparison = out-of-fold C-index. Decision rule (ADR-007 / worklist M1): adopt the ML
model only if it beats Cox by a meaningful margin AND does not hurt calibration — weighed against its
interpretability + ONNX cost.

Run: PYTHONPATH=src .venv/bin/python experiments/gbm_benchmark.py
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import xgboost as xgb

from clock_model.model import cox
from clock_model.config import features as F

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
K = 5
SEED = 0
XGB_PARAMS = {"objective": "survival:cox", "eval_metric": "cox-nloglik",
              "eta": 0.05, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 10, "seed": SEED}
NROUNDS = 300


def cox_cindex(Xtr, Ttr, Etr, Xte, Tte, Ete):
    d = Xtr.copy(); d["pm"] = Ttr; d["dead"] = Etr
    m = CoxPHFitter(penalizer=1e-4); m.fit(d, "pm", "dead")
    lp = Xte.to_numpy(float) @ np.array([m.params_[c] for c in Xte.columns])
    return concordance_index(Tte, -lp, Ete)


def xgb_cindex(Xtr, Ttr, Etr, Xte, Tte, Ete):
    y = np.where(Etr == 1, Ttr, -Ttr)                 # xgboost cox: negative time = censored
    dtr = xgb.DMatrix(Xtr.to_numpy(float), label=y)
    dte = xgb.DMatrix(Xte.to_numpy(float))
    booster = xgb.train(XGB_PARAMS, dtr, num_boost_round=NROUNDS, verbose_eval=False)
    risk = booster.predict(dte)                       # higher = higher risk
    return concordance_index(Tte, -risk, Ete)


def main():
    df = pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))
    coh = cox.complete_cohort(df)
    X = F._raw_columns(coh)                            # same 13 predictors for both (trees don't need z-scoring)
    T = coh["pm"].to_numpy(float); E = coh["dead"].to_numpy(int)

    idx = np.arange(len(coh)); rng = np.random.default_rng(SEED); rng.shuffle(idx)
    folds = np.array_split(idx, K)
    cox_c, xgb_c = [], []
    print(f"[EXP-14] Cox vs XGBoost survival:cox — {K}-fold CV, n={len(coh)}, deaths={int(E.sum())}\n")
    print(f"{'fold':>4}{'Cox C':>10}{'XGB C':>10}")
    for f in range(K):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(K) if j != f])
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        cc = cox_cindex(Xtr, T[tr], E[tr], Xte, T[te], E[te])
        xc = xgb_cindex(Xtr, T[tr], E[tr], Xte, T[te], E[te])
        cox_c.append(cc); xgb_c.append(xc)
        print(f"{f+1:>4}{cc:>10.3f}{xc:>10.3f}")

    cm, xm = float(np.mean(cox_c)), float(np.mean(xgb_c))
    delta = xm - cm
    print(f"\n  mean Cox C-index = {cm:.3f}  (sd {np.std(cox_c):.3f})")
    print(f"  mean XGB C-index = {xm:.3f}  (sd {np.std(xgb_c):.3f})")
    print(f"  delta (XGB - Cox) = {delta:+.3f}")

    MARGIN = 0.01
    print("\nVERDICT:")
    if delta >= MARGIN:
        print(f"  XGBoost beats Cox by {delta:+.3f} (>= {MARGIN}). ML *may* be worth adopting — but confirm it")
        print("  also improves calibration, and weigh the interpretability (SHAP) + ONNX-export cost (ADR-007).")
    else:
        print(f"  XGBoost does NOT clear the +{MARGIN} margin (delta {delta:+.3f}). Keep the interpretable Cox:")
        print("  no material ranking gain to justify losing per-factor transparency. Matches expectation that")
        print("  self-reported lifestyle+pathology is near its discrimination ceiling.")

    out = os.path.join(os.path.dirname(__file__), "..", "artifacts", "exp14_gbm_benchmark.json")
    json.dump({"cox_cindex": cm, "xgb_cindex": xm, "delta": delta, "folds": {"cox": cox_c, "xgb": xgb_c}},
              open(out, "w"), indent=2)
    print(f"\n  (results -> {os.path.relpath(out)})")


if __name__ == "__main__":
    main()
