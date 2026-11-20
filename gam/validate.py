"""Model choices made on a VALIDATION year (2024-09-01 .. 2025-08-31), never on the test year.

1. Smoothing coefficient alpha of the temperature (exponential moving average).
2. Long-term term: 12-month level blocks vs a smooth trend s(t).
Writes ../docs/data/validation.json (the page reads its numbers from there).
"""
import json
from pathlib import Path

import numpy as np
from pygam import f, s

import pandas as pd

from features import FEATS, add_features, err_correct, fit, load, mape, predict_d1

HERE = Path(__file__).parent
VAL_START, VAL_END = "2024-09-01", "2025-08-31"
raw = load(HERE / "data" / "daily.csv")

res = {"period": [VAL_START, VAL_END], "alpha": {}, "trend": {}}
for a in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]:
    d = add_features(raw, VAL_START, a) if a else add_features(raw, VAL_START, 1.0)
    if not a:
        d["temp_s"] = d["temp"]
    tr, va = d[d.index < VAL_START], d[(d.index >= VAL_START) & (d.index <= VAL_END)]
    g = fit(tr)
    res["alpha"][str(a)] = round(mape(g.predict(va[FEATS].values), va["load_gw"]), 2)
best = min(res["alpha"], key=res["alpha"].get)
res["alpha_chosen"] = float(best)

# Level blocks vs smooth trend, both with the chosen alpha.
d = add_features(raw, VAL_START, float(best) or 1.0)
d["t"] = (d.index - d.index[0]).days
tr, va = d[d.index < VAL_START], d[(d.index >= VAL_START) & (d.index <= VAL_END)]
res["trend"]["level_blocks"] = round(mape(fit(tr).predict(va[FEATS].values), va["load_gw"]), 2)
F2 = FEATS[:-1] + ["t"]
terms2 = s(0, n_splines=20) + s(1, basis="cp", n_splines=20) + f(2) + f(3) + f(4) + s(5, n_splines=8)
res["trend"]["smooth_trend"] = round(mape(fit(tr, terms2, F2).predict(va[F2].values), va["load_gw"]), 2)

# 3. Error correction with the latest known errors (up to D-2): k and beta for both GAM variants.
d = add_features(raw, VAL_START, float(best) or 1.0)
d = d[d.index <= VAL_END]
g = fit(d[d.index < VAL_START])
vm = d.index >= VAL_START
grid = [(round(k, 1), b) for k in np.arange(0, 1.01, 0.1) for b in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)]
res["errcorr"] = {}
for name, p in (("gam", pd.Series(g.predict(d[FEATS].values), d.index)), ("gam_d1", pd.Series(predict_d1(g, d), d.index))):
    score = lambda kb: mape(err_correct(p, d["load_gw"], *kb)[vm], d["load_gw"][vm])
    k, b = min(grid, key=score)
    res["errcorr"][name] = {"k": float(k), "beta": b, "val_before": round(mape(p[vm], d["load_gw"][vm]), 2), "val_after": round(score((k, b)), 2)}

json.dump(res, open(HERE.parent / "docs" / "data" / "validation.json", "w"), indent=1)
print(json.dumps(res, indent=1))
