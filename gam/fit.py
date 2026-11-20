"""Fit a GAM on daily French load and write the page data to ../docs/data/.

Model (pygam LinearGAM):
  load_gw ~ s(temp_s) + s(doy, cyclic) + f(weekday) + f(holiday) + f(level)
  temp_s = exponentially smoothed daily temperature (alpha 0.5: today 50%, yesterday 25%, ...),
           because buildings react to the last days of weather. 0.5 was chosen on a VALIDATION year
           (2024-09 .. 2025-08), not on the test year.
  level  = one level per 12-month block before the test; test days use the last block. A smooth trend
           curve extrapolated badly (validation MAPE 13.9% vs 2.1%), so it was dropped.
Train: 2016-01-01 .. 2025-08-31.  Test: 2025-09-01 .. 2026-08-31 (last 12 months with the RTE forecast).

IMPORTANT LIMIT: the GAM uses the OBSERVED temperature of the day. RTE's day-ahead forecast
(prevision_j1) had to use a weather FORECAST. So the GAM has an unfair advantage; the comparison
shows what the structure can explain, it is not a "we beat RTE" claim.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(HERE / "data" / "daily.csv", parse_dates=["date"], index_col="date")
TEST_START, TEST_END = "2025-09-01", "2026-08-31"
ALPHA = 0.5
df["temp_s"] = df["temp"].ewm(alpha=ALPHA, adjust=False).mean()
k = ((pd.Timestamp(TEST_START) - df.index).days - 1) // 365
df["level"] = np.clip(10 - k, 0, 10)
train = df[df.index < TEST_START]
test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()

FEATS = ["temp_s", "doy", "weekday", "holiday", "level"]
NAMES = ["Temperature", "Season", "Weekday", "Holiday", "Yearly level"]
terms = s(0, n_splines=20) + s(1, basis="cp", n_splines=20) + f(2) + f(3) + f(4)
gam = LinearGAM(terms)
gam.gridsearch(train[FEATS].values, train["load_gw"].values, lam=np.logspace(-2, 3, 6), progress=False)

Xte = test[FEATS].values
test["gam"] = gam.predict(Xte)
intercept = float(gam.coef_[-1])
raw = {n: gam.partial_dependence(term=i, X=Xte) for i, n in enumerate(NAMES)}
assert np.allclose(intercept + sum(raw.values()), test["gam"].values, atol=1e-6)

# Re-centre the terms so that "0" means an ordinary day, and move the shifts into the base:
#   Temperature, Season: 0 = mean effect over the training days
#   Weekday: 0 = mean of the 7 weekdays;  Holiday: 0 = not a holiday
#   Yearly level: the same for every test day (last 12-month block) -> fully inside the base.
Xtr = train[FEATS].values
ref = {
    "Temperature": gam.partial_dependence(term=0, X=Xtr).mean(),
    "Season": gam.partial_dependence(term=1, X=Xtr).mean(),
}
wk = np.tile(Xtr[0], (7, 1)); wk[:, 2] = np.arange(7)
ref["Weekday"] = gam.partial_dependence(term=2, X=wk).mean()
hd = Xtr[:1].copy(); hd[0, 3] = 0
ref["Holiday"] = gam.partial_dependence(term=3, X=hd)[0]
SHOWN = ["Temperature", "Season", "Weekday", "Holiday"]
contrib = {n: raw[n] - ref[n] for n in SHOWN}
base_arr = intercept + sum(ref.values()) + raw["Yearly level"]
assert np.ptp(base_arr) < 1e-9  # one base for the whole test year
intercept = float(base_arr[0])
assert np.allclose(intercept + sum(contrib.values()), test["gam"].values, atol=1e-6)

# "Normal weather" what-if: same day with the mean temperature of that day of year in training years.
normal_temp = train.groupby("doy")["temp_s"].mean()
Xn = Xte.copy()
Xn[:, 0] = test["doy"].map(normal_temp).values
test["gam_normal_temp"] = gam.predict(Xn)

# Seasonal naive: load of the same weekday 52 weeks before.
test["naive"] = df["load_gw"].shift(364).reindex(test.index)


def metrics(pred):
    m = test[["load_gw"]].assign(p=pred).dropna()
    e = m["p"] - m["load_gw"]
    return {
        "n_days": int(len(m)),
        "rmse_gw": round(float(np.sqrt((e**2).mean())), 3),
        "mae_gw": round(float(e.abs().mean()), 3),
        "mape_pct": round(float((e.abs() / m["load_gw"]).mean() * 100), 2),
    }


both = test.dropna(subset=["rte_d1_gw"])
res = {
    "train": [str(train.index.min().date()), str(train.index.max().date())],
    "test": [TEST_START, TEST_END],
    "models": {
        "GAM (observed temperature)": metrics(test["gam"]),
        "RTE day-ahead forecast": metrics(test["rte_d1_gw"]),
        "Seasonal naive (52 weeks before)": metrics(test["naive"]),
    },
    "smoothing_alpha": ALPHA,
    "pseudo_r2": round(float(gam.statistics_["pseudo_r2"]["explained_deviance"]), 4),
    "days_without_rte_forecast": int(test["rte_d1_gw"].isna().sum()),
    "days_gam_closer_than_rte": int(
        ((both["gam"] - both["load_gw"]).abs() < (both["rte_d1_gw"] - both["load_gw"]).abs()).sum()
    ),
    "days_compared": int(len(both)),
}

days = []
for k, (d, r) in enumerate(test.iterrows()):
    days.append({
        "date": str(d.date()),
        "actual": round(r["load_gw"], 3),
        "gam": round(r["gam"], 3),
        "rte": None if pd.isna(r["rte_d1_gw"]) else round(r["rte_d1_gw"], 3),
        "naive": None if pd.isna(r["naive"]) else round(r["naive"], 3),
        "temp": round(r["temp"], 2),
        "temp_s": round(r["temp_s"], 2),
        "temp_normal": round(float(Xn[k, 0]), 2),
        "weather_effect": round(r["gam"] - r["gam_normal_temp"], 3),
        "holiday": int(r["holiday"]),
        "base": round(intercept, 3),
        "terms": {n: round(float(contrib[n][k]), 3) for n in SHOWN},
    })

# Shape of each smooth term over its range (for the "what the model learned" view).
shapes = {}
for i, (n, col) in enumerate(zip(NAMES, FEATS)):
    if n in ("Temperature", "Season"):
        grid = gam.generate_X_grid(term=i, n=100)
        XX = np.tile(train[FEATS].median().values, (len(grid), 1))
        XX[:, i] = grid[:, i]
        shapes[n] = {"x": [round(float(v), 2) for v in XX[:, i]],
                     "y": [round(float(v), 3) for v in gam.partial_dependence(term=i, X=XX)]}
    else:
        vals = sorted(train[col].unique())
        XX = np.tile(train[FEATS].median().values, (len(vals), 1))
        XX[:, i] = vals
        shapes[n] = {"x": [int(v) for v in vals],
                     "y": [round(float(v), 3) for v in gam.partial_dependence(term=i, X=XX)]}

json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
json.dump(days, open(OUT / "days.json", "w"))
json.dump(shapes, open(OUT / "shapes.json", "w"))
print(json.dumps(res, indent=1))

# Same data as one script file, so docs/index.html also works when opened from disk (file://).
with open(OUT / "data.js", "w") as fh:
    fh.write("window.GAM_DATA = ")
    json.dump({"metrics": res, "days": days, "shapes": shapes}, fh)
    fh.write(";\n")
