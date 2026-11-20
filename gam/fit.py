"""Fit the GAM on daily French load and write the page data to ../docs/data/.

Model (pygam LinearGAM), all choices from validate.py (validation year 2024-09 .. 2025-08):
  load_gw ~ s(temp_s) + s(doy, cyclic) + f(daytype) + f(pont) + f(xmas) + f(level)
  temp_s  = exponential moving average of the daily temperature (alpha from validation)
  daytype = Monday..Sunday, plus one level for a holiday on a weekday; a weekend holiday changes nothing
  pont    = bridge day (Monday before a Tuesday holiday, Friday after a Thursday holiday)
  xmas    = 24 Dec .. 1 Jan (not the holidays themselves)
  level   = one level per 12-month block; test days use the last training block
Train: 2016-01-01 .. 2025-08-31.  Test: 2025-09-01 .. 2026-08-31.

LIMITS kept visible on the page:
- The GAM uses the OBSERVED temperature. A second variant uses the temperature known up to D-1 (persistence).
- Error correction (validate.py): add k x the smoothed past errors known at forecast time (up to D-2),
  k and beta chosen on the validation year. Shown for both GAM variants.
- RTE's prevision_j1 sits about 1 GW under the actual since 2023 (level/perimeter offset, also against
  eCO2mix's own "consommation"). We show RTE raw AND corrected by the mean offset of the 12 months
  before the test (known in advance, so the correction is fair).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATS, add_features, err_correct, fit, load, predict_d1

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
VAL = json.load(open(OUT / "validation.json"))
ALPHA = VAL["alpha_chosen"]
TEST_START, TEST_END = "2025-09-01", "2026-08-31"

df = add_features(load(HERE / "data" / "daily.csv"), TEST_START, ALPHA)
train = df[df.index < TEST_START]
test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
gam = fit(train)

Xte = test[FEATS].values
test["gam"] = gam.predict(Xte)
Xd1 = Xte.copy(); Xd1[:, 0] = test["temp_s_d1"].values
test["gam_d1"] = gam.predict(Xd1)

# ---- term contributions, re-centred so that 0 = an ordinary day
pd_ = lambda i, X: gam.partial_dependence(term=i, X=X)
raw = {i: pd_(i, Xte) for i in range(6)}
intercept0 = float(gam.coef_[-1])
assert np.allclose(intercept0 + sum(raw.values()), test["gam"].values, atol=1e-6)
Xtr = train[FEATS].values
ref_temp = pd_(0, Xtr).mean()
ref_season = pd_(1, Xtr).mean()
row = Xtr[:1].copy()
def at(i, v):
    r = row.copy(); r[0, i] = v; return float(pd_(i, r)[0])
dt = np.array([at(2, v) for v in range(8)])          # day-type effect for each value 0..7
ref_week = dt[:7].mean()
ref_pont, ref_xmas = at(3, 0), at(4, 0)
wk = test["weekday"].values
contrib = {
    "Temperature": raw[0] - ref_temp,
    "Season": raw[1] - ref_season,
    "Weekday": dt[wk] - ref_week,                     # effect of the weekday itself
    "Holiday": raw[2] - dt[wk],                       # extra effect of a weekday holiday vs that weekday
    "Bridge": raw[3] - ref_pont,
    "Christmas": raw[4] - ref_xmas,
}
base_arr = intercept0 + ref_temp + ref_season + ref_week + ref_pont + ref_xmas + raw[5]
assert np.ptp(base_arr) < 1e-9
BASE = float(base_arr[0])
assert np.allclose(BASE + sum(contrib.values()), test["gam"].values, atol=1e-6)

# ---- "normal weather" what-if: smoothed 2016-2025 mean of temp_s for the same day of year
clim = train.groupby("doy")["temp_s"].mean().reindex(range(1, 367)).interpolate()
clim = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
clim.index = range(1, 367)
Xn = Xte.copy(); Xn[:, 0] = test["doy"].map(clim).values
test["gam_clim"] = gam.predict(Xn)

# ---- baselines, by DATE (not by row)
L = df["load_gw"]
test["naive52"] = L.reindex(test.index - pd.Timedelta(days=364)).values
test["naive7"] = L.reindex(test.index - pd.Timedelta(days=7)).values
pre = df[(df.index >= "2024-09-01") & (df.index < TEST_START)]
OFFSET = float((pre["rte_d1_gw"] - pre["load_gw"]).mean())
OFFSET_ECO2 = float((pre["rte_d1_gw"] - pre["eco2_cons_gw"]).mean())
test["rte_corr"] = test["rte_d1_gw"] - OFFSET

# ---- error correction with the latest known errors (up to D-2); first test days use training-period errors
full = df[df.index <= TEST_END]
y = full["load_gw"]
for name, p in (("gam", pd.Series(gam.predict(full[FEATS].values), full.index)),
                ("gam_d1", pd.Series(predict_d1(gam, full), full.index))):
    ec = VAL["errcorr"][name]
    test[name + "_corr"] = err_correct(p, y, ec["k"], ec["beta"]).reindex(test.index)


def metrics(pred):
    m = test[["load_gw"]].assign(p=pred).dropna()
    e = m["p"] - m["load_gw"]
    return {
        "n_days": int(len(m)),
        "mape_pct": round(float((e.abs() / m["load_gw"]).mean() * 100), 2),
        "mae_gw": round(float(e.abs().mean()), 3),
        "rmse_gw": round(float(np.sqrt((e**2).mean())), 3),
        "bias_gw": round(float(e.mean()), 3),
    }


res_gam = test["gam"] - test["load_gw"]
hol = test["holiday"].astype(bool)
monthly_bias = res_gam.groupby(test.index.to_period("M")).mean()
x = np.linspace(-5, 32, 371)
XX = np.tile(np.median(Xtr, axis=0), (len(x), 1)); XX[:, 0] = x
temp_curve = pd_(0, XX) - ref_temp
slope = lambda a, b: float((np.interp(b, x, temp_curve) - np.interp(a, x, temp_curve)) / (b - a))
res = {
    "train": [str(train.index.min().date()), str(train.index.max().date())],
    "test": [TEST_START, TEST_END],
    "alpha": ALPHA,
    "models": {
        "gam": metrics(test["gam"]),
        "gam_d1": metrics(test["gam_d1"]),
        "gam_corr": metrics(test["gam_corr"]),
        "gam_d1_corr": metrics(test["gam_d1_corr"]),
        "rte_raw": metrics(test["rte_d1_gw"]),
        "rte_corr": metrics(test["rte_corr"]),
        "naive7": metrics(test["naive7"]),
        "naive52": metrics(test["naive52"]),
    },
    "rte_offset_gw": round(OFFSET, 2),
    "rte_offset_vs_eco2mix_gw": round(OFFSET_ECO2, 2),
    "residual_lag1_autocorr": round(float(res_gam.autocorr(1)), 2),
    "residual_lag2_autocorr": round(float(res_gam.autocorr(2)), 2),
    "gam_corr_bias_summer_gw": round(float((test["gam_corr"] - test["load_gw"])[test.index.month.isin([6, 7, 8])].mean()), 2),
    "gam_d1_corr_bias_summer_gw": round(float((test["gam_d1_corr"] - test["load_gw"])[test.index.month.isin([6, 7, 8])].mean()), 2),
    "gam_mae_holidays_gw": round(float(res_gam[hol].abs().mean()), 2),
    "gam_bias_weekday_holidays_gw": round(float(res_gam[hol & (test["weekday"] < 5)].mean()), 2),
    "gam_mae_other_gw": round(float(res_gam[~hol].abs().mean()), 2),
    "gam_bias_summer_gw": round(float(res_gam[test.index.month.isin([6, 7, 8])].mean()), 2),
    "gam_bias_winter_gw": round(float(res_gam[test.index.month.isin([12, 1, 2])].mean()), 2),
    "max_temp_s_train": round(float(train["temp_s"].max()), 1),
    "max_temp_s_test": round(float(test["temp_s"].max()), 1),
    "slope_0_5_gw_per_c": round(slope(0, 5), 2),
    "slope_5_10_gw_per_c": round(slope(5, 10), 2),
    "corr_temp_season_terms_train": round(float(np.corrcoef(pd_(0, Xtr), pd_(1, Xtr))[0, 1]), 2),
    "interpolated_temp_days_test": int(test["temp_interp"].sum()),
    "lam": [float(np.ravel(l)[0]) for l in gam.lam],
    "monthly_bias_gw": {str(k): round(float(v), 2) for k, v in monthly_bias.items()},
}

NAMES = list(contrib)
days = []
for k, (d, r) in enumerate(test.iterrows()):
    days.append({
        "date": str(d.date()),
        "actual": round(r["load_gw"], 3),
        "gam": round(r["gam"], 3),
        "rte": None if pd.isna(r["rte_corr"]) else round(r["rte_corr"], 3),
        "temp": round(r["temp"], 2),
        "temp_s": round(r["temp_s"], 2),
        "temp_clim": round(float(Xn[k, 0]), 2),
        "weather_effect": round(r["gam"] - r["gam_clim"], 3),
        "holiday": int(r["holiday"]), "pont": int(r["pont"]), "xmas": int(r["xmas"]),
        "temp_interp": int(r["temp_interp"]),
        "base": round(BASE, 3),
        "terms": {n: round(float(contrib[n][k]), 3) for n in NAMES},
    })
shapes = {"Temperature": {"x": [round(float(v), 2) for v in x], "y": [round(float(v), 3) for v in temp_curve]}}

json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
json.dump(days, open(OUT / "days.json", "w"))
json.dump(shapes, open(OUT / "shapes.json", "w"))
with open(OUT / "data.js", "w") as fh:
    fh.write("window.GAM_DATA = ")
    json.dump({"metrics": res, "validation": VAL, "days": days, "shapes": shapes}, fh)
    fh.write(";\n")
print(json.dumps(res, indent=1))
