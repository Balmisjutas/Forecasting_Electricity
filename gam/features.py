"""Shared features for fit.py and validate.py."""
import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s

FEATS = ["temp_s", "doy", "daytype", "pont", "xmas", "level"]


def load(path):
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def add_features(df, test_start, alpha):
    """Add model features. `level` = 12-month block counted back from test_start (test days get the last block)."""
    df = df.copy()
    df["temp_s"] = df["temp"].ewm(alpha=alpha, adjust=False).mean()
    # Temperature known only up to the day before (persistence): smoothed value built from D-1.
    df["temp_s_d1"] = (alpha * df["temp"].shift(1) + (1 - alpha) * df["temp_s"].shift(1))
    wd = df.index.day_of_week
    hol = df["holiday"].astype(bool)
    # Day type: Monday..Sunday = 0..6; a holiday on a weekday = 7 (its own level); a weekend holiday changes nothing.
    df["weekday"] = wd
    df["daytype"] = np.where(hol & (wd < 5), 7, wd)
    # Bridge day ("pont"): Monday before a Tuesday holiday, Friday after a Thursday holiday.
    nxt = hol.shift(-1, fill_value=False)
    prv = hol.shift(1, fill_value=False)
    df["pont"] = (((wd == 0) & nxt) | ((wd == 4) & prv)) & ~hol
    df["pont"] = df["pont"].astype(int)
    md = df.index.month * 100 + df.index.day
    df["xmas"] = (((md >= 1224) | (md <= 101)) & ~hol).astype(int)
    k = ((pd.Timestamp(test_start) - df.index).days - 1) // 365
    df["level"] = np.clip(10 - k, 0, 10)
    return df


def terms_level():
    return (s(0, n_splines=20) + s(1, basis="cp", n_splines=20) + f(2) + f(3) + f(4) + f(5))


def fit(train, terms=None, feats=FEATS):
    gam = LinearGAM(terms if terms is not None else terms_level())
    gam.gridsearch(train[feats].values, train["load_gw"].values, lam=np.logspace(-2, 3, 6), progress=False)
    return gam


def mape(p, y):
    p, y = np.asarray(p), np.asarray(y)
    ok = ~np.isnan(p) & ~np.isnan(y)
    return float(np.mean(np.abs(p[ok] - y[ok]) / y[ok]) * 100)


def err_correct(pred, actual, k, beta):
    """Add k x the smoothed past errors that are KNOWN when a J-1 forecast is made (errors up to D-2)."""
    e = (actual - pred).ewm(alpha=beta, adjust=False).mean().shift(2)
    return pred + k * e


def predict_d1(gam, d):
    """Prediction with the temperature known up to the day before (persistence)."""
    X = d[FEATS].values.copy()
    X[:, 0] = d["temp_s_d1"].bfill().values
    return gam.predict(X)
