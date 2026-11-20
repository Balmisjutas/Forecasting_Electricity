"""Build one daily table from the open RTE / ODRE files in data_raw/.

Output: data/daily.csv with, for each day (Paris time):
  load_gw      mean national load (GW), from consommation-quotidienne-brute (30-min values)
  rte_d1_gw    mean of RTE's day-ahead forecast (prevision_j1), from eco2mix-national-cons-def
  temp         national daily mean temperature (C) = plain mean of the regional means (not weighted)
  holiday, weekday, doy, t (days since start)
Daily energy in GWh = load_gw * 24.
"""
from pathlib import Path

import holidays
import pandas as pd

# Sources (ODRE exports, CSV with ";"):
#   load.csv : https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/consommation-quotidienne-brute/exports/csv?select=date_heure,consommation_brute_electricite_rte&delimiter=%3B
#   temp.csv : https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/temperature-quotidienne-regionale/exports/csv?delimiter=%3B
#   rte.csv  : https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv?select=date_heure,consommation,prevision_j1,prevision_j&delimiter=%3B
RAW = Path(__file__).parent / "data_raw"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def daily_mean(path, cols):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    ts = pd.to_datetime(df["date_heure"], utc=True).dt.tz_convert("Europe/Paris")
    df["day"] = ts.dt.date
    g = df.groupby("day")[cols]
    out = g.mean()
    out["n"] = g.size()
    return out


load = daily_mean(RAW / "load.csv", ["consommation_brute_electricite_rte"])
load = load[load["n"] >= 46]  # full days only (48 half hours; 46/50 on clock-change days)
load = load.rename(columns={"consommation_brute_electricite_rte": "load_gw"})[["load_gw"]] / 1000

rte = daily_mean(RAW / "rte.csv", ["prevision_j1"])
rte = rte[rte["n"] >= 90].rename(columns={"prevision_j1": "rte_d1_gw"})[["rte_d1_gw"]] / 1000

temp = pd.read_csv(RAW / "temp.csv", sep=";", encoding="utf-8-sig")
nreg = temp.groupby("date")["tmoy"].count()
temp = temp.groupby("date")["tmoy"].mean().rename("temp").to_frame()
temp = temp[nreg >= 12]
temp.index = pd.to_datetime(temp.index).date

df = load.join(temp, how="inner").join(rte, how="left")
df.index = pd.to_datetime(df.index)
df.index.name = "date"
df = df[df.index >= "2016-01-01"]
fr = holidays.France(years=range(2016, 2027))
df["holiday"] = [int(d in fr) for d in df.index.date]
df["weekday"] = df.index.day_of_week
df["doy"] = df.index.day_of_year
df["t"] = (df.index - df.index[0]).days
df.to_csv(OUT / "daily.csv", float_format="%.4f")

# Check: annual energy vs RTE published totals (France consumption is about 440-470 TWh per year).
annual = (df["load_gw"] * 24 / 1000).groupby(df.index.year).sum()
print("days", len(df), df.index.min().date(), "..", df.index.max().date())
print("annual TWh (full years):")
print(annual.round(1).to_string())
print("missing RTE forecast days:", df["rte_d1_gw"].isna().sum())
