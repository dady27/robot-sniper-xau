"""Traversée sur d'autres marchés : figures, demi-figures, veille et semaine, par période.

Lit un fichier de trades de scripts/etude_traversee.py par marché (même règle partout : entrée au
marché à la clôture de la minute du contact, stop 1 ATR H1, cible 3R, sortie à 16h New York sinon)
et donne, par marché, période et groupe : trades par jour, spread en R, résultat à 3R, contrôle
neutre, écart au contrôle et statistiques t. Les chiffres ronds sont coupés en figures (multiples
de GRAND_PAS) et demi-figures (les autres).
usage: h_trav_marches.py OUT_PREFIX MARCHE=TRADES.csv:GRAND_PAS [...]
exemple: h_trav_marches.py out/marches USDJPY=out/usdjpy/traversee.csv:1 EURUSD=out/eurusd/traversee.csv:0.01
"""
import sys

import numpy as np
import pandas as pd

KEEP = ["marche", "periode", "groupe", "day", "time", "famille", "level", "side", "entry", "risk", "spread_r",
        "r1", "c1", "r3", "c3", "r_end"]


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan


def read(spec: str) -> pd.DataFrame:
    name, rest = spec.split("=", 1)
    path, big = rest.rsplit(":", 1)
    big = float(big)
    ev = pd.read_csv(path)
    year = pd.to_datetime(ev["day"]).dt.year
    figure = np.isclose(np.round(ev["level"] / big) * big, ev["level"], atol=big * 1e-6, rtol=0)
    ev["marche"] = name
    ev["periode"] = np.where(year <= 2023, "2015-2023", "2024-2026")
    ev["groupe"] = np.where(ev["famille"] == "ronds", np.where(figure, "figures", "demi-figures"), ev["famille"])
    return ev


def main():
    out = sys.argv[1]
    ev = pd.concat([read(s) for s in sys.argv[2:]], ignore_index=True)
    rows = []
    for (m, per), g0 in ev.groupby(["marche", "periode"], sort=False):
        n_days = pd.bdate_range(g0["day"].min(), g0["day"].max()).size
        for grp, g in [("tous", g0)] + list(g0.groupby("groupe")):
            d = g["r3"] - g["c3"]
            yrs = g.groupby(pd.to_datetime(g["day"]).dt.year)["r3"].mean()
            rows.append({"marche": m, "periode": per, "groupe": grp, "n": len(g), "par_jour": len(g) / n_days,
                         "spread_r": g["spread_r"].mean(), "E_r3": g["r3"].mean(), "t_r3": tstat(g["r3"]),
                         "ctrl_r3": g["c3"].mean(), "ecart_r3": d.mean(), "t_ecart_r3": tstat(d),
                         "E_16h": g["r_end"].mean(), "t_16h": tstat(g["r_end"]),
                         "annees_positives": f"{(yrs > 0).sum()}/{len(yrs)}"})
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}_synthese.csv", index=False)
    ev[KEEP].to_csv(f"{out}_trades.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:+.3f}".format):
        print(res.to_string(index=False))


main()
