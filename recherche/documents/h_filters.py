"""Filtres de tendance des documents (biais HTF, premium / discount) appliqués aux sweeps de l'Asie.

Pour chaque événement déjà mesuré (sweep + confirmation, 2015-2023), on regarde si le trade va
dans le sens de la tendance journalière, de la veille, de la semaine, ou s'il achète en discount
(sous le milieu du range de la veille) / vend en premium. Écart au contrôle neutre et t.
usage: h_filters.py EVENTS.csv [EVENTS2.csv ...]
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from lab import daily, load

m1 = load(date(2015, 1, 1), date(2024, 1, 1))
D = daily(m1)
D.index = pd.to_datetime(D.index)
f = pd.DataFrame(index=D.index)
f["sma20"] = np.sign(D["close"] - D["close"].rolling(20).mean()).shift(1)
f["ema100"] = np.sign(D["close"] - D["close"].ewm(span=100).mean()).shift(1)
f["veille"] = np.sign(D["close"] - D["open"]).shift(1)
f["semaine"] = np.sign(D["close"] - D["close"].shift(5)).shift(1)
f["mid_veille"] = ((D["high"] + D["low"]) / 2).shift(1)
for path in sys.argv[1:]:
    ev = pd.read_csv(path, parse_dates=["day"])
    ev = ev.join(f, on="day")
    sgn = np.where(ev["side"] == "long", 1, -1)
    ev["discount"] = np.sign(ev["mid_veille"] - ev["entry"]) * sgn  # +1 : achat sous le milieu / vente au-dessus
    print(f"\n== {path.split('/')[-1]} : {len(ev)} événements")
    for col in ("sma20", "ema100", "veille", "semaine", "discount"):
        al = (ev[col] * (sgn if col != "discount" else 1)) > 0
        parts = []
        for name, g in (("avec", ev[al]), ("contre", ev[~al & ev[col].notna()])):
            d = g["r2.5"] - g["c2.5"]
            parts.append(f"{name} {d.mean():+.3f}R (t {d.mean() / d.std(ddof=1) * np.sqrt(len(d)):+.2f}, n {len(d)}, E {g['r2.5'].mean():+.3f})")
        print(f"  {col:9s} " + " | ".join(parts))
