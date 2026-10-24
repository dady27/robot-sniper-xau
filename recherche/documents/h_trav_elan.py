"""Traversée : combien de temps dure l'élan après le contact du niveau ?

Pour chaque trade d'un fichier de traversée (scripts/etude_traversee.py), mouvement dans le sens de
la cassure, en ATR H1, de la clôture de la minute du contact (ou du niveau) jusqu'à t + h minutes,
sans dépasser 16h New York. Le contrôle neutre (achat et vente au même instant) vaut zéro par
construction : la moyenne mesure l'élan.
usage: h_trav_elan.py TRADES.csv OUT.csv
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from lab import AtrBook, load, ny, sessions

H = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360)


def stat(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.mean():+.3f} (t {x.mean() / x.std(ddof=1) * np.sqrt(len(x)):+.1f}, n {len(x)})"


def main():
    ev = pd.read_csv(sys.argv[1])
    days = pd.to_datetime(ev["day"]).dt.date
    m1 = load(days.min(), days.max())
    atr_h1 = AtrBook(m1).fn("h1")
    t = m1.index
    C = m1["close"].to_numpy(float)
    ts = pd.to_datetime(ev["time"], utc=True)
    rows = []
    for k in range(len(ev)):
        i0 = int(t.searchsorted(ts.iat[k]))
        sign = 1 if ev["side"].iat[k] == "long" else -1
        day: date = days.iat[k]
        a = atr_h1(t[int(t.searchsorted(ny(day, 1)))])  # ATR H1 connu à 01h, comme sniper/traversee.py
        lvl = ev["level"].iat[k]
        end = ny(day, 16)
        row = {"time": t[i0], "famille": ev["famille"].iat[k], "side": ev["side"].iat[k], "level": lvl,
               "year": day.year, "hour_ny": t[i0].tz_convert(sessions.NEW_YORK).hour,
               "fill": sign * (C[i0] - lvl) / a}
        for h in H:
            th = t[i0] + pd.Timedelta(minutes=h)
            if th > end:
                row[f"c{h}"] = row[f"l{h}"] = np.nan
                continue
            j = int(t.searchsorted(th, "right")) - 1
            row[f"c{h}"] = sign * (C[j] - C[i0]) / a  # depuis la clôture de la minute du contact
            row[f"l{h}"] = sign * (C[j] - lvl) / a    # depuis le niveau (ordre stop rempli pile)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(sys.argv[2], index=False)
    print(f"{len(df)} trades ; avance prise dans la minute du contact : {stat(df['fill'])}")
    cent = df["famille"].eq("ronds") & np.isclose(df["level"] % 100, 0)
    groups = [("tous", df)] + [(f, g) for f, g in df.groupby("famille")] + [("ronds, multiples de 100 $", df[cent])]
    for name, g in groups:
        if g.empty:
            continue
        print(f"\n== {name}")
        for h in H:
            print(f"  +{h:>3} min  depuis la clôture {stat(g[f'c{h}'])}   depuis le niveau {stat(g[f'l{h}'])}")


main()
