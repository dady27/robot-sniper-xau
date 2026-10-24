"""Traversée : quelle entrée, quel stop, quelle sortie ?

Rejoue chaque contact d'un fichier de traversée (scripts/etude_traversee.py) avec :
- entrée : ordre stop au niveau (« stop ») ou au marché à la clôture de la minute du contact (« close ») ;
- stop : k × ATR H1 (connu à 01h New York) sous l'entrée ;
- sortie : cible x R, sinon au bout de N minutes (ou à 16h New York), le premier des deux.
Hypothèse pessimiste : stop d'abord si la minute touche les deux. Contrôle : achat et vente au marché
à la clôture de la minute du contact, mêmes stop, cible et sortie, moyenne des deux.
usage: h_trav_exits.py TRADES.csv OUT.csv [famille ...]
"""
import sys

import numpy as np
import pandas as pd

from lab import AtrBook, load, ny

STOPS = (0.5, 1.0, 1.5)
TARGETS = (1.0, 2.0, 3.0, None)
TIMES = (15, 30, 60, 120, 240, None)  # None = 16h New York


def outcome(e, risk, up, dn, cl, n_t, x):
    """R d'un trade suivi sur n_t minutes : cible x R (None = sans cible), stop à -1R."""
    k_sl = np.searchsorted(dn[:n_t], -(e - risk), "left")
    k_tp = np.searchsorted(up[:n_t], e + x * risk, "left") if x is not None else n_t
    if k_sl < n_t and k_sl <= k_tp:
        return -1.0
    if k_tp < n_t:
        return x
    r = (cl[n_t - 1] - e) / risk
    return min(r, x) if x is not None else r


def key(ks, T, x):
    return f"{ks:g}_{'16h' if T is None else T}_{'none' if x is None else f'{x:g}'}"


def main():
    ev = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    if sys.argv[3:]:
        ev = ev[ev["famille"].isin(sys.argv[3:])].reset_index(drop=True)
    days = pd.to_datetime(ev["day"]).dt.date
    m1 = load(days.min(), days.max())
    atr_h1 = AtrBook(m1).fn("h1")
    t = m1.index
    O, H, L, C, SP = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "spread"))
    zero = np.zeros(len(t))
    space = {1: (O, H, L, C, SP, zero), -1: (-O, -L, -H, -C, zero, SP)}
    ts = pd.to_datetime(ev["time"], utc=True)
    rows = []
    for k in range(len(ev)):
        i0 = int(t.searchsorted(ts.iat[k]))
        sign = 1 if ev["side"].iat[k] == "long" else -1
        day = days.iat[k]
        a = atr_h1(t[int(t.searchsorted(ny(day, 1)))])
        b = int(t.searchsorted(ny(day, 16)))
        if b <= i0 + 1:
            continue
        # minutes suivies pour chaque sortie au temps
        n_of = {T: (b - i0 - 1 if T is None else
                    max(1, min(b, int(t.searchsorted(t[i0] + pd.Timedelta(minutes=T), "right"))) - i0 - 1))
                for T in TIMES}
        paths = {}
        for s2 in (1, -1):
            o, h, l, c, ec, xc = space[s2]
            paths[s2] = (np.maximum.accumulate(h[i0 + 1:b] - xc[i0 + 1:b]),
                         np.maximum.accumulate(-(l[i0 + 1:b] - xc[i0 + 1:b])),
                         c[i0 + 1:b] - xc[i0 + 1:b], c[i0] + ec[i0])
        o, h, l, c, ec, xc = space[sign]
        entries = {"stop": max(sign * ev["level"].iat[k], o[i0]) + ec[i0], "close": c[i0] + ec[i0]}
        row = {"time": t[i0], "day": day, "famille": ev["famille"].iat[k], "level": ev["level"].iat[k],
               "side": ev["side"].iat[k], "year": day.year, "atr_h1": a, "spread_atr": SP[i0] / a}
        up, dn, cl, _ = paths[sign]
        for ks in STOPS:
            for T in TIMES:
                for x in TARGETS:
                    kk = key(ks, T, x)
                    for name, e in entries.items():
                        row[f"{name}_{kk}"] = outcome(e, ks * a, up, dn, cl, n_of[T], x)
                    row[f"ctrl_{kk}"] = np.mean([outcome(p[3], ks * a, p[0], p[1], p[2], n_of[T], x)
                                                 for p in paths.values()])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    res = []
    for ks in STOPS:
        for T in TIMES:
            for x in TARGETS:
                kk = key(ks, T, x)
                r = {"stop_atr": ks, "sortie_min": "16h" if T is None else T, "cible_r": "aucune" if x is None else x}
                for name in ("stop", "close"):
                    v, d = df[f"{name}_{kk}"], df[f"{name}_{kk}"] - df[f"ctrl_{kk}"]
                    r[f"E_{name}"] = v.mean()
                    r[f"t_{name}"] = v.mean() / v.std(ddof=1) * np.sqrt(len(v))
                    r[f"ecart_{name}"] = d.mean()
                    r[f"t_ecart_{name}"] = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
                r["E_ctrl"] = df[f"ctrl_{kk}"].mean()
                res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out.replace(".csv", "_synthese.csv"), index=False)
    print(f"{len(df)} contacts, familles {sorted(df['famille'].unique())}, spread moyen {df['spread_atr'].mean():.3f} ATR H1")
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:+.3f}".format):
        print(res.to_string(index=False))


main()
