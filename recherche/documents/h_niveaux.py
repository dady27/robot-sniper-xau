"""Cassure de nouveaux niveaux : range asiatique, Londres, ouverture de New York, heure précédente.

Même principe que la traversée : au premier contact du niveau dans la fenêtre, entrée au marché à la
clôture de la minute du contact (et, pour info, ordre stop au niveau), stop k ATR H1, cible x R ou
sortie au temps. Contrôle : achat et vente au marché au même instant. On mesure aussi l'élan :
mouvement dans le sens de la cassure, de la clôture de la minute du contact à t + h minutes.
usage: h_niveaux.py START END OUT_PREFIX
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from lab import AtrBook, business_days, load, ny, sessions

HZ = (1, 5, 15, 30, 60, 120, 240)
STOPS = (0.5, 1.0)
TIMES = (15, 30, 60, 120, None)
TARGETS = (2.0, 3.0, None)


def outcome(e, risk, up, dn, cl, n_t, x):
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
    start, end, out = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), sys.argv[3]
    m1 = load(start, end)
    t = m1.index
    O, H, L, C, SP = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "spread"))
    zero = np.zeros(len(t))
    space = {1: (O, H, L, C, SP, zero), -1: (-O, -L, -H, -C, zero, SP)}
    atr_h1 = AtrBook(m1).fn("h1")

    def idx(ts):
        return int(t.searchsorted(ts))

    rows = []
    for d in business_days(start, end):
        i01, i16 = idx(ny(d, 1)), idx(ny(d, 16))
        if i16 - i01 < 600:
            continue
        a = atr_h1(t[i01])
        if not (np.isfinite(a) and a > 0):
            continue
        cands = []  # (famille, i_range0, i_range1, i_from, i_to, distance minimale en ATR H1)
        cands.append(("asie", idx(sessions.trading_day(d).start), i01, i01, i16, 0.25))
        cands.append(("londres", idx(ny(d, 2)), idx(ny(d, 7)), idx(ny(d, 7)), i16, 0.25))
        cands.append(("ouverture_ny", idx(ny(d, 8, 30)), idx(ny(d, 9, 30)), idx(ny(d, 9, 30)), i16, 0.25))
        for hh in range(2, 16):
            cands.append(("heure_prec", idx(ny(d, hh - 1)), idx(ny(d, hh)), idx(ny(d, hh)), idx(ny(d, hh + 1)), 0.1))
        for fam, r0, r1, i_from, i_to, dmin in cands:
            if r1 - r0 < 30 or i_to - i_from < 30:
                continue
            ref = C[i_from - 1]
            for sign, lvl in ((1, H[r0:r1].max()), (-1, L[r0:r1].min())):
                o, h, _, c, ec, _ = space[sign]
                if sign * ref > sign * lvl - dmin * a:  # trop près du niveau au début de la fenêtre
                    continue
                touch = np.nonzero(h[i_from:i_to] >= sign * lvl)[0]
                if len(touch) == 0:
                    continue
                i0 = i_from + int(touch[0])
                b = i16 if fam != "heure_prec" else min(i16, i0 + 240)
                if b <= i0 + 2:
                    continue
                n_of = {T: (b - i0 - 1 if T is None else
                            max(1, min(b, idx(t[i0] + pd.Timedelta(minutes=T) + pd.Timedelta(seconds=1))) - i0 - 1))
                        for T in TIMES}
                paths = {}
                for s2 in (1, -1):
                    _, h2, l2, c2, ec2, xc2 = space[s2]
                    paths[s2] = (np.maximum.accumulate(h2[i0 + 1:b] - xc2[i0 + 1:b]),
                                 np.maximum.accumulate(-(l2[i0 + 1:b] - xc2[i0 + 1:b])),
                                 c2[i0 + 1:b] - xc2[i0 + 1:b], c2[i0] + ec2[i0])
                e_close = c[i0] + ec[i0]
                e_stop = max(sign * lvl, o[i0]) + ec[i0]
                row = {"famille": fam, "day": d, "time": t[i0], "side": "long" if sign == 1 else "short",
                       "level": lvl, "hour_ny": t[i0].tz_convert(sessions.NEW_YORK).hour, "atr_h1": a,
                       "spread_atr": SP[i0] / a, "fill": sign * (C[i0] - lvl) / a}
                for hz in HZ:
                    j = idx(t[i0] + pd.Timedelta(minutes=hz) + pd.Timedelta(seconds=1)) - 1
                    row[f"m{hz}"] = sign * (C[j] - C[i0]) / a if j < b else np.nan
                up, dn, cl, _ = paths[sign]
                for ks in STOPS:
                    for T in TIMES:
                        for x in TARGETS:
                            kk = key(ks, T, x)
                            row[f"close_{kk}"] = outcome(e_close, ks * a, up, dn, cl, n_of[T], x)
                            row[f"stop_{kk}"] = outcome(e_stop, ks * a, up, dn, cl, n_of[T], x)
                            row[f"ctrl_{kk}"] = np.mean([outcome(p[3], ks * a, p[0], p[1], p[2], n_of[T], x)
                                                         for p in paths.values()])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}_trades.csv", index=False)
    n_days = len(list(business_days(start, end)))

    def st(x):
        x = x.dropna()
        return f"{x.mean():+.3f}({x.mean() / x.std(ddof=1) * np.sqrt(len(x)):+.1f})"

    res = []
    for fam, g in df.groupby("famille"):
        print(f"\n== {fam} : {len(g)} contacts, {len(g) / n_days:.2f} par jour, spread {g['spread_atr'].mean():.3f} ATR H1, "
              f"avance dans la minute {st(g['fill'])}")
        print("   élan (ATR H1) " + " ".join(f"{hz}m:{st(g[f'm{hz}'])}" for hz in HZ))
        for ks in STOPS:
            for T in TIMES:
                for x in TARGETS:
                    kk = key(ks, T, x)
                    v, cv = g[f"close_{kk}"], g[f"ctrl_{kk}"]
                    d_ = v - cv
                    res.append({"famille": fam, "n": len(g), "par_jour": len(g) / n_days, "stop_atr": ks,
                                "sortie": "16h" if T is None else T, "cible": "aucune" if x is None else x,
                                "E_close": v.mean(), "t_close": v.mean() / v.std(ddof=1) * np.sqrt(len(v)),
                                "E_stop": g[f"stop_{kk}"].mean(), "E_ctrl": cv.mean(),
                                "ecart": d_.mean(), "t_ecart": d_.mean() / d_.std(ddof=1) * np.sqrt(len(d_))})
    res = pd.DataFrame(res)
    res.to_csv(f"{out}_synthese.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 400, "display.float_format", "{:+.3f}".format):
        best = res.sort_values("t_close", ascending=False).groupby("famille").head(4)
        print("\nMeilleurs réglages par famille (entrée au marché) :")
        print(best.to_string(index=False))


main()
