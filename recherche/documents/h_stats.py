"""Affirmations statistiques des documents, sans règle d'entrée :

A1. Londres -> New York : continuation (BFX) ou retournement (Sir Jones, « New York reversal ») ?
A2. Le stop hunt de Londres (01h-05h NY, hors du range asiatique) fait-il le HOD/LOD du jour ?
    Comparé à une marche au hasard qui garde exactement la volatilité minute par minute du jour
    (mêmes |rendements M1|, signes tirés au hasard).
A3. Cycle de 3 jours (après 3 baisses, le marché remonte ?) et jour du plus haut / plus bas de la semaine,
    comparés au même tirage de signes.
usage: h_stats.py START END
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from lab import _span, business_days, daily, load, ny, sessions

RNG = np.random.default_rng(7)
DRAWS = 40


def a1(m1, D, start, end):
    rows = []
    for d in business_days(start, end):
        if d not in D.index or not np.isfinite(D.at[d, "adr"]):
            continue
        bars = _span(m1, ny(d, 3), ny(d, 16))
        if len(bars) < 600:
            continue
        t = bars.index
        c = bars["close"].to_numpy()
        o0 = bars["open"].iloc[0]
        i7, i11 = int(t.searchsorted(ny(d, 7))), int(t.searchsorted(ny(d, 11)))
        adr = D.at[d, "adr"]
        rows.append({"day": d, "lon": (c[i7 - 1] - o0) / adr, "ny11": (c[i11 - 1] - c[i7 - 1]) / adr,
                     "ny16": (c[-1] - c[i7 - 1]) / adr})
    x = pd.DataFrame(rows)
    print(f"\n== A1 Londres (03h-07h NY) puis New York : {len(x)} jours")
    for col in ("ny11", "ny16"):
        r = x["lon"].corr(x[col])
        same = (np.sign(x["lon"]) == np.sign(x[col])).mean()
        big = x[x["lon"].abs() >= x["lon"].abs().quantile(2 / 3)]
        cond = (np.sign(big["lon"]) * big[col])
        print(f"  {col}: corrélation {r:+.3f} (t {r * np.sqrt(len(x) - 2) / np.sqrt(1 - r * r):+.2f}), même sens {same:.3f}, "
              f"tiers des plus gros Londres : mouvement NY dans le sens de Londres {cond.mean():+.3f} ADR "
              f"(t {cond.mean() / cond.std(ddof=1) * np.sqrt(len(cond)):+.2f}, n {len(cond)})")


def flip_paths(c: np.ndarray, draws: int) -> np.ndarray:
    """Chemins de même |variation| minute par minute, signes au hasard (martingale)."""
    dc = np.diff(c)
    s = RNG.choice((-1.0, 1.0), size=(draws, len(dc)))
    return c[0] + np.concatenate([np.zeros((draws, 1)), np.cumsum(s * np.abs(dc), axis=1)], axis=1)


def hod_lod_stat(paths: np.ndarray, i_ae: int, i_r0: int, i_r1: int):
    """Pour chaque chemin : côté du raid (0 si aucun) et vrai si l'extrême du raid à 05h tient jusqu'à 17h."""
    hi, lo = paths[:, :i_ae].max(1), paths[:, :i_ae].min(1)
    win = paths[:, i_r0:i_r1]
    up = win > hi[:, None]
    dn = win < lo[:, None]
    first_up = np.where(up.any(1), up.argmax(1), 10**9)
    first_dn = np.where(dn.any(1), dn.argmax(1), 10**9)
    side = np.where((first_up < first_dn), 1, np.where(first_dn < first_up, -1, 0))
    rest = paths[:, i_r1:]
    ext_up, ext_dn = win.max(1), win.min(1)
    holds = np.where(side == 1, rest.max(1) <= ext_up, np.where(side == -1, rest.min(1) >= ext_dn, False))
    return side, holds


def a2(m1, D, start, end):
    real, null = [], []
    for d in business_days(start, end):
        td = sessions.trading_day(d)
        bars = _span(m1, td.start, td.end)
        if len(bars) < 1000:
            continue
        t = bars.index
        c = bars["close"].to_numpy()
        i_ae = int(t.searchsorted(sessions.asia_range(d).end))
        i_r0, i_r1 = max(i_ae, int(t.searchsorted(ny(d, 1)))), int(t.searchsorted(ny(d, 5)))
        if not (100 < i_ae <= i_r0 < i_r1 < len(c) - 100):
            continue
        s, h = hod_lod_stat(c[None, :], i_ae, i_r0, i_r1)
        real.append((s[0], h[0]))
        s, h = hod_lod_stat(flip_paths(c, DRAWS), i_ae, i_r0, i_r1)
        null.append((np.mean(s != 0), np.mean(h[s != 0]) if (s != 0).any() else np.nan))
    real = np.array(real, dtype=float)
    null = np.array(null, dtype=float)
    raided = real[:, 0] != 0
    p_real = real[raided, 1].mean()
    se = np.sqrt(p_real * (1 - p_real) / raided.sum())
    p_null = np.nanmean(null[:, 1])
    print(f"\n== A2 stop hunt de Londres -> HOD/LOD : {len(real)} jours, raid hors du range asiatique entre 01h et 05h NY "
          f"{raided.mean():.3f} des jours (hasard {null[:, 0].mean():.3f})")
    print(f"  l'extrême du raid à 05h NY reste le HOD/LOD jusqu'à 17h : réel {p_real:.3f} ± {se:.3f}, "
          f"hasard même volatilité {p_null:.3f}, écart {(p_real - p_null) / se:+.2f} écarts-types")


def a3(m1, D, start, end):
    d = D[(D.index >= start) & (D.index < end)].copy()
    d["r"] = (d["close"] - d["close"].shift(1)) / d["adr"]
    d = d.dropna(subset=["r"])
    r = d["r"].to_numpy()
    print(f"\n== A3 cycle de 3 jours : {len(r)} jours, rendement moyen {r.mean():+.3f} ADR")
    for k in (2, 3, 4):
        for sgn, name in ((-1, "baisses"), (1, "hausses")):
            run = np.ones(len(r), bool)
            for j in range(1, k + 1):
                run[k:] &= np.sign(r[k - j:len(r) - j]) == sgn
            run[:k] = False
            nxt = r[run]
            print(f"  après {k} {name} de suite : lendemain {nxt.mean():+.3f} ADR (t {nxt.mean() / nxt.std(ddof=1) * np.sqrt(len(nxt)):+.2f}, n {len(nxt)})")
    # jour du plus haut / plus bas de la semaine, réel contre signes tirés au hasard (clôtures M1)
    days = pd.Index(d.index)
    weeks = pd.Series(days, index=days).groupby(lambda x: (x - timedelta(days=x.weekday()))).apply(list)
    cnt_real = np.zeros((2, 5))
    cnt_null = np.zeros((2, 5))
    for wk, dl in weeks.items():
        if len(dl) != 5:
            continue
        w0, w1 = sessions.trading_day(dl[0]).start, sessions.trading_day(dl[-1]).end
        bars = _span(m1, w0, w1)
        if len(bars) < 5000:
            continue
        c = bars["close"].to_numpy()
        wd = np.array([x.weekday() for x in pd.Index(bars.index.tz_convert(sessions.NEW_YORK) + pd.Timedelta(hours=7)).date])
        wd = np.clip(wd, 0, 4)
        cnt_real[0, wd[c.argmax()]] += 1
        cnt_real[1, wd[c.argmin()]] += 1
        p = flip_paths(c, DRAWS)
        np.add.at(cnt_null[0], wd[p.argmax(1)], 1 / DRAWS)
        np.add.at(cnt_null[1], wd[p.argmin(1)], 1 / DRAWS)
    n = cnt_real[0].sum()
    print(f"  jour du plus haut / plus bas de la semaine ({int(n)} semaines complètes), réel puis hasard même volatilité :")
    for k, name in ((0, "plus haut"), (1, "plus bas")):
        print("   ", name, " ".join(f"{dn}:{cnt_real[k, i] / n:.2f}/{cnt_null[k, i] / n:.2f}" for i, dn in enumerate(("lun", "mar", "mer", "jeu", "ven"))))
    mid_r = (cnt_real[:, 1:3].sum()) / (2 * n)
    mid_n = (cnt_null[:, 1:3].sum()) / (2 * n)
    print(f"  extrême de la semaine un mardi ou un mercredi : réel {mid_r:.3f}, hasard {mid_n:.3f} "
          f"(écart {(mid_r - mid_n) / np.sqrt(mid_n * (1 - mid_n) / (2 * n)):+.2f} écarts-types, approx.)")


def main():
    start, end = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
    which = sys.argv[3] if len(sys.argv) > 3 else "123"
    m1 = load(start, end)
    D = daily(m1)
    if "1" in which:
        a1(m1, D, start, end)
    if "2" in which:
        a2(m1, D, start, end)
    if "3" in which:
        a3(m1, D, start, end)


main()
