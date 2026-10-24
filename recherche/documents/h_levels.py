"""Niveaux « aimants » et « points de retournement » des documents, par premier passage :

B3. Equal highs / equal lows (Sir Jones : « price will always be attracted by equal highs and lows ») :
    depuis la confirmation du 2e sommet, le prix touche-t-il le niveau avant de reculer d'autant ?
    Comparé aux sommets simples (sans jumeau) à la même distance. Marche au hasard : 50 %.
B4. Chiffres ronds (BFX : niveaux institutionnels 00, 20, 50, 80) : au premier contact du jour,
    le prix se retourne-t-il de d avant de continuer de d ? Comparé à des niveaux non ronds.
usage: h_levels.py START END [b3|b4]
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from lab import AtrBook, business_days, load, ny

H1 = 60


def first_passage(hi, lo, i0, up, dn, horizon):
    """Premier franchissement de `up` (par le haut) ou `dn` (par le bas) après la bougie i0.

    Retourne 1 (haut d'abord), 0 (bas d'abord), nan (aucun ou les deux dans la même bougie M1).
    """
    a, b = i0 + 1, min(i0 + 1 + horizon, len(hi))
    if a >= b:
        return np.nan
    u = hi[a:b] >= up
    d = lo[a:b] <= dn
    iu = int(u.argmax()) if u.any() else 10**9
    idn = int(d.argmax()) if d.any() else 10**9
    if iu == idn:
        return np.nan
    return 1.0 if iu < idn else 0.0


def m15(m1):
    g = m1.resample("15min", label="left", closed="left")
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    return b.dropna()


def b3(m1, start, end, tol=0.10, look=96, horizon=1440):
    atr_h1 = AtrBook(m1).fn("h1")
    bars = m15(m1)
    bars = bars[(bars.index >= pd.Timestamp(start, tz="UTC")) & (bars.index < pd.Timestamp(end, tz="UTC"))]
    H, L, C = (bars[k].to_numpy() for k in ("high", "low", "close"))
    t15 = bars.index
    mt = m1.index
    mh, ml = m1["high"].to_numpy(), m1["low"].to_numpy()
    rows = []
    for sign in (1, -1):
        h = H if sign == 1 else -L
        swings = []  # (index, prix) des sommets encore intacts
        for k in range(2, len(h) - 2):
            if not (h[k] > h[k - 1] and h[k] > h[k - 2] and h[k] >= h[k + 1] and h[k] >= h[k + 2]):
                continue
            kc = k + 2  # confirmé à la clôture de la bougie k+2
            a = atr_h1(t15[kc])
            if not np.isfinite(a) or a <= 0:
                continue
            # sommets précédents des dernières 24 h, intacts : rien au-dessus d'eux avant le nouveau sommet
            swings = [(j, pj) for (j, pj) in swings if j >= k - look]
            twin = [pj for (j, pj) in swings
                    if k - j >= 3 and h[j + 1:k].max() <= pj and abs(pj - h[k]) <= tol * a]
            swings.append((k, h[k]))
            level = max([h[k]] + twin)
            p0 = sign * C[kc]
            dist = level - p0
            if not (0.25 * a <= dist <= 2.0 * a):
                continue
            i0 = int(mt.searchsorted(t15[kc] + pd.Timedelta(minutes=15))) - 1
            hi, lo = (mh, ml) if sign == 1 else (-ml, -mh)
            res = first_passage(hi, lo, i0, level, p0 - dist, horizon)
            rows.append({"time": t15[kc], "side": sign, "equal": bool(twin), "dist_atr": dist / a, "magnet": res})
    ev = pd.DataFrame(rows).dropna(subset=["magnet"])
    print(f"\n== B3 equal highs/lows comme aimant : {len(ev)} niveaux (M15, tolérance {tol} ATR H1, distance 0,25-2 ATR H1)")
    for eq, g in ev.groupby("equal"):
        p = g["magnet"].mean()
        se = np.sqrt(p * (1 - p) / len(g))
        print(f"  {'equal highs/lows' if eq else 'sommets simples'} : niveau touché avant un recul égal {p:.3f} ± {se:.3f} (n {len(g)})")
    for lo_, hi_ in ((0.25, 0.75), (0.75, 1.25), (1.25, 2.0)):
        g = ev[(ev.dist_atr >= lo_) & (ev.dist_atr < hi_)]
        e, s = g[g.equal]["magnet"], g[~g.equal]["magnet"]
        print(f"    distance {lo_}-{hi_} ATR : equal {e.mean():.3f} (n {len(e)}), simples {s.mean():.3f} (n {len(s)})")
    return ev


def b4(m1, start, end, step=10.0, d_atr=0.5, horizon=1440):
    atr_h1 = AtrBook(m1).fn("h1")
    mt = m1.index
    mh, ml, mo = m1["high"].to_numpy(), m1["low"].to_numpy(), m1["open"].to_numpy()
    rows = []
    offsets = {"rond 10": 0.0, "non rond +3,33": step / 3, "non rond +6,67": 2 * step / 3}
    for d in business_days(start, end):
        i_a, i_b = int(mt.searchsorted(ny(d, 1))), int(mt.searchsorted(ny(d, 16)))
        if i_b - i_a < 600:
            continue
        a = atr_h1(mt[i_a])
        if not np.isfinite(a) or a <= 0:
            continue
        p_open = mo[i_a]
        dist = d_atr * a
        for name, off in offsets.items():
            base = np.floor((ml[i_a:i_b].min() - off) / step) * step + off
            for lvl in np.arange(base, mh[i_a:i_b].max() + step, step):
                for sign in (1, -1):
                    # contact par le haut (sign=1, prix venant d'en dessous) ou par le bas
                    if sign == 1 and not (p_open < lvl - 0.25 * a):
                        continue
                    if sign == -1 and not (p_open > lvl + 0.25 * a):
                        continue
                    touch = (mh[i_a:i_b] >= lvl) if sign == 1 else (ml[i_a:i_b] <= lvl)
                    if not touch.any():
                        continue
                    i0 = i_a + int(touch.argmax())
                    hi, lo = (mh, ml) if sign == 1 else (-ml, -mh)
                    L0 = sign * lvl
                    res = first_passage(hi, lo, i0, L0 + dist, L0 - dist, horizon)
                    big = "100" if abs(lvl / 100 - round(lvl / 100)) < 1e-9 else ("50" if abs(lvl / 50 - round(lvl / 50)) < 1e-9 else "")
                    rows.append({"day": d, "kind": name, "big": big, "side": sign, "level": lvl,
                                 "reversal": np.nan if np.isnan(res) else 1 - res})
    ev = pd.DataFrame(rows).dropna(subset=["reversal"])
    print(f"\n== B4 chiffres ronds (pas de {step:g} $) : premier contact du jour entre 01h et 16h NY, "
          f"retour de {d_atr} ATR H1 avant une continuation de {d_atr} ATR H1 (hasard 0,5)")
    for k, g in ev.groupby("kind"):
        p = g["reversal"].mean()
        print(f"  {k} : retournement {p:.3f} ± {np.sqrt(p * (1 - p) / len(g)):.3f} (n {len(g)})")
    r = ev[ev.kind == "rond 10"]
    for big, g in r.groupby("big"):
        p = g["reversal"].mean()
        print(f"    multiples de {big or '10 seulement'} : {p:.3f} ± {np.sqrt(p * (1 - p) / len(g)):.3f} (n {len(g)})")
    return ev


def main():
    start, end = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
    which = sys.argv[3] if len(sys.argv) > 3 else "b3b4"
    m1 = load(start, end)
    if "b3" in which:
        b3(m1, start, end)
    if "b4" in which:
        b4(m1, start, end)


main()
