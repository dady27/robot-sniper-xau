"""Sir Jones / BTMM : raid du range asiatique à l'ouverture de Londres, puis M ou W, entrée sur la 2e jambe.

Règles du document (pips EURUSD convertis en ADR, 75 pips ~ 1 ADR) :
- range asiatique valide < 50 pips -> largeur <= 0,67 ADR ;
- stop hunt commencé entre 01h et 05h New York ;
- M ou W de 30 à 90 min : la 2e jambe revient vers le HOD/LOD sans le casser en clôture M15 ;
- stop 7 à 10 pips au-delà du HOD/LOD -> 0,12 ADR ; cible 50 pips -> 0,67 ADR ;
- stop hunt prolongé : si l'extrême est cassé, on attend un nouveau M/W plus loin.
Paramètres ajoutés (absents du document) : repli du milieu du M >= 0,15 ADR, la 2e jambe
revient à 0,10 ADR de l'extrême, déclencheur = première bougie M15 qui clôture dans le sens
du retournement, au plus 90 min après l'extrême, au plus tard 07h New York.
usage: h_btmm.py START END OUT [asia_max] [pull] [tol] [buf]
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from lab import PIP_ADR, Space, _span, business_days, by_year, daily, evaluate, load, ny, sessions, show


def btmm_day(m1, D, day, asia_max=50 * PIP_ADR, pull=0.15, tol=0.10, buf=8.5 * PIP_ADR, max_min=90):
    if day not in D.index or not np.isfinite(D.at[day, "adr"]):
        return []
    adr = D.at[day, "adr"]
    asia = sessions.asia_range(day)
    bars = _span(m1, asia.start, ny(day, 16))
    sp = Space(bars)
    if sp.n < 600:
        return []
    i_ae = sp.idx(asia.end)
    i_r0, i_r1, i_te = max(i_ae, sp.idx(ny(day, 1))), sp.idx(ny(day, 5)), sp.idx(ny(day, 7))
    O, H, L, C = sp.raw
    if i_ae < 200:
        return []
    ah, al = H[:i_ae].max(), L[:i_ae].min()
    if ah - al > asia_max * adr:
        return []
    t = sp.t
    b = t.floor("15min").asi8
    m15_end = np.r_[b[1:] != b[:-1], True]
    m15_start = np.r_[True, b[1:] != b[:-1]]
    rows = []
    for sign in (1, -1):
        o, h, l, c, ec, xc = sp.sp[sign]
        level = sign * (al if sign == 1 else ah)
        hit = np.nonzero(l[i_r0:i_r1] < level)[0]
        if len(hit) == 0:
            continue
        i_sw = i_r0 + int(hit[0])
        e1, i_e1 = l[i_sw], i_sw
        state, l2 = "leg1", np.inf
        m15_open = o[i_sw]
        n_reset = 0
        for j in range(i_sw, i_te):
            if m15_start[j]:
                m15_open = o[j]
            if state == "leg1":
                if l[j] < e1:
                    e1, i_e1 = l[j], j
                if h[j] >= e1 + pull * adr:
                    state = "pullback"
                continue
            # pullback / leg2 : invalidé si le prix casse l'extrême (stop) ou clôture M15 au-delà
            if l[j] < e1 - buf * adr or (m15_end[j] and c[j] < e1):
                e1, i_e1 = min(e1, l[j]), (j if l[j] < e1 else i_e1)
                state, l2, n_reset = "leg1", np.inf, n_reset + 1
                continue
            if (t[j] - t[i_e1]).total_seconds() > max_min * 60:
                break
            if state == "pullback" and l[j] <= e1 + tol * adr:
                state = "leg2"
            if state == "leg2":
                l2 = min(l2, l[j])
                if m15_end[j] and c[j] > m15_open:
                    stop = min(e1, l2) - buf * adr
                    entry = c[j] + ec[j]
                    row = evaluate(sp, j, sign, entry - stop, sp.n,
                                   extra={"50p": entry + 50 * PIP_ADR * adr, "opp": sign * (ah if sign == 1 else al)})
                    if row:
                        row.update({"day": day, "level": "AL" if sign == 1 else "AH",
                                    "depth_adr": (level - e1) / adr, "asia_adr": (ah - al) / adr,
                                    "risk_adr": (entry - stop) / adr, "m_minutes": (t[j] - t[i_e1]).total_seconds() / 60,
                                    "resets": n_reset, "hour_ny": t[j].tz_convert(sessions.NEW_YORK).hour})
                        rows.append(row)
                    break
    return rows


def main():
    start, end, out = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), sys.argv[3]
    kw = {}
    for k, v in zip(("asia_max", "pull", "tol", "buf"), sys.argv[4:]):
        kw[k] = float(v)
    m1 = load(start, end)
    D = daily(m1)
    rows = []
    for d in business_days(start, end):
        rows += btmm_day(m1, D, d, **kw)
    ev = pd.DataFrame(rows)
    ev.to_csv(out, index=False)
    show(ev, f"BTMM M/W 2e jambe {start}..{end} {kw}", extra=("50p", "opp"))
    if not ev.empty:
        print("écart 2,5R par année :", by_year(ev))
        print("écart 50 pips par année :", by_year(ev, "r_50p", "c_50p"))
        print("risque médian", round(ev.risk_adr.median(), 3), "ADR ; profondeur médiane", round(ev.depth_adr.median(), 3),
              "ADR ; durée M médiane", ev.m_minutes.median(), "min ; resets>0 :", int((ev.resets > 0).sum()))
        deep = ev[ev.depth_adr >= 25 * PIP_ADR]
        show(deep, "raid d'au moins 25 pips (0,33 ADR)", extra=("50p",))


main()
