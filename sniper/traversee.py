"""Traversée des niveaux où les stops s'accumulent : le prix les traverse-t-il ?

Niveaux, par famille :
- ronds : multiples de 50 $ ;
- veille : plus haut et plus bas du jour de trading précédent (18h-17h New York) ;
- semaine : plus haut et plus bas de la semaine précédente.

Chaque jour ouvré, entre 01h et 16h New York, on attend le premier contact de chaque niveau.
Le prix de 01h doit être à au moins 0,25 ATR H1 de l'autre côté du niveau ; un niveau de la
veille ou de la semaine doit être resté intact depuis sa formation. Au contact, ordre stop dans
le sens de la traversée (achat à l'ask au-dessus, vente au bid en dessous), stop à 0,5 ATR H1
de l'entrée, sortie à 16h New York. Un seul trade par minute et par sens, toutes familles confondues.

Contrôle neutre comme events.py : au même instant, même risque, achat et vente au marché.
Mesuré sur 2015-2023 : +0,16R par trade à 2,5R ; sur 2024-2026 : +0,09R. L'ordre stop rempli
pile au niveau est optimiste : `slippage_atr_h1` et `entry="close"` mesurent ce qu'il en reste.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from . import sessions
from .backtest import AtrBook
from .events import follow, outcome_r

FAMILIES = ("ronds", "veille", "semaine")


@dataclass(frozen=True)
class CrossParams:
    families: tuple[str, ...] = FAMILIES
    step: float = 50.0                  # pas des chiffres ronds, en dollars
    offset: float = 0.0                 # décalage de la grille : contrôle sur des niveaux non ronds
    min_dist_atr_h1: float = 0.25       # distance minimale au niveau à 01h New York
    stop_atr_h1: float = 0.5
    entry: str = "stop"                 # stop : rempli au niveau ; close : au marché à la clôture de la minute du contact
    slippage_atr_h1: float = 0.0        # glissement de l'ordre stop
    start_ny: time = time(1, 0)
    end_ny: time = time(16, 0)
    targets_r: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def _ny(day: date, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, t, tzinfo=sessions.NEW_YORK)).tz_convert("UTC")


def _previous_week(day: date) -> sessions.Window:
    monday = day - timedelta(days=day.weekday())
    return sessions.Window("previous_week", sessions.trading_day(monday - timedelta(days=7)).start,
                           sessions.trading_day(monday - timedelta(days=3)).end)


class _Prices:
    """Tableaux M1 en espace « achat » pour chaque sens : une vente se traite en retournant les prix."""

    def __init__(self, m1: pd.DataFrame):
        self.t = m1.index
        O, H, L, C, SP = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "spread"))
        zero = np.zeros(len(m1))
        self.spread = SP
        self.space = {1: (O, H, L, C, SP, zero), -1: (-O, -L, -H, -C, zero, SP)}

    def idx(self, ts: pd.Timestamp) -> int:
        return int(self.t.searchsorted(ts))


def _candidates(px: _Prices, day: date, i_a: int, i_b: int, a_h1: float, p: CrossParams):
    """(famille, sens, niveau) des niveaux jouables ce jour-là."""
    _, H, L, _, _, _ = px.space[1]
    o_ref = px.space[1][0][i_a]
    out = []
    if "ronds" in p.families:
        lo = np.floor((L[i_a:i_b].min() - p.offset) / p.step) * p.step + p.offset
        for lvl in np.arange(lo, H[i_a:i_b].max() + p.step, p.step):
            sign = 1 if o_ref < lvl - p.min_dist_atr_h1 * a_h1 else (-1 if o_ref > lvl + p.min_dist_atr_h1 * a_h1 else 0)
            if sign:
                out.append(("ronds", sign, float(lvl)))
    for fam, ref in (("veille", sessions.previous_trading_day(day)), ("semaine", _previous_week(day))):
        if fam not in p.families:
            continue
        r0, r1 = px.idx(ref.start), px.idx(ref.end)
        if r1 - r0 < 500:
            continue
        for sign, lvl in ((1, H[r0:r1].max()), (-1, L[r0:r1].min())):
            h = px.space[sign][1]
            if h[r1:i_a].max(initial=-np.inf) >= sign * lvl:  # déjà touché avant 01h
                continue
            if sign * o_ref < sign * lvl - p.min_dist_atr_h1 * a_h1:
                out.append((fam, sign, float(lvl)))
    return out


def run_cross(m1: pd.DataFrame, p: CrossParams = CrossParams(),
              start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Un trade par premier contact de niveau, pour les jours ouvrés entre `start` et `end` (exclu)."""
    atr_h1 = AtrBook(m1).fn("h1")
    px = _Prices(m1)
    day = start or (m1.index[0].date() + timedelta(days=8))
    last = end or m1.index[-1].date()
    rows: list[dict] = []
    while day < last:
        if day.weekday() < 5:
            rows.extend(_day_trades(px, atr_h1, day, p))
        day += timedelta(days=1)
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    return ev.sort_values("time", kind="stable").drop_duplicates(subset=["time", "side"]).reset_index(drop=True)


def _day_trades(px: _Prices, atr_h1, day: date, p: CrossParams) -> list[dict]:
    i_a, i_b = px.idx(_ny(day, p.start_ny)), px.idx(_ny(day, p.end_ny))
    if i_b - i_a < 600 or i_a == 0:
        return []
    a = atr_h1(px.t[i_a])
    if not (np.isfinite(a) and a > 0):
        return []
    rows = []
    risk = p.stop_atr_h1 * a
    for fam, sign, lvl in _candidates(px, day, i_a, i_b, a, p):
        o, h, l, c, ec, xc = px.space[sign]
        touch = np.nonzero(h[i_a:i_b] >= sign * lvl)[0]
        if len(touch) == 0:
            continue
        i0 = i_a + int(touch[0])
        if p.entry == "stop":
            entry = max(sign * lvl, o[i0]) + ec[i0] + p.slippage_atr_h1 * a  # un gap au-delà du niveau s'exécute à l'ouverture
        else:
            entry = c[i0] + ec[i0]
        mfe, stopped, r_end, _ = follow(h, l, c, xc, i0 + 1, i_b, entry, entry - risk, {})
        ctrl = []
        for s2 in (1, -1):
            _, h2, l2, c2, ec2, xc2 = px.space[s2]
            e2 = c2[i0] + ec2[i0]
            ctrl.append(follow(h2, l2, c2, xc2, i0 + 1, i_b, e2, e2 - risk, {}))
        row = {
            "day": day, "famille": fam, "level": lvl, "side": "long" if sign == 1 else "short",
            "time": px.t[i0], "hour_ny": px.t[i0].tz_convert(sessions.NEW_YORK).hour,
            "entry": sign * entry, "risk": risk, "spread_r": px.spread[i0] / risk,
            "mfe": mfe, "stopped": stopped, "r_end": r_end,
            "hit_opp": False, "opp_r": np.nan, "r_opp": np.nan, "c_opp": np.nan,
        }
        for x in p.targets_r:
            row[f"r{x:g}"] = outcome_r(mfe, stopped, r_end, x)
            row[f"c{x:g}"] = np.mean([outcome_r(m, s, r, x) for m, s, r, _ in ctrl])
        rows.append(row)
    return rows
