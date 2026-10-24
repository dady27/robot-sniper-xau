"""Simulation d'un ordre limite puis de sa sortie (triple barrière : TP, SL ou temps).

Les bougies sont en prix bid. Un achat est rempli au prix ask (bid + spread) et sort au bid ;
une vente est remplie au bid et sort à l'ask. Quand une même bougie M1 touche le SL et le TP,
on compte le SL (hypothèse pessimiste). Sur la bougie de remplissage, seul le SL est vérifié.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyParams
from .signals import DayArrays, Order


@dataclass
class TradeResult:
    status: str                 # filled, cancelled_runaway, cancelled_fvg, cancelled_window, expired, no_data
    fill_time: pd.Timestamp | None = None
    exit_time: pd.Timestamp | None = None
    exit_reason: str | None = None  # tp, sl, crt, time, end_of_data
    exit_price: float | None = None
    r: float | None = None


def simulate(
    bars: pd.DataFrame,
    order: Order,
    window_end: pd.Timestamp,
    params: StrategyParams,
) -> TradeResult:
    s = order.sign
    a = DayArrays.from_frame(bars, s)
    t = a.times
    n = len(t)
    entry, sl, tp, d = s * order.entry, s * order.sl, s * order.tp, order.risk
    level, distal = s * order.level, s * order.fvg_distal
    fill_cost = a.spread if s == 1 else np.zeros(n)   # achat rempli à l'ask
    exit_cost = np.zeros(n) if s == 1 else a.spread   # vente clôturée à l'ask

    start = int(t.searchsorted(order.trigger_time))
    if start >= n:
        return TradeResult("no_data")

    # Juge CRT H1 : dernière bougie M1 de l'heure qui contient l'extrême
    crt_idx = None
    if params.use_crt_h1_exit:
        hour_end = order.extreme_time.floor("1h") + pd.Timedelta(hours=1)
        k = int(t.searchsorted(hour_end)) - 1
        crt_idx = k if k >= 0 else None
    exit_deadline = window_end + pd.Timedelta(minutes=params.exit_after_window_minutes)
    runaway = entry + params.cancel_at_r * d

    fill_idx = None
    for j in range(start, min(start + params.order_valid_m1_bars, n)):
        if t[j] >= window_end:
            return TradeResult("cancelled_window")
        if a.l[j] + fill_cost[j] <= entry:
            fill_idx = j
            break
        if a.h[j] >= runaway:
            return TradeResult("cancelled_runaway")
        if a.c[j] < distal:
            return TradeResult("cancelled_fvg")
    if fill_idx is None:
        return TradeResult("expired")

    fill_time = t[fill_idx]

    def done(j: int, reason: str, price: float) -> TradeResult:
        return TradeResult("filled", fill_time, t[j], reason, s * price, (price - entry) / d)

    if a.l[fill_idx] - exit_cost[fill_idx] <= sl:
        return done(fill_idx, "sl", sl - params.sl_slippage)

    for j in range(fill_idx + 1, n):
        if a.l[j] - exit_cost[j] <= sl:
            return done(j, "sl", sl - params.sl_slippage)
        if a.h[j] - exit_cost[j] >= tp:
            return done(j, "tp", tp)
        if crt_idx is not None and j == crt_idx and a.c[j] < level:
            return done(j, "crt", a.c[j] - exit_cost[j])
        if t[j] + pd.Timedelta(minutes=1) >= exit_deadline:
            return done(j, "time", a.c[j] - exit_cost[j])
    return done(n - 1, "end_of_data", a.c[n - 1] - exit_cost[n - 1])
