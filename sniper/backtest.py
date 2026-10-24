"""Rejoue les modèles A (reversal Londres) et B (reversal New York) sur un historique M1.

Un trade au plus par killzone : parmi les niveaux candidats, on garde le premier setup
complet. Chaque ligne du résultat décrit un setup, son ordre et ce que l'ordre est devenu.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import sessions
from .config import DEFAULT_PARAMS, StrategyParams
from .data import atr_known_at, resample, value_at
from .signals import Order, detect
from .simulate import simulate


class AtrBook:
    """ATR M1, M5 et H1, interrogeables à un instant donné sans regarder le futur."""

    def __init__(self, m1: pd.DataFrame, period: int = 14):
        self.m1 = atr_known_at(m1, 1, period)
        self.m5 = atr_known_at(resample(m1, "5min"), 5, period)
        self.h1 = atr_known_at(resample(m1, "1h"), 60, period)

    def fn(self, which: str):
        series = getattr(self, which)
        return lambda t: value_at(series, t)


def _hl(m1: pd.DataFrame, w: sessions.Window) -> tuple[float, float]:
    sl = m1.loc[(m1.index >= w.start) & (m1.index < w.end)]
    if sl.empty:
        return float("nan"), float("nan")
    return float(sl["high"].max()), float(sl["low"].min())


def _untouched(m1: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp, level: float, side: str) -> bool:
    """Vrai si le niveau n'a pas été balayé entre `since` et `until`."""
    sl = m1.loc[(m1.index >= since) & (m1.index < until)]
    if sl.empty:
        return True
    return bool(sl["low"].min() >= level) if side == "low" else bool(sl["high"].max() <= level)


def day_levels(m1: pd.DataFrame, day: date, atr_h1_at, params: StrategyParams) -> dict:
    """Niveaux de liquidité du jour, avec l'heure à partir de laquelle chacun est connu."""
    asia, lses = sessions.asia_range(day), sessions.london_session(day)
    prev = sessions.previous_trading_day(day)
    td = sessions.trading_day(day)
    pdh, pdl = _hl(m1, prev)
    ah, al = _hl(m1, asia)
    lh, ll = _hl(m1, lses)
    levels = {
        "PDH": ("high", pdh, td.start), "PDL": ("low", pdl, td.start),
        "LH": ("high", lh, lses.end), "LL": ("low", ll, lses.end),
    }
    a_h1 = atr_h1_at(asia.end)
    width = ah - al
    if np.isfinite(width) and np.isfinite(a_h1) and a_h1 > 0 and \
            params.asia_width_min_atr_h1 <= width / a_h1 <= params.asia_width_max_atr_h1:
        levels["AH"] = ("high", ah, asia.end)
        levels["AL"] = ("low", al, asia.end)
    return levels


MODELS = {
    "A_london": (sessions.london_killzone, ("AH", "AL", "PDH", "PDL")),
    "B_newyork": (sessions.ny_killzone, ("LH", "LL", "AH", "AL", "PDH", "PDL")),
}


def _span(m1: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return m1.iloc[int(m1.index.searchsorted(start)):int(m1.index.searchsorted(end))]


def run_day(m1: pd.DataFrame, day: date, atrs: AtrBook, params: StrategyParams = DEFAULT_PARAMS) -> list[dict]:
    """Setups du jour. `m1` peut être l'historique complet : seule la tranche utile est lue."""
    rows = []
    a1, a5, ah = atrs.fn("m1"), atrs.fn("m5"), atrs.fn("h1")
    horizon = pd.Timedelta(minutes=params.exit_after_window_minutes + 5)
    m1 = _span(m1, sessions.previous_trading_day(day).start, sessions.ny_killzone(day).end + horizon)
    levels = day_levels(m1, day, ah, params)
    td = sessions.trading_day(day)
    for model, (window_fn, names) in MODELS.items():
        w = window_fn(day)
        bars = _span(m1, td.start, w.end + horizon)
        if bars.empty or bars.index[-1] < w.start:
            continue
        candidates: list[Order] = []
        for name in names:
            if name not in levels:
                continue
            kind, price, known_from = levels[name]
            if not np.isfinite(price) or not _untouched(m1, known_from, w.start, price, kind):
                continue
            side = "long" if kind == "low" else "short"
            opposite = [p for (k, p, _) in levels.values() if k != kind and np.isfinite(p)]
            order = detect(bars, side, name, price, w.start, w.end, a1, a5, ah, params, opposite)
            if order is not None:
                candidates.append(order)
        if not candidates:
            continue
        order = min(candidates, key=lambda o: o.trigger_time)
        res = simulate(bars, order, w.end, params)
        row = {"day": day, "model": model}
        row.update({k: v for k, v in asdict(order).items() if k != "features"})
        row.update(order.features)
        row.update(asdict(res))
        rows.append(row)
    return rows


def run(m1: pd.DataFrame, params: StrategyParams = DEFAULT_PARAMS,
        start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Rejoue tous les jours ouvrés entre `start` et `end` (exclu)."""
    atrs = AtrBook(m1, params.atr_period)
    first = start or (m1.index[0].date() + timedelta(days=2))
    last = end or m1.index[-1].date()
    rows: list[dict] = []
    day = first
    while day < last:
        if day.weekday() < 5:
            rows.extend(run_day(m1, day, atrs, params))
        day += timedelta(days=1)
    return pd.DataFrame(rows)
