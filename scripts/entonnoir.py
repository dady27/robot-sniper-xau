"""Entonnoir des modèles A et B : à quelle règle chaque niveau candidat est éliminé.

    python scripts/entonnoir.py --m1-dir data/m1 --start 2015-02-01 --end 2024-01-01

Reprend pas à pas signals.detect_long et la boucle de backtest.run_day, en notant la
première règle qui rejette le niveau. À garder aligné avec signals.detect_long.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_backtest import load_m1_dir  # noqa: E402
from sniper import sessions  # noqa: E402
from sniper import signals as sg  # noqa: E402
from sniper.backtest import MODELS, AtrBook, _span, _untouched, day_levels  # noqa: E402
from sniper.config import DEFAULT_PARAMS, StrategyParams  # noqa: E402


def _waiting(reclaim_idx, cisd_idx) -> str:
    return "sans réintégration" if reclaim_idx is None else ("sans CISD" if cisd_idx is None else "sans FVG")


def rejection(a: sg.DayArrays, i0: int, i1: int, level: float, atr_m1, atr_m5, atr_h1,
              params: StrategyParams, obstacles=()) -> str:
    """Même logique que signals.detect_long, mais renvoie la raison du rejet (ou « setup »)."""
    t = a.times
    sweep_idx = next((i for i in range(i0, i1) if a.l[i] < level - params.sweep_min_atr_m5 * atr_m5(t[i])), None)
    if sweep_idx is None:
        return "pas de sweep"
    a1, a5, ah = atr_m1(t[sweep_idx]), atr_m5(t[sweep_idx]), atr_h1(t[sweep_idx])
    if not all(np.isfinite(v) and v > 0 for v in (a1, a5, ah)):
        return "ATR inconnu"
    ext_idx, ext = sweep_idx, a.l[sweep_idx]
    cisd = sg._cisd_level(a, ext_idx)
    reclaim_idx = cisd_idx = None
    for j in range(sweep_idx, i1):
        if a.l[j] < ext:
            ext_idx, ext = j, a.l[j]
            cisd = sg._cisd_level(a, ext_idx)
            reclaim_idx = cisd_idx = None
        if level - ext > params.max_excursion_atr_h1 * ah:
            return "excursion > 0,5 ATR H1"
        if (t[j] - t[ext_idx]) > pd.Timedelta(minutes=params.trigger_max_minutes):
            return "délai de 30 min dépassé, " + _waiting(reclaim_idx, cisd_idx)
        if (reclaim_idx is None and sg._is_m5_close(t[j]) and a.c[j] > level
                and sg._m5_bucket(t[j]) - sg._m5_bucket(t[ext_idx]) < params.reclaim_max_m5_bars):
            reclaim_idx = j
        if cisd_idx is None and j > ext_idx and a.c[j] > cisd:
            cisd_idx = j
        if reclaim_idx is None or cisd_idx is None:
            continue
        fvg = next(((float(a.l[k]), float(a.h[k - 2])) for k in range(ext_idx + 2, j + 1)
                    if a.l[k] - a.h[k - 2] >= params.fvg_min_atr_m1 * a1), None)
        if fvg is None:
            continue
        proximal, distal = fvg
        entry = proximal - params.entry_depth * (proximal - distal)
        sl = ext - max(params.sl_buffer_spread_mult * float(a.spread[j]), params.sl_buffer_atr_m5 * a5)
        d = entry - sl
        if d < params.sl_min_atr_m1 * a1:
            sl = entry - params.sl_min_atr_m1 * a1
            d = entry - sl
        if d > params.sl_max_atr_h1 * ah:
            entry = (proximal + distal) / 2
            if entry - sl > params.sl_max_atr_h1 * ah:
                return "SL > 0,35 ATR H1"
            d = entry - sl
        above = [x for x in obstacles if np.isfinite(x) and x > entry]
        if above and (min(above) - entry) / d < params.room_min_r:
            return "niveau opposé à moins de 2,5R"
        return "setup"
    return "fin de killzone, " + _waiting(reclaim_idx, cisd_idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-dir", type=Path, required=True)
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    a = ap.parse_args()

    p = DEFAULT_PARAMS
    m1 = load_m1_dir(a.m1_dir)
    m1 = m1[(m1.index >= pd.Timestamp(a.start - timedelta(days=30), tz="UTC")) & (m1.index < pd.Timestamp(a.end, tz="UTC"))]
    atrs = AtrBook(m1, p.atr_period)
    a1, a5, ah = atrs.fn("m1"), atrs.fn("m5"), atrs.fn("h1")
    horizon = pd.Timedelta(minutes=p.exit_after_window_minutes + 5)
    counts: Counter = Counter()
    days = asia_off = 0
    day = a.start
    while day < a.end:
        if day.weekday() < 5:
            mm = _span(m1, sessions.previous_trading_day(day).start, sessions.ny_killzone(day).end + horizon)
            if len(mm) > 500:
                days += 1
                levels = day_levels(mm, day, ah, p)
                asia_off += "AH" not in levels
                td = sessions.trading_day(day)
                for model, (window_fn, names) in MODELS.items():
                    w = window_fn(day)
                    bars = _span(mm, td.start, w.end + horizon)
                    for name in (n for n in names if n in levels):
                        kind, price, known_from = levels[name]
                        if not np.isfinite(price):
                            reason = "niveau inconnu"
                        elif not _untouched(mm, known_from, w.start, price, kind):
                            reason = "niveau déjà pris avant la killzone"
                        else:
                            sign = 1 if kind == "low" else -1
                            arr = sg.DayArrays.from_frame(bars, sign)
                            i0, i1 = int(arr.times.searchsorted(w.start)), int(arr.times.searchsorted(w.end))
                            opp = [sign * x for (k, x, _) in levels.values() if k != kind and np.isfinite(x)]
                            reason = rejection(arr, i0, i1, sign * price, a1, a5, ah, p, opp)
                        counts[(model, reason)] += 1
        day += timedelta(days=1)
    print(f"{days} jours ; range asiatique hors largeur (AH/AL absents) : {asia_off} jours")
    table = pd.Series(counts).unstack(0).fillna(0).astype(int)
    print(table.sort_values(table.columns[0], ascending=False).to_string())


if __name__ == "__main__":
    main()
