"""Marché aléatoire de contrôle.

Sur une marche aléatoire, aucun setup ne peut avoir d'avantage : un TP à 2,5R doit être
touché au plus 1 / (1 + 2,5) = 28,6 % du temps, un peu moins après les coûts. Un backtest
qui fait mieux sur ces données triche, en général en lisant une bougie pas encore clôturée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_m1(
    start: str,
    days: int,
    seed: int = 0,
    sigma_tick: float = 0.12,
    ticks_per_min: int = 30,
    spread: float = 0.30,
    price: float = 4000.0,
) -> pd.DataFrame:
    """Bougies M1 bid d'une marche aléatoire, aux heures d'ouverture de l'or (dim. 22:00 à ven. 21:00 UTC)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    dow, hour = idx.dayofweek, idx.hour
    idx = idx[((dow < 5) | ((dow == 6) & (hour >= 22))) & ~((dow == 4) & (hour >= 21))]
    n = len(idx)
    path = price + np.cumsum(rng.normal(0, sigma_tick, n * ticks_per_min)).reshape(n, ticks_per_min)
    o = np.r_[price, path[:-1, -1]]
    return pd.DataFrame({
        "open": o,
        "high": np.maximum(path.max(1), o),
        "low": np.minimum(path.min(1), o),
        "close": path[:, -1],
        "spread": spread,
        "ticks": ticks_per_min,
    }, index=idx)
