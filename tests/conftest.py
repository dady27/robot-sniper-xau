import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sniper.config import StrategyParams  # noqa: E402

# Jour d'hiver : Londres = UTC. Les bougies 0 à 9 précèdent la killzone, la bougie 10 ouvre à 07:00.
T0 = pd.Timestamp("2026-01-14 06:50", tz="UTC")

# (open, high, low, close) : un sweep du niveau 4000 puis une remontée qui laisse un FVG.
LONG_SETUP = [
    *[(4003.0, 4003.6, 4002.6, 4003.2)] * 9,
    (4003.2, 4004.2, 4003.0, 4004.0),   # 9  haussière : la série de livraison commence après
    (4004.0, 4004.2, 4002.3, 4002.5),   # 10 07:00 baissière
    (4002.5, 4002.6, 4001.0, 4001.2),   # 11
    (4001.2, 4001.3, 4000.1, 4000.3),   # 12
    (4000.3, 4000.4, 3999.0, 3999.4),   # 13 sweep (3999.0 < 4000 - 0,2)
    (3999.4, 3999.6, 3998.5, 3998.9),   # 14 07:04 extrême 3998.5
    (3998.9, 4000.8, 3998.8, 4000.6),   # 15 déplacement
    (4000.6, 4002.6, 4000.5, 4002.4),   # 16 FVG : Low 4000.5 > High[14] 3999.6
    (4002.4, 4003.3, 4002.2, 4003.0),   # 17
    (4003.0, 4004.5, 4002.9, 4004.3),   # 18 CISD : clôture > 4004.0 (ouverture de la bougie 10)
    (4004.3, 4004.6, 4003.8, 4004.0),   # 19 07:09 clôture M5 au-dessus de 4000 : réintégration
    (4004.0, 4004.1, 4001.8, 4002.0),   # 20
    (4002.0, 4002.1, 3999.9, 4000.6),   # 21 remplissage (ask = 3999.9 + 0,3 <= entrée)
    (4000.6, 4003.2, 4000.4, 4003.0),   # 22
    (4003.0, 4006.6, 4002.9, 4006.5),   # 23 TP touché
    *[(4006.5, 4006.8, 4006.2, 4006.6)] * 20,
]


def make_bars(rows, t0=T0, spread=0.30):
    idx = pd.date_range(t0, periods=len(rows), freq="1min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["spread"] = spread
    df["ticks"] = 10
    return df


def mirror(df, axis=8000.0):
    """Retourne les prix autour de axis/2 : un setup d'achat devient un setup de vente."""
    out = df.copy()
    out["open"], out["close"] = axis - df["open"], axis - df["close"]
    out["high"], out["low"] = axis - df["low"], axis - df["high"]
    return out


@pytest.fixture
def params():
    return StrategyParams()


@pytest.fixture
def const_atr():
    return (lambda t: 1.0), (lambda t: 2.0), (lambda t: 10.0)
