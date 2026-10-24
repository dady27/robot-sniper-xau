"""Détection du setup Sweep · Reclaim · CISD · FVG.

Toute la logique est écrite pour un achat (balayage d'un niveau bas). Une vente est
traitée en retournant les prix (x -> -x) : le plus haut devient le plus bas, et le
même code s'applique. Cela évite deux implémentations miroir qui divergent.

Séquence pour un achat sur le niveau L :
1. Sweep : une bougie M1 passe sous L d'au moins 0,1 x ATR(14) M5.
2. L'excès sous L reste inférieur à 0,5 x ATR(14) H1, sinon setup annulé.
3. Réintégration : une bougie M5 clôture au-dessus de L dans les 3 bougies M5
   qui suivent l'extrême (la bougie M5 de l'extrême compte comme la première).
4. CISD : une bougie M1 clôture au-dessus de l'ouverture de la série de bougies
   baissières qui a amené le prix à l'extrême.
5. FVG haussier dans la jambe de remontée (Low[k] > High[k-2]), taille >= 0,15 x ATR M1.
3, 4 et 5 doivent être réunis au plus 30 min après l'extrême. Un nouvel extrême
remet les compteurs à zéro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import StrategyParams

AtrFn = Callable[[pd.Timestamp], float]


@dataclass
class DayArrays:
    """Bougies M1 d'une journée sous forme de tableaux, éventuellement retournées pour une vente."""

    times: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    spread: np.ndarray
    sign: int = 1  # +1 achat, -1 vente (prix retournés)

    @classmethod
    def from_frame(cls, bars: pd.DataFrame, sign: int = 1) -> "DayArrays":
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        spread = bars["spread"].to_numpy(float) if "spread" in bars else np.zeros(len(bars))
        spread = np.where(np.isnan(spread), np.nanmedian(spread) if np.isfinite(spread).any() else 0.0, spread)
        if sign == -1:
            o, h, l, c = -o, -l, -h, -c
        return cls(bars.index, o, h, l, c, spread, sign)


@dataclass
class Order:
    """Ordre limite produit par un setup, en prix réels."""

    side: str
    level_name: str
    level: float
    sweep_time: pd.Timestamp
    extreme_time: pd.Timestamp
    extreme: float
    trigger_time: pd.Timestamp  # clôture de la bougie qui complète le setup
    trigger_idx: int
    extreme_idx: int
    entry: float
    sl: float
    tp: float
    risk: float                 # distance entrée - SL, en dollars
    fvg_proximal: float
    fvg_distal: float
    features: dict = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1


def _cisd_level(a: DayArrays, ext_idx: int, max_lookback: int = 60) -> float:
    """Ouverture de la première bougie de la série baissière qui a livré le prix à l'extrême."""
    j = ext_idx
    if a.c[j] >= a.o[j]:  # bougie de l'extrême haussière (marteau) : la série finit juste avant
        j -= 1
    end = j
    stop = max(0, ext_idx - max_lookback)
    while j >= stop and a.c[j] < a.o[j]:
        j -= 1
    start = j + 1
    if start > end:
        return float(a.h[ext_idx])
    return float(a.o[start])


def _m5_bucket(t: pd.Timestamp) -> int:
    return int(t.value // (5 * 60 * 10**9))


def _is_m5_close(t: pd.Timestamp) -> bool:
    return t.minute % 5 == 4


def _round_above(x: float, step: float = 50.0) -> float:
    return float(np.floor(x / step) * step + step)


def detect_long(
    a: DayArrays,
    i0: int,
    i1: int,
    level: float,
    atr_m1: AtrFn,
    atr_m5: AtrFn,
    atr_h1: AtrFn,
    params: StrategyParams,
    obstacles: Sequence[float] = (),
) -> dict | None:
    """Cherche un setup d'achat sur `level` (en espace « achat ») entre les index i0 et i1 exclus.

    Retourne les valeurs en espace achat, ou None. `obstacles` = niveaux opposés (au-dessus).
    """
    t = a.times
    sweep_idx = None
    for i in range(i0, i1):
        if a.l[i] < level - params.sweep_min_atr_m5 * atr_m5(t[i]):
            sweep_idx = i
            break
    if sweep_idx is None:
        return None

    a1, a5, ah = atr_m1(t[sweep_idx]), atr_m5(t[sweep_idx]), atr_h1(t[sweep_idx])
    if not all(np.isfinite(v) and v > 0 for v in (a1, a5, ah)):
        return None
    max_exc = params.max_excursion_atr_h1 * ah

    ext_idx, ext = sweep_idx, a.l[sweep_idx]
    cisd = _cisd_level(a, ext_idx)
    reclaim_idx = cisd_idx = None

    for j in range(sweep_idx, i1):
        if a.l[j] < ext:
            ext_idx, ext = j, a.l[j]
            cisd = _cisd_level(a, ext_idx)
            reclaim_idx = cisd_idx = None
        if level - ext > max_exc:
            return None
        if (t[j] - t[ext_idx]) > pd.Timedelta(minutes=params.trigger_max_minutes):
            return None
        if (reclaim_idx is None and _is_m5_close(t[j]) and a.c[j] > level
                and _m5_bucket(t[j]) - _m5_bucket(t[ext_idx]) < params.reclaim_max_m5_bars):
            reclaim_idx = j
        if cisd_idx is None and j > ext_idx and a.c[j] > cisd:
            cisd_idx = j
        if reclaim_idx is None or cisd_idx is None:
            continue

        fvg = None
        for k in range(ext_idx + 2, j + 1):
            gap = a.l[k] - a.h[k - 2]
            if gap >= params.fvg_min_atr_m1 * a1:
                fvg = (float(a.l[k]), float(a.h[k - 2]), k)
                break
        if fvg is None:
            continue

        proximal, distal, fvg_idx = fvg
        entry = proximal - params.entry_depth * (proximal - distal)
        sp = float(a.spread[j])
        sl = ext - max(params.sl_buffer_spread_mult * sp, params.sl_buffer_atr_m5 * a5)
        d = entry - sl
        if d < params.sl_min_atr_m1 * a1:
            sl = entry - params.sl_min_atr_m1 * a1
            d = entry - sl
        if d > params.sl_max_atr_h1 * ah:
            entry = (proximal + distal) / 2
            d = entry - sl
            if d > params.sl_max_atr_h1 * ah:
                return None
        tp = entry + params.rr * d
        above = [x for x in obstacles if np.isfinite(x) and x > entry]
        room_r = (min(above) - entry) / d if above else float("inf")
        if room_r < params.room_min_r:
            return None

        body = abs(a.c[cisd_idx] - a.o[cisd_idx])
        features = {
            "sweep_depth_atr_m5": (level - ext) / a5,
            "cisd_body_atr_m1": body / a1,
            "fvg_size_atr_m1": (proximal - distal) / a1,
            "risk_atr_m1": d / a1,
            "risk_atr_h1": d / ah,
            "minutes_extreme_to_trigger": (t[j] - t[ext_idx]).total_seconds() / 60,
            "room_r": room_r,
            "round50_r": (_round_above(entry) - entry) / d,
            "spread": sp,
            "atr_m1": a1,
            "atr_h1": ah,
        }
        return {
            "sweep_idx": sweep_idx, "extreme_idx": ext_idx, "extreme": float(ext),
            "trigger_idx": j, "entry": float(entry), "sl": float(sl), "tp": float(tp), "risk": float(d),
            "fvg_proximal": proximal, "fvg_distal": distal, "features": features,
        }
    return None


def detect(
    bars: pd.DataFrame,
    side: str,
    level_name: str,
    level: float,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    atr_m1: AtrFn,
    atr_m5: AtrFn,
    atr_h1: AtrFn,
    params: StrategyParams,
    obstacles: Sequence[float] = (),
) -> Order | None:
    """Détecte un setup sur un niveau dans une fenêtre, pour un achat ou une vente, en prix réels."""
    sign = 1 if side == "long" else -1
    a = DayArrays.from_frame(bars, sign)
    i0 = int(a.times.searchsorted(window_start))
    i1 = int(a.times.searchsorted(window_end))
    if i1 <= i0:
        return None
    res = detect_long(a, i0, i1, sign * level, atr_m1, atr_m5, atr_h1, params,
                      [sign * x for x in obstacles])
    if res is None:
        return None
    s = sign
    return Order(
        side=side, level_name=level_name, level=level,
        sweep_time=a.times[res["sweep_idx"]], extreme_time=a.times[res["extreme_idx"]],
        extreme=s * res["extreme"],
        trigger_time=a.times[res["trigger_idx"]] + pd.Timedelta(minutes=1),
        trigger_idx=res["trigger_idx"], extreme_idx=res["extreme_idx"],
        entry=s * res["entry"], sl=s * res["sl"], tp=s * res["tp"], risk=res["risk"],
        fvg_proximal=s * res["fvg_proximal"], fvg_distal=s * res["fvg_distal"],
        features=res["features"],
    )
