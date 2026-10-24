"""Mise optimale pour atteindre un objectif de capital avant la fin d'un horizon de trades.

Programmation dynamique : en partant de la fin, pour chaque solde et chaque nombre de
trades restants, on garde la mise qui maximise la probabilité finale d'atteindre l'objectif.
Chaque trade a deux issues : TP (gain = mise x b) avec probabilité p, SL (perte = mise).

Comptes :
- « standard » : la mise est un multiple du risque du lot minimum
  (0,01 lot x 100 onces x (SL + spread) = 3,30 $ pour un SL de 3 $ et 0,30 $ de spread) ;
- « cent » : la mise est quasi continue (0,01 lot cent = 0,01 once).
Chez Exness, levier illimité et stop-out à 0 % : la seule contrainte est de pouvoir perdre la mise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SizingResult:
    probability: float      # chance d'atteindre l'objectif avec la mise optimale
    risk: float             # risque optimal pour le prochain trade, en dollars (0 = ne pas trader)
    lots: float             # taille de lot correspondante
    b: float                # gain net en R après frais


def net_payoff(rr: float, sl: float, spread: float) -> float:
    """Gain net par dollar risqué : (RR x SL - spread) / (SL + spread)."""
    return (rr * sl - spread) / (sl + spread)


def _risk_per_lot(account: str, sl: float, spread: float, contract_oz: float) -> float:
    oz = contract_oz if account == "standard" else contract_oz / 100.0
    return oz * (sl + spread)


def solve(
    balance: float,
    goal: float,
    trades_left: int,
    p: float,
    rr: float = 2.5,
    sl: float = 3.0,
    spread: float = 0.30,
    account: str = "standard",
    min_lot: float = 0.01,
    contract_oz: float = 100.0,
    resolution: float = 0.10,
) -> SizingResult:
    b = net_payoff(rr, sl, spread)
    per_lot = _risk_per_lot(account, sl, spread, contract_oz)
    unit = per_lot * min_lot                       # risque du plus petit lot, en dollars
    S = int(round(goal / resolution))
    e0 = min(S, int(np.floor(balance / resolution + 1e-9)))
    if e0 >= S:
        return SizingResult(1.0, 0.0, 0.0, b)
    if trades_left <= 0:
        return SizingResult(0.0, 0.0, 0.0, b)

    E = np.arange(S + 1)
    V = (E >= S).astype(float)
    if unit / resolution >= 2:                     # mises discrètes : k lots minimum
        steps = [(int(round(k * unit / resolution)), int(round(k * unit * b / resolution)))
                 for k in range(1, int(S * resolution / unit) + 1)]
    else:                                          # mises quasi continues : fractions du solde
        steps = None
    best_act = np.zeros(S + 1)
    for n in range(trades_left):
        best = V.copy()                            # option : ne pas trader
        act = np.zeros(S + 1)
        if steps is not None:
            for loss, win in steps:
                ok = (E >= loss) & (E < S)
                up = np.minimum(E + win, S)
                dn = np.maximum(E - loss, 0)
                val = p * V[up] + (1 - p) * V[dn]
                m = ok & (val > best + 1e-15)
                best = np.where(m, val, best)
                act = np.where(m, loss * resolution, act)
        else:
            for f in np.arange(0.01, 1.0001, 0.01):
                r = np.floor(f * E).astype(int)
                up = np.minimum(E + np.round(r * b).astype(int), S)
                dn = np.maximum(E - r, 0)
                val = p * V[up] + (1 - p) * V[dn]
                m = (E < S) & (r >= 1) & (val > best + 1e-15)
                best = np.where(m, val, best)
                act = np.where(m, r * resolution, act)
        V = best
        best_act = act
    risk = float(best_act[e0])
    lots = np.floor(risk / per_lot / min_lot + 1e-9) * min_lot if risk > 0 else 0.0
    return SizingResult(float(V[e0]), risk, float(round(lots, 2)), b)
