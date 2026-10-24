"""Synthèse d'un backtest : remplissage, taux de TP, intervalle de confiance, espérance."""

from __future__ import annotations

import math

import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson à 95 % pour une proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def summarize(df: pd.DataFrame, trading_days: int | None = None) -> pd.DataFrame:
    """Une ligne par modèle et une ligne « total »."""
    if df.empty:
        return pd.DataFrame()
    groups = [(m, g) for m, g in df.groupby("model")] + [("total", df)]
    days = trading_days or df["day"].nunique()
    rows = []
    for name, g in groups:
        filled = g[g["status"] == "filled"]
        n = len(filled)
        tp = int((filled["exit_reason"] == "tp").sum())
        lo, hi = wilson(tp, n)
        rows.append({
            "modèle": name,
            "setups": len(g),
            "remplis": n,
            "taux_remplissage": n / len(g) if len(g) else float("nan"),
            "tp": tp,
            "sl": int((filled["exit_reason"] == "sl").sum()),
            "autres_sorties": int((~filled["exit_reason"].isin(["tp", "sl"])).sum()),
            "taux_tp": tp / n if n else float("nan"),
            "ic95_bas": lo,
            "ic95_haut": hi,
            "esperance_r": filled["r"].mean() if n else float("nan"),
            "trades_par_jour": n / days if days else float("nan"),
        })
    return pd.DataFrame(rows)
