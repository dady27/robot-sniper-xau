"""Outils communs pour tester les règles tirées des documents de Dady.

Toutes les entrées sont évaluées avec le même contrôle neutre qu'events.py :
au même instant, même distance de stop, achat et vente, moyenne des deux.
Les distances exprimées en pips EURUSD dans les documents sont converties en
fraction de l'ATR journalier (ADR) : 75 pips EURUSD ~ 1 ADR (2015-2019).
"""
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_backtest import load_m1_dir  # noqa: E402
from sniper import sessions  # noqa: E402
from sniper.backtest import AtrBook, _span  # noqa: E402, F401 (réexportés pour les scripts h_*)
from sniper.events import follow, outcome_r  # noqa: E402

XS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
PIP_ADR = 1 / 75  # 1 pip EURUSD en ADR


def load(start: date, end: date, pad_days: int = 40) -> pd.DataFrame:
    """Bougies M1 du dossier LAB_M1 (data/m1 par défaut), ou d'un pickle (ex. marche aléatoire)."""
    src = os.environ.get("LAB_M1", str(ROOT / "data" / "m1"))
    m1 = pd.read_pickle(src) if src.endswith(".pkl") else load_m1_dir(Path(src))
    lo = pd.Timestamp(start - timedelta(days=pad_days), tz="UTC")
    hi = pd.Timestamp(end + timedelta(days=2), tz="UTC")
    return m1[(m1.index >= lo) & (m1.index < hi)]


def ny(day: date, hh: int, mm: int = 0) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time(hh, mm), tzinfo=sessions.NEW_YORK)).tz_convert("UTC")


def ldn(day: date, hh: int, mm: int = 0) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time(hh, mm), tzinfo=sessions.LONDON)).tz_convert("UTC")


def ny_day_label(index: pd.DatetimeIndex) -> np.ndarray:
    """Journée de trading New York (18h la veille -> 17h) de chaque horodatage."""
    loc = index.tz_convert(sessions.NEW_YORK).tz_localize(None) + pd.Timedelta(hours=7)
    return loc.normalize()


def daily(m1: pd.DataFrame) -> pd.DataFrame:
    """Bougies journalières New York et ADR (ATR 14 jours connu à l'ouverture du jour)."""
    g = m1.groupby(ny_day_label(m1.index))
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "bars": g["close"].size()})
    d = d[d["bars"] > 300]
    pc = d["close"].shift(1)
    tr = np.maximum(d["high"], pc) - np.minimum(d["low"], pc)
    d["adr"] = tr.rolling(14).mean().shift(1)
    d.index = d.index.date
    return d


def business_days(start: date, end: date):
    d = start
    while d < end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


class Space:
    """Tableaux d'une tranche de M1, en espace « achat » pour chaque sens (vente = prix retournés)."""

    def __init__(self, bars: pd.DataFrame):
        self.t = bars.index
        self.n = len(bars)
        O, H, L, C, SP = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "spread"))
        zero = np.zeros(self.n)
        self.SP = SP
        self.raw = (O, H, L, C)
        self.sp = {1: (O, H, L, C, SP, zero), -1: (-O, -L, -H, -C, zero, SP)}

    def idx(self, ts: pd.Timestamp) -> int:
        return int(self.t.searchsorted(ts))


def evaluate(sp: Space, i: int, sign: int, risk: float, n_end: int, entry: float | None = None,
             extra: dict | None = None) -> dict | None:
    """Trade dans le sens `sign` décidé à la bougie i.

    entry None : ordre au marché à la clôture de i (achat à l'ask, vente au bid).
    entry donné (espace « achat ») : ordre limite exécuté pendant la bougie i ; si cette
    bougie touche aussi le stop, le trade est perdu (le prix est passé par l'entrée avant).
    extra : cibles supplémentaires {nom: prix cible en espace « achat »}.
    Contrôle : achat et vente au marché à la clôture de i, même risque.
    """
    if not (np.isfinite(risk) and risk > 0) or i + 1 >= n_end:
        return None
    o, h, l, c, ec, xc = sp.sp[sign]
    if entry is None:
        entry = c[i] + ec[i]
        res = follow(h, l, c, xc, i + 1, n_end, entry, entry - risk, {})
    elif l[i] - xc[i] <= entry - risk:
        res = (0.0, True, -1.0, {})
    else:
        res = follow(h, l, c, xc, i + 1, n_end, entry, entry - risk, {})
    mfe, stopped, r_end, _ = res
    ctrl = []
    for s2 in (1, -1):
        _, h2, l2, c2, ec2, xc2 = sp.sp[s2]
        e2 = c2[i] + ec2[i]
        ctrl.append(follow(h2, l2, c2, xc2, i + 1, n_end, e2, e2 - risk, {}))
    row = {"time": sp.t[i], "side": "long" if sign == 1 else "short", "entry": sign * entry, "risk": risk,
           "spread_r": sp.SP[i] / risk, "mfe": mfe, "stopped": stopped, "r_end": r_end,
           "hit_opp": False, "opp_r": np.nan, "r_opp": np.nan, "c_opp": np.nan}
    for x in XS:
        row[f"r{x:g}"] = outcome_r(mfe, stopped, r_end, x)
        row[f"c{x:g}"] = np.mean([outcome_r(m, s, r, x) for m, s, r, _ in ctrl])
    for name, target in (extra or {}).items():
        x = (target - entry) / risk
        row[f"x_{name}"] = x
        row[f"r_{name}"] = outcome_r(mfe, stopped, r_end, x) if x > 0 else np.nan
        row[f"c_{name}"] = np.mean([outcome_r(m, s, r, x) for m, s, r, _ in ctrl]) if x > 0 else np.nan
    return row


def summary(ev: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Par cible : n, taux de TP, hasard 1/(1+x), espérance nette et son t, écart au contrôle et son t."""
    rows = []
    cols = [(f"{x:g}", f"r{x:g}", f"c{x:g}", ev["mfe"] >= x, 1 / (1 + x)) for x in XS]
    for name in extra:
        xr = ev[f"x_{name}"]
        cols.append((name, f"r_{name}", f"c_{name}", ev["mfe"] >= xr, 1 / (1 + xr.clip(lower=0))))
    for name, r_col, c_col, tp, chance in cols:
        ok = ev[r_col].notna()
        n = int(ok.sum())
        if n < 2:
            continue
        r, cc = ev.loc[ok, r_col], ev.loc[ok, c_col]
        d = r - cc
        rows.append({"cible": name, "n": n, "tp": tp[ok].mean(),
                     "hasard": float(np.mean(chance[ok])) if np.ndim(chance) else chance,
                     "E_r": r.mean(), "t_E": r.mean() / (r.std(ddof=1) / np.sqrt(n)),
                     "E_ctrl": cc.mean(), "ecart": d.mean(), "t_ecart": d.mean() / (d.std(ddof=1) / np.sqrt(n))})
    return pd.DataFrame(rows)


def show(ev: pd.DataFrame, title: str, extra: tuple[str, ...] = ()) -> None:
    if ev.empty:
        print(f"\n== {title} : aucun événement")
        return
    print(f"\n== {title} : {len(ev)} trades, spread moyen {ev['spread_r'].mean():.3f}R, "
          f"{len(ev) / max(1, ev['day'].nunique()):.2f} par jour actif")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary(ev, extra).to_string(index=False))


def by_year(ev: pd.DataFrame, col: str = "r2.5", ctrl: str = "c2.5") -> str:
    y = pd.to_datetime(ev["day"]).dt.year
    g = ev.groupby(y)
    return " ".join(f"{k}:{(gg[col] - gg[ctrl]).mean():+.2f}({len(gg)})" for k, gg in g)
