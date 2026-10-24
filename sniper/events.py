"""Étude d'événements : sweep du range asiatique, confirmation, puis mouvement qui suit.

Pour chaque jour et chaque côté du range asiatique (00h-06h Londres) :
1. Sweep : premier dépassement du haut (ou du bas) entre 06h Londres et 11h New York.
2. Confirmation, au choix :
   - reclaim : première clôture M5 (ou M15) revenue dans le range ;
   - cisd : réintégration, puis clôture M1 au-delà de l'ouverture de la série de bougies
     qui a fait l'extrême.
   Un nouvel extrême remet la confirmation à zéro.
3. Entrée au marché à la clôture de confirmation, dans le sens du retour (achat à l'ask,
   vente au bid). Stop au-delà de l'extrême + 0,1 ATR M5, éventuellement élargi.
4. Suivi M1 jusqu'à 16h New York : excursion favorable maximale avant le stop, en R.

Une cible à x R est gagnée si l'excursion atteint x avant le stop ; sinon le trade perd 1R,
ou sort à 16h au prix du moment. Sans information sur le sens, la probabilité de toucher
+x R avant -1 R vaut 1 / (1 + x) (marche aléatoire), un peu moins après le spread.

Contrôle neutre : au même instant, avec la même distance de stop, on mesure l'achat et la
vente et on fait la moyenne. L'écart événement - contrôle isole l'information apportée par
le sens du setup ; le moment, la volatilité du jour et le spread sont les mêmes des deux côtés.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from . import sessions
from .backtest import AtrBook, _span


@dataclass(frozen=True)
class EventParams:
    confirmation: str = "reclaim"       # reclaim ou cisd
    conf_minutes: int = 5               # unité de la clôture de réintégration : 5 (M5) ou 15 (M15)
    stop_buffer_atr_m5: float = 0.10
    min_risk_atr_h1: float = 0.0        # stop au moins à cette distance : le spread pèse moins en R
    exit_ny: time = time(16, 0)
    targets_r: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def _cisd_level(o, h, c, ext_idx: int, max_lookback: int = 60) -> float:
    """Ouverture de la série baissière qui a livré l'extrême (même règle que signals._cisd_level)."""
    j = ext_idx
    if c[j] >= o[j]:
        j -= 1
    end = j
    stop = max(0, ext_idx - max_lookback)
    while j >= stop and c[j] < o[j]:
        j -= 1
    start = j + 1
    return float(h[ext_idx]) if start > end else float(o[start])


def follow(h, l, c, exit_cost, start: int, end: int, entry: float, stop: float, levels: dict[str, float]):
    """Suit un achat (prix retournés pour une vente) de la bougie `start` à `end` exclue.

    Retourne (excursion max avant le stop en R, stoppé ?, R à la sortie de temps, niveaux atteints).
    Sur une bougie qui touche le stop, l'excursion de cette bougie ne compte pas (hypothèse pessimiste).
    """
    risk = entry - stop
    mfe, hit = 0.0, {k: False for k in levels}
    for k in range(start, end):
        if l[k] - exit_cost[k] <= stop:
            return mfe, True, -1.0, hit
        best = h[k] - exit_cost[k]
        mfe = max(mfe, (best - entry) / risk)
        for name, lvl in levels.items():
            hit[name] = hit[name] or best >= lvl
    return mfe, False, (c[end - 1] - exit_cost[end - 1] - entry) / risk, hit


def outcome_r(mfe: float, stopped: bool, r_end: float, x: float) -> float:
    """R d'un trade avec un TP à x R, à partir de son excursion."""
    if mfe >= x:
        return x
    return -1.0 if stopped else min(r_end, x)


def day_events(m1: pd.DataFrame, atrs: AtrBook, day: date, p: EventParams) -> list[dict]:
    asia = sessions.asia_range(day)
    w_end = sessions.ny_killzone(day).end
    h_end = pd.Timestamp(datetime.combine(day, p.exit_ny, tzinfo=sessions.NEW_YORK)).tz_convert("UTC")
    bars = _span(m1, asia.start, h_end)
    t = bars.index
    i_ae, i_we, n = int(t.searchsorted(asia.end)), int(t.searchsorted(w_end)), len(t)
    if i_ae < 200 or n - i_ae < 400:
        return []
    O, H, L, C, SP = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "spread"))
    zero = np.zeros(n)
    ah, al = H[:i_ae].max(), L[:i_ae].min()
    a_h1 = atrs.fn("h1")(asia.end)
    a_m5 = atrs.fn("m5")
    conf_close = ((t.minute.to_numpy() + 1) % p.conf_minutes) == 0
    # Prix en espace « achat » : une vente se traite en retournant les prix (x -> -x).
    space = {1: (O, H, L, C, SP, zero), -1: (-O, -L, -H, -C, zero, SP)}
    rows = []
    for side, sign in (("low", 1), ("high", -1)):
        o, h, l, c, entry_cost, exit_cost = space[sign]
        level, opposite, mid = sign * (al if sign == 1 else ah), sign * (ah if sign == 1 else al), sign * (ah + al) / 2
        swept = np.nonzero(l[i_ae:i_we] < level)[0]
        if len(swept) == 0:
            continue
        i_sw = i_ae + int(swept[0])
        ext, i_ext, reclaimed, i_conf = l[i_sw], i_sw, False, None
        for j in range(i_sw, i_we):
            if l[j] < ext:
                ext, i_ext, reclaimed = l[j], j, False
            if not reclaimed and conf_close[j] and c[j] > level:
                reclaimed = True
                if p.confirmation == "reclaim":
                    i_conf = j
                    break
            if p.confirmation == "cisd" and reclaimed and j > i_ext and c[j] > _cisd_level(o, h, c, i_ext):
                i_conf = j
                break
        if i_conf is None or i_conf + 1 >= n:
            continue
        a5 = a_m5(t[i_conf])
        if not (np.isfinite(a5) and np.isfinite(a_h1)):
            continue
        entry = c[i_conf] + entry_cost[i_conf]
        stop = min(ext - p.stop_buffer_atr_m5 * a5, entry - p.min_risk_atr_h1 * a_h1)
        risk = entry - stop
        mfe, stopped, r_end, hit = follow(h, l, c, exit_cost, i_conf + 1, n, entry, stop, {"mid": mid, "opp": opposite})
        # Contrôle neutre : achat et vente au même instant, même risque.
        ctrl = []
        for s2 in (1, -1):
            _, h2, l2, c2, ec2, xc2 = space[s2]
            e2 = c2[i_conf] + ec2[i_conf]
            ctrl.append(follow(h2, l2, c2, xc2, i_conf + 1, n, e2, e2 - risk, {}))
        row = {
            "day": day, "side": "long" if sign == 1 else "short", "level": "AL" if sign == 1 else "AH",
            "sweep_time": t[i_sw], "conf_time": t[i_conf],
            "conf_hour_ny": t[i_conf].tz_convert(sessions.NEW_YORK).hour,
            "asia_width_atr_h1": (ah - al) / a_h1, "depth_atr_h1": (level - ext) / a_h1,
            "minutes_sweep_to_conf": (t[i_conf] - t[i_sw]).total_seconds() / 60,
            "entry": sign * entry, "stop": sign * stop, "risk": risk, "risk_atr_h1": risk / a_h1,
            "spread_r": SP[i_conf] / risk, "opp_r": (opposite - entry) / risk, "mid_r": (mid - entry) / risk,
            "mfe": mfe, "stopped": stopped, "r_end": r_end, "hit_mid": hit["mid"], "hit_opp": hit["opp"],
        }
        for x in p.targets_r:
            row[f"r{x:g}"] = outcome_r(mfe, stopped, r_end, x)
            row[f"c{x:g}"] = np.mean([outcome_r(m, s, r, x) for m, s, r, _ in ctrl])
        x = row["opp_r"]
        row["r_opp"] = outcome_r(mfe, stopped, r_end, x) if x > 0 else np.nan
        row["c_opp"] = np.mean([outcome_r(m, s, r, x) for m, s, r, _ in ctrl]) if x > 0 else np.nan
        rows.append(row)
    return rows


def run_events(m1: pd.DataFrame, p: EventParams = EventParams(),
               start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Événements de tous les jours ouvrés entre `start` et `end` (exclu)."""
    atrs = AtrBook(m1)
    day = start or (m1.index[0].date() + timedelta(days=2))
    last = end or m1.index[-1].date()
    rows: list[dict] = []
    while day < last:
        if day.weekday() < 5:
            rows.extend(day_events(m1, atrs, day, p))
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def summarize_events(ev: pd.DataFrame, targets_r=EventParams.targets_r) -> pd.DataFrame:
    """Par cible : taux de TP, espérance, espérance du contrôle neutre, écart et sa statistique t."""
    rows = []
    cols = [(f"{x:g}", f"r{x:g}", f"c{x:g}", ev["mfe"] >= x, 1 / (1 + x)) for x in targets_r]
    cols.append(("opp", "r_opp", "c_opp", ev["hit_opp"], 1 / (1 + ev["opp_r"].clip(lower=0))))
    for name, r_col, c_col, tp, chance in cols:
        ok = ev[r_col].notna()
        d = ev.loc[ok, r_col] - ev.loc[ok, c_col]
        n = int(ok.sum())
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "cible_r": name, "n": n,
            "taux_tp": tp[ok].mean(), "taux_hasard": np.mean(chance[ok]) if np.ndim(chance) else chance,
            "esperance_r": ev.loc[ok, r_col].mean(), "esperance_controle_r": ev.loc[ok, c_col].mean(),
            "ecart_r": d.mean(), "t": d.mean() / se if se and se > 0 else np.nan,
        })
    return pd.DataFrame(rows)
