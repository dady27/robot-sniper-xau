"""+20 / +30 pips sur l'or (1 pip = 0,10 $) : à quelle fréquence, et avant quel stop ?

Reprend les trades des autres scripts (même minute, même sens, même prix d'entrée) et les suit
minute par minute jusqu'à 16h New York, avec une cible et un stop fixes en pips. Si la cible et
le stop tombent dans la même minute, on compte le stop. Contrôle : achat et vente au marché à la
clôture de la minute du signal, moyenne des deux.

usage: h_pips.py DOSSIER_DES_TRADES OUT_PREFIX [SPREAD_PIPS]
Le dossier contient les fichiers nommés comme dans backtest/documents (sir_jones_mw_*.csv,
bfx_62_*.csv, sweep_asie_cisd_*.csv, traversee_*.csv). Avec SPREAD_PIPS, le spread est fixe
et les entrées sont recalculées depuis les prix M1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lab import AtrBook, load, sessions

PIP = 0.10
TARGETS = (20, 30)
STOPS = (20, 30, 50, 100, 200, 300)
# (famille, motif des fichiers, colonne de l'heure du signal, type d'entrée)
SOURCES = [
    ("Sir Jones M/W", "sir_jones_mw_*.csv", "time", "market"),
    ("BFX 62 %", "bfx_62_[0-9]*.csv", "time", "limit"),
    ("BFX 62 % sans minimum", "bfx_62_sans_minimum_*.csv", "time", "limit"),
    ("Sweep Asie + CISD", "sweep_asie_cisd_*.csv", "conf_time", "market"),
    ("Traversée", "traversee_[0-9]*-[0-9]*[0-9].csv", "time", "stop"),  # sans les fichiers _glissement
]


def paths(h, l, c, xc, a, b):
    """Meilleure sortie atteinte et pire prix atteint (signe inversé), cumulés de a à b exclu."""
    up = np.maximum.accumulate(h[a:b] - xc[a:b])
    dn = np.maximum.accumulate(-(l[a:b] - xc[a:b]))
    return up, dn, c[b - 1] - xc[b - 1]


def result(e, up, dn, close_end, T, S, stopped_now=False):
    """(pips, gagné ?) : cible T pips, stop S pips, sortie à 16h sinon."""
    if stopped_now:
        return -S, False
    k_tp = np.searchsorted(up, e + T * PIP, "left")
    k_sl = np.searchsorted(dn, -(e - S * PIP), "left")
    if k_sl < len(up) and k_sl <= k_tp:
        return -S, False
    if k_tp < len(up):
        return T, True
    return min((close_end - e) / PIP, T), False


def trades(folder: Path, m1: pd.DataFrame, fixed: float | None) -> pd.DataFrame:
    t = m1.index
    O, H, L, C, SP = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "spread"))
    if fixed is not None:
        SP = np.full(len(t), fixed * PIP)
    zero = np.zeros(len(t))
    space = {1: (O, H, L, C, SP, zero), -1: (-O, -L, -H, -C, zero, SP)}
    atr_h1 = AtrBook(m1).fn("h1")
    rows = []
    for fam, pattern, tcol, kind in SOURCES:
        for path in sorted(folder.glob(pattern)):
            ev = pd.read_csv(path)
            ts = pd.to_datetime(ev[tcol], utc=True)
            for k in range(len(ev)):
                sign = 1 if ev["side"].iat[k] == "long" else -1
                i = int(t.searchsorted(ts.iat[k]))
                if i >= len(t) or t[i] != ts.iat[k]:
                    continue
                loc = t[i].tz_convert(sessions.NEW_YORK)
                b = int(t.searchsorted((loc.normalize() + pd.Timedelta(hours=16)).tz_convert("UTC")))
                if b <= i + 1:
                    continue
                o, h, l, c, ec, xc = space[sign]
                e = sign * float(ev["entry"].iat[k])
                if fixed is not None and kind == "market":
                    e = c[i] + ec[i]
                elif fixed is not None and kind == "stop":
                    e = max(sign * float(ev["level"].iat[k]), o[i]) + ec[i]
                up, dn, ce = paths(h, l, c, xc, i + 1, b)
                ctrl = {}
                for s2 in (1, -1):
                    _, h2, l2, c2, ec2, xc2 = space[s2]
                    e2 = c2[i] + ec2[i]
                    ctrl[s2] = (e2,) + paths(h2, l2, c2, xc2, i + 1, b)
                row = {"famille": fam, "time": t[i], "year": loc.year, "side": "long" if sign == 1 else "short",
                       "spread_pips": SP[i] / PIP, "atr_h1_pips": atr_h1(t[i]) / PIP}
                for T in TARGETS:
                    e_o, up_o = ctrl[-sign][0], ctrl[-sign][1]
                    row[f"mfe{T}"] = bool(np.searchsorted(up, e + T * PIP, "left") < len(up))
                    row[f"mfe{T}_opp"] = bool(np.searchsorted(up_o, e_o + T * PIP, "left") < len(up_o))
                    for S in STOPS:
                        # ordre limite : la minute du remplissage peut déjà toucher le stop
                        now = kind == "limit" and l[i] - xc[i] <= e - S * PIP
                        row[f"p{T}_{S}"], row[f"w{T}_{S}"] = result(e, up, dn, ce, T, S, now)
                        cp = [result(*v, T, S) for v in ctrl.values()]
                        row[f"cp{T}_{S}"] = np.mean([x[0] for x in cp])
                        row[f"cw{T}_{S}"] = np.mean([x[1] for x in cp])
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(periode=pd.cut(df["year"], [2014, 2019, 2023, 2100],
                                  labels=["2015-2019", "2020-2023", "2024-2026"]))
    groups = [(k, g) for k, g in df.groupby(["famille", "periode"], observed=True)]
    groups += [(("Documents (hors traversée)", p), g)
               for p, g in df[df["famille"] != "Traversée"].groupby("periode", observed=True)]
    out = []
    for (fam, per), g in groups:
        for T in TARGETS:
            for S in STOPS:
                d = g[f"p{T}_{S}"] - g[f"cp{T}_{S}"]
                out.append({"famille": fam, "periode": per, "n": len(g), "cible": T, "stop": S,
                            "seuil_%": 100 * S / (S + T),
                            "gagnants_%": 100 * g[f"w{T}_{S}"].mean(), "hasard_%": 100 * g[f"cw{T}_{S}"].mean(),
                            "pips_moy": g[f"p{T}_{S}"].mean(), "pips_hasard": g[f"cp{T}_{S}"].mean(),
                            "t_ecart": d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 else np.nan,
                            "mfe_%": 100 * g[f"mfe{T}"].mean(), "mfe_opp_%": 100 * g[f"mfe{T}_opp"].mean(),
                            "spread": g["spread_pips"].mean(), "atr_h1": g["atr_h1_pips"].median()})
    return pd.DataFrame(out)


def main():
    folder, out = Path(sys.argv[1]), sys.argv[2]
    fixed = float(sys.argv[3]) if len(sys.argv) > 3 else None
    times = []
    for _, pattern, tcol, _ in SOURCES:
        times += [pd.to_datetime(pd.read_csv(p, usecols=[tcol])[tcol], utc=True) for p in folder.glob(pattern)]
    times = pd.concat(times)
    m1 = load(times.min().date(), times.max().date())
    df = trades(folder, m1, fixed)
    df.to_csv(f"{out}_trades.csv", index=False)
    s = summarize(df)
    s.to_csv(f"{out}_synthese.csv", index=False)
    print(f"{len(df)} signaux, spread {'fixe ' + str(fixed) + ' pips' if fixed is not None else 'du fichier M1'}")
    cols = ["famille", "periode", "n", "cible", "stop", "seuil_%", "gagnants_%", "hasard_%", "pips_moy", "pips_hasard",
            "t_ecart", "mfe_%", "mfe_opp_%"]
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.1f}".format):
        print(s[s["stop"].isin([20, 100, 200])][cols].to_string(index=False))


main()
