"""Traversée des multiples de 50 $ et des plus hauts / plus bas de la veille et de la semaine.

Exemples :
    python scripts/etude_traversee.py --m1-dir data/m1 --start 2015-02-01 --end 2024-01-01
    python scripts/etude_traversee.py --m1-dir data/m1 --start 2024-01-01 --glissement-atr-h1 0.05
    python scripts/etude_traversee.py --m1-dir data/m1 --familles ronds --decalage 16.67   # niveaux non ronds
    python scripts/etude_traversee.py --m1-dir data/m1 --familles ronds --pas 100 --stop-atr-h1 1 --entree close

Sortie : out/traversee.csv (un trade par ligne) et la synthèse par cible, toutes familles
puis par famille, avec l'écart au contrôle neutre et l'espérance par année.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_backtest import load_m1_dir  # noqa: E402
from sniper.events import summarize_events  # noqa: E402
from sniper.traversee import FAMILIES, CrossParams, run_cross  # noqa: E402


def show(ev: pd.DataFrame, title: str, targets) -> None:
    days = ev["day"].nunique()
    print(f"\n== {title} : {len(ev)} trades sur {days} jours, spread moyen {ev['spread_r'].mean():.2f}R")
    s = summarize_events(ev, targets)
    s = s[s["cible_r"] != "opp"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(s.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-dir", type=Path, required=True)
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--familles", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    ap.add_argument("--pas", type=float, default=50.0, help="pas des chiffres ronds, en dollars")
    ap.add_argument("--decalage", type=float, default=0.0, help="décalage de la grille (contrôle)")
    ap.add_argument("--stop-atr-h1", type=float, default=0.5)
    ap.add_argument("--entree", choices=["stop", "close"], default="stop")
    ap.add_argument("--glissement-atr-h1", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()

    p = CrossParams(families=tuple(a.familles), step=a.pas, offset=a.decalage, stop_atr_h1=a.stop_atr_h1,
                    entry=a.entree, slippage_atr_h1=a.glissement_atr_h1)
    m1 = load_m1_dir(a.m1_dir)
    ev = run_cross(m1, p, a.start, a.end)
    if ev.empty:
        raise SystemExit("Aucun trade sur la période")
    a.out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(a.out / "traversee.csv", index=False)
    show(ev, "toutes familles", p.targets_r)
    for fam, g in ev.groupby("famille"):
        show(g, fam, p.targets_r)
    year = pd.to_datetime(ev["day"]).dt.year
    print("\nEspérance à 2,5R par année :",
          " ".join(f"{y}: {g['r2.5'].mean():+.2f}R ({len(g)})" for y, g in ev.groupby(year)))


if __name__ == "__main__":
    main()
