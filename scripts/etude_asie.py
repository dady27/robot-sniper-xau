"""Sweep du range asiatique puis confirmation : le mouvement qui suit bat-il le hasard ?

Exemples :
    python scripts/etude_asie.py --m1-dir data/m1 --start 2015-02-01 --end 2024-01-01
    python scripts/etude_asie.py --m1-dir data/m1 --confirmation cisd --min-risk-atr-h1 1.0

Sortie : out/evenements_asie.csv (un événement par ligne) et la synthèse par cible,
pour tous les événements puis par session de la confirmation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_backtest import load_m1_dir  # noqa: E402
from sniper.events import EventParams, run_events, summarize_events  # noqa: E402


def show(ev: pd.DataFrame, title: str) -> None:
    print(f"\n== {title} : {len(ev)} événements, spread moyen {ev['spread_r'].mean():.2f}R")
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(summarize_events(ev).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-dir", type=Path, required=True)
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--confirmation", choices=["reclaim", "cisd"], default="reclaim")
    ap.add_argument("--conf-minutes", type=int, default=5)
    ap.add_argument("--min-risk-atr-h1", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()

    m1 = load_m1_dir(a.m1_dir)
    p = EventParams(confirmation=a.confirmation, conf_minutes=a.conf_minutes, min_risk_atr_h1=a.min_risk_atr_h1)
    ev = run_events(m1, p, a.start, a.end)
    a.out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(a.out / "evenements_asie.csv", index=False)
    show(ev, "tous")
    session = np.select([ev["conf_hour_ny"].between(3, 7), ev["conf_hour_ny"].between(8, 11)],
                        ["Londres (3h-8h New York)", "New York (8h-12h)"], "avant Londres (6h-8h Londres)")
    for name, g in ev.groupby(session):
        show(g, name)
    print("\nUn écart proche de 0 (|t| < 2) veut dire : pas mieux qu'un trade au hasard au même instant.")


if __name__ == "__main__":
    main()
