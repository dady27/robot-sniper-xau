"""Rejoue la stratégie sur une marche aléatoire : le taux de TP doit rester vers 28 %.

    python scripts/controle_hasard.py --jours 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sniper.backtest import run  # noqa: E402
from sniper.report import summarize  # noqa: E402
from sniper.synthetic import random_walk_m1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jours", type=int, default=1500)
    ap.add_argument("--graine", type=int, default=0)
    a = ap.parse_args()
    m1 = random_walk_m1("2025-01-06", a.jours, seed=a.graine)
    syn = summarize(run(m1), trading_days=int(pd.Series(m1.index.date).nunique()))
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(syn.to_string(index=False))
    print("Hasard pur sans coûts : 1 / (1 + 2,5) = 0,286")


if __name__ == "__main__":
    main()
