"""Mise optimale pour le prochain trade.

Exemple (compte standard Exness, 10 $, 44 trades restants, 50 % de TP à 2,5R) :
    python scripts/mise.py --solde 10 --trades 44 --p 0.50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sniper.sizing import solve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solde", type=float, required=True)
    ap.add_argument("--objectif", type=float, default=1000.0)
    ap.add_argument("--trades", type=int, required=True, help="trades restants dans l'horizon")
    ap.add_argument("--p", type=float, required=True, help="taux de TP touché, entre 0 et 1")
    ap.add_argument("--rr", type=float, default=2.5)
    ap.add_argument("--sl", type=float, default=3.0, help="distance du SL en dollars de prix")
    ap.add_argument("--spread", type=float, default=0.30)
    ap.add_argument("--compte", choices=["standard", "cent"], default="standard")
    a = ap.parse_args()
    r = solve(a.solde, a.objectif, a.trades, a.p, a.rr, a.sl, a.spread, a.compte)
    print(f"Risque optimal : {r.risk:.2f} $ ({r.risk / a.solde:.0%} du solde), lot {r.lots:.2f}")
    print(f"Chance d'atteindre {a.objectif:.0f} $ : {r.probability:.1%}  (RR net {r.b:.2f})")


if __name__ == "__main__":
    main()
