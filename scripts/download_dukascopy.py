"""Télécharge les ticks XAU/USD de Dukascopy et les enregistre en bougies M1, un fichier par mois.

Exemple :
    python scripts/download_dukascopy.py --start 2015-01-01 --end 2026-10-01

Les fichiers horaires bruts sont gardés en cache dans data/cache (relance sans re-télécharger).
Sortie : data/m1/XAUUSD_AAAA-MM.csv.gz
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sniper.data import dukascopy_ticks, ticks_to_m1  # noqa: E402


def month_starts(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d < end:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield max(d, start), min(nxt, end)
        d = nxt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    out_dir = args.data / "m1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for a, b in month_starts(args.start, args.end):
        out = out_dir / f"{args.symbol}_{a:%Y-%m}.csv.gz"
        if out.exists():
            print(f"{out} existe déjà")
            continue
        ticks = dukascopy_ticks(args.symbol, a, b, args.data / "cache")
        m1 = ticks_to_m1(ticks)
        m1.to_csv(out)
        print(f"{out} : {len(ticks):,} ticks, {len(m1):,} bougies M1")


if __name__ == "__main__":
    main()
