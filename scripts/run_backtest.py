"""Rejoue les modèles A et B sur l'historique M1 et affiche la synthèse.

Exemples :
    python scripts/run_backtest.py --m1-dir data/m1
    python scripts/run_backtest.py --mt5 export_XAUUSD_M1.csv --server-utc-offset 0

Sortie : out/setups.csv (un setup par ligne) et out/synthese.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sniper.backtest import run  # noqa: E402
from sniper.data import load_mt5_bars  # noqa: E402
from sniper.report import summarize  # noqa: E402


def load_m1_dir(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"Aucun fichier M1 dans {path}")
    frames = [pd.read_csv(f, index_col=0, parse_dates=True) for f in files]
    m1 = pd.concat(frames).sort_index()
    m1.index = pd.DatetimeIndex(m1.index).tz_convert("UTC") if m1.index.tz else pd.DatetimeIndex(m1.index).tz_localize("UTC")
    return m1[~m1.index.duplicated()]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--m1-dir", type=Path)
    src.add_argument("--mt5", type=Path)
    ap.add_argument("--server-utc-offset", type=float, default=0.0)
    ap.add_argument("--point", type=float, default=None, help="taille du point MT5 (déduite des prix par défaut)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    m1 = load_m1_dir(args.m1_dir) if args.m1_dir else load_mt5_bars(args.mt5, args.server_utc_offset, args.point)
    print(f"{len(m1):,} bougies M1 du {m1.index[0]:%Y-%m-%d} au {m1.index[-1]:%Y-%m-%d}")
    setups = run(m1)
    args.out.mkdir(parents=True, exist_ok=True)
    setups.to_csv(args.out / "setups.csv", index=False)
    syn = summarize(setups, trading_days=int(pd.Series(m1.index.date).nunique()))
    syn.to_csv(args.out / "synthese.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(syn.to_string(index=False))


if __name__ == "__main__":
    main()
