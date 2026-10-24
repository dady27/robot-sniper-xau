"""Télécharge les bougies M1 XAU/USD de HistData et les enregistre au format de data/m1.

Exemple :
    python scripts/download_histdata.py --start 2015 --end 2026-09

HistData publie une archive par année, puis une par mois pour l'année en cours. Les prix
sont les prix bid de Dukascopy ; le spread est estimé (voir sniper.data.load_histdata).
Les archives sont gardées dans data/histdata (relance sans re-télécharger).
Sortie : data/m1/XAUUSD_AAAA-MM.csv.gz
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import date
from http.cookiejar import CookieJar
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sniper.data import HISTDATA_SPREAD_BP, load_histdata  # noqa: E402

PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes/{pair}/{period}"
GET = "https://www.histdata.com/get.php"


def periods(start: int, end: str):
    """Années complètes, puis les mois de la dernière année (HistData ne fournit pas l'année en cours d'un bloc)."""
    last_year, last_month = (int(x) for x in end.split("-"))
    for y in range(start, last_year):
        yield str(y)
    for m in range(1, last_month + 1):
        yield f"{last_year}/{m}"


def download(pair: str, period: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    page_url = PAGE.format(pair=pair.lower(), period=period)
    html = opener.open(page_url, timeout=60).read().decode("utf-8", "replace")
    form = {k: re.search(rf'name="{k}" id="{k}" value="([^"]*)"', html).group(1) for k in ("tk", "date", "datemonth")}
    form.update(platform="MT", timeframe="M1", fxpair=pair.upper())
    req = urllib.request.Request(GET, data=urllib.parse.urlencode(form).encode(), headers={"Referer": page_url})
    raw = opener.open(req, timeout=120).read()
    if not raw.startswith(b"PK"):
        raise SystemExit(f"{period} : HistData n'a pas renvoyé d'archive")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="XAUUSD")
    ap.add_argument("--start", type=int, default=2015, help="première année")
    ap.add_argument("--end", default=f"{date.today():%Y-%m}", help="dernier mois inclus, AAAA-MM")
    ap.add_argument("--spread-bp", type=float, default=HISTDATA_SPREAD_BP)
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    frames = []
    for period in periods(args.start, args.end):
        z = download(args.pair, period, args.data / "histdata" / f"{args.pair}_{period.replace('/', '-')}.zip")
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(name) as f:
                frames.append(load_histdata(f, args.spread_bp))
        print(f"{period} : {len(frames[-1]):,} bougies M1")
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated()]
    out_dir = args.data / "m1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for (y, m), g in m1.groupby([m1.index.year, m1.index.month]):
        g.to_csv(out_dir / f"{args.pair}_{y:04d}-{m:02d}.csv.gz")
    print(f"{len(m1):,} bougies du {m1.index[0]:%Y-%m-%d} au {m1.index[-1]:%Y-%m-%d} dans {out_dir}")


if __name__ == "__main__":
    main()
