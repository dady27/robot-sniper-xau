"""Chargement des données : ticks Dukascopy, export de bougies MT5, bougies M1 et ATR.

Format interne : un DataFrame de bougies M1 indexé par l'heure d'ouverture en UTC,
colonnes open, high, low, close (prix bid), spread (ask - bid moyen, en dollars)
et ticks (nombre de ticks dans la minute).
"""

from __future__ import annotations

import lzma
import struct
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
# XAUUSD est coté avec 3 décimales dans le flux Dukascopy.
DUKASCOPY_PRICE_SCALE = {"XAUUSD": 1000.0}
_TICK = struct.Struct(">IIIff")  # ms depuis le début de l'heure, ask, bid, volume ask, volume bid


def decode_bi5(raw: bytes, hour_start: datetime, scale: float) -> pd.DataFrame:
    """Décode un fichier horaire .bi5 (LZMA) en ticks bid/ask horodatés UTC."""
    if not raw:
        return pd.DataFrame(columns=["bid", "ask"], index=pd.DatetimeIndex([], tz="UTC"))
    data = lzma.decompress(raw)
    n = len(data) // _TICK.size
    rec = np.frombuffer(data[: n * _TICK.size], dtype=np.dtype([
        ("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_vol", ">f4"), ("bid_vol", ">f4"),
    ]))
    t0 = pd.Timestamp(hour_start).tz_convert("UTC") if pd.Timestamp(hour_start).tzinfo else pd.Timestamp(hour_start, tz="UTC")
    idx = t0 + pd.to_timedelta(rec["ms"].astype(np.int64), unit="ms")
    return pd.DataFrame({"bid": rec["bid"] / scale, "ask": rec["ask"] / scale}, index=idx)


def download_dukascopy_hour(symbol: str, hour: datetime, cache_dir: Path) -> bytes:
    """Télécharge (ou relit du cache) une heure de ticks. Le mois est numéroté à partir de 0 dans l'URL."""
    path = cache_dir / symbol / f"{hour:%Y/%m/%d/%H}h_ticks.bi5"
    if path.exists():
        return path.read_bytes()
    url = DUKASCOPY_URL.format(symbol=symbol, y=hour.year, m=hour.month - 1, d=hour.day, h=hour.hour)
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 (URL fixe)
        raw = r.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def dukascopy_ticks(symbol: str, start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    """Ticks de `start` inclus à `end` exclu, week-ends sautés."""
    scale = DUKASCOPY_PRICE_SCALE[symbol]
    frames = []
    day = start
    while day < end:
        if day.weekday() != 5:  # le samedi le marché de l'or est fermé
            for h in range(24):
                hour = datetime(day.year, day.month, day.day, h)
                raw = download_dukascopy_hour(symbol, hour, cache_dir)
                frames.append(decode_bi5(raw, hour, scale))
        day += timedelta(days=1)
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["bid", "ask"], index=pd.DatetimeIndex([], tz="UTC"))
    return pd.concat(frames).sort_index()


def ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    """Agrège des ticks bid/ask en bougies M1 sur le prix bid, avec le spread moyen."""
    bid = ticks["bid"].resample("1min", label="left", closed="left")
    out = bid.ohlc()
    out["spread"] = (ticks["ask"] - ticks["bid"]).resample("1min", label="left", closed="left").mean()
    out["ticks"] = bid.count()
    return out.dropna(subset=["open"])


def load_mt5_bars(path: Path, server_utc_offset_hours: float = 0.0, point: float | None = None) -> pd.DataFrame:
    """Lit un export de bougies M1 de MetaTrader 5 (fenêtre Symboles, onglet Barres, séparateur tabulation).

    Colonnes attendues : <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>.
    Le spread MT5 est en points ; sans `point`, sa taille est déduite du nombre de décimales
    des prix (2 décimales : 1 point = 0,01 $). Exness affiche ses horaires serveur en UTC+0.
    """
    path = Path(path)
    encoding = "utf-16" if path.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    df = pd.read_csv(path, sep="\t", dtype=str, encoding=encoding)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]
    if point is None:
        decimals = df["close"].str.partition(".")[2].str.len().max()
        point = 10.0 ** -int(decimals or 0)
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="mixed", yearfirst=True)
    ts = ts - pd.Timedelta(hours=server_utc_offset_hours)
    out = pd.DataFrame({
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "close": df["close"].astype(float),
        "spread": df["spread"].astype(float) * point, "ticks": df["tickvol"].astype(float),
    })
    out.index = pd.DatetimeIndex(ts).tz_localize("UTC")
    return out.sort_index()


def resample(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Bougies d'unité supérieure, indexées par l'heure d'ouverture."""
    agg = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean", "ticks": "sum"}
    )
    return agg.dropna(subset=["open"])


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR simple (moyenne mobile du true range), connu à la clôture de chaque bougie."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def atr_known_at(bars: pd.DataFrame, rule_minutes: int, period: int = 14) -> pd.Series:
    """ATR indexé par l'heure de **clôture** de la bougie, pour l'interroger sans regarder le futur."""
    a = atr(bars, period)
    a.index = a.index + pd.Timedelta(minutes=rule_minutes)
    return a


def value_at(series: pd.Series, t: pd.Timestamp) -> float:
    """Dernière valeur connue à l'instant t (asof)."""
    v = series.asof(t)
    return float(v) if pd.notna(v) else float("nan")
