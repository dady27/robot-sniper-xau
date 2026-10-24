import lzma
import struct
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from sniper import sessions
from sniper.data import atr_known_at, decode_bi5, load_mt5_bars, ticks_to_m1


def test_london_killzone_follows_daylight_saving():
    winter = sessions.london_killzone(date(2026, 1, 14))
    summer = sessions.london_killzone(date(2026, 7, 14))
    assert winter.start == pd.Timestamp("2026-01-14 07:00", tz="UTC")
    assert summer.start == pd.Timestamp("2026-07-14 06:00", tz="UTC")


def test_ny_killzone_follows_daylight_saving():
    assert sessions.ny_killzone(date(2026, 1, 14)).start == pd.Timestamp("2026-01-14 13:00", tz="UTC")
    assert sessions.ny_killzone(date(2026, 7, 14)).start == pd.Timestamp("2026-07-14 12:00", tz="UTC")


def test_previous_trading_day_of_monday_is_friday():
    prev = sessions.previous_trading_day(date(2026, 10, 26))  # lundi
    assert prev.end == pd.Timestamp("2026-10-23 21:00", tz="UTC")  # vendredi 17:00 New York (heure d'été)


def test_decode_bi5_roundtrip():
    rec = struct.pack(">IIIff", 1500, 4166250, 4165950, 1.0, 2.0) + struct.pack(">IIIff", 61000, 4166500, 4166200, 1.0, 1.0)
    df = decode_bi5(lzma.compress(rec, format=lzma.FORMAT_ALONE), datetime(2026, 10, 16, 10), 1000.0)
    assert list(df["bid"]) == pytest.approx([4165.95, 4166.2])
    assert list(df["ask"]) == pytest.approx([4166.25, 4166.5])
    assert df.index[1] == pd.Timestamp("2026-10-16 10:01:01", tz="UTC")


def test_ticks_to_m1_uses_bid_and_mean_spread():
    idx = pd.to_datetime(["2026-10-16 10:00:01", "2026-10-16 10:00:30", "2026-10-16 10:01:05"], utc=True)
    ticks = pd.DataFrame({"bid": [10.0, 12.0, 11.0], "ask": [10.2, 12.4, 11.3]}, index=idx)
    m1 = ticks_to_m1(ticks)
    assert m1.iloc[0][["open", "high", "low", "close"]].tolist() == [10.0, 12.0, 10.0, 12.0]
    assert m1.iloc[0]["spread"] == pytest.approx(0.3)
    assert m1.iloc[1]["ticks"] == 1


def test_atr_is_only_known_after_bar_close():
    idx = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 1.0, "close": 1.5}, index=idx)
    a = atr_known_at(bars, 1, period=14)
    assert np.isnan(a.asof(idx[13]))          # la 14e bougie n'est pas encore clôturée
    assert a.asof(idx[14]) == pytest.approx(1.0)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_load_mt5_bars_reads_export(tmp_path, encoding):
    text = (
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2026.10.16\t10:00:00\t4166.250\t4167.100\t4165.900\t4166.800\t120\t0\t160\n"
        "2026.10.16\t10:01:00\t4166.800\t4167.000\t4166.000\t4166.500\t95\t0\t180\n"
    )
    f = tmp_path / "XAUUSDm_M1.csv"
    f.write_text(text, encoding=encoding)
    bars = load_mt5_bars(f, server_utc_offset_hours=2)
    assert bars.index[0] == pd.Timestamp("2026-10-16 08:00", tz="UTC")
    assert bars["spread"].tolist() == pytest.approx([0.16, 0.18])  # 3 décimales : 1 point = 0,001 $
    assert bars["close"].iloc[1] == pytest.approx(4166.5)
