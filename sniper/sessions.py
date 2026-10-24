"""Fenêtres de session en heure locale de Londres et de New York, converties en UTC.

Les heures sont définies en heure locale pour que le changement d'heure (DST)
soit géré automatiquement, contrairement à des heures GMT fixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp  # UTC, inclus
    end: pd.Timestamp    # UTC, exclu


def local_window(day: date, tz: ZoneInfo, start: time, end: time, name: str) -> Window:
    """Fenêtre [start, end) en heure locale `tz` le jour `day`, convertie en UTC."""
    s = datetime.combine(day, start, tzinfo=tz).astimezone(UTC)
    e = datetime.combine(day, end, tzinfo=tz).astimezone(UTC)
    return Window(name, pd.Timestamp(s), pd.Timestamp(e))


def asia_range(day: date) -> Window:
    return local_window(day, LONDON, time(0, 0), time(6, 0), "asia")


def london_killzone(day: date) -> Window:
    return local_window(day, LONDON, time(7, 0), time(10, 0), "london_kz")


def london_session(day: date) -> Window:
    """Session de Londres dont le plus haut / plus bas sert de cible à New York."""
    return local_window(day, LONDON, time(7, 0), time(11, 0), "london")


def ny_killzone(day: date) -> Window:
    return local_window(day, NEW_YORK, time(8, 0), time(11, 0), "ny_kz")


def trading_day(day: date) -> Window:
    """Journée de trading de l'or : de 18:00 New York la veille à 17:00 New York."""
    s = datetime.combine(day - timedelta(days=1), time(18, 0), tzinfo=NEW_YORK).astimezone(UTC)
    e = datetime.combine(day, time(17, 0), tzinfo=NEW_YORK).astimezone(UTC)
    return Window("trading_day", pd.Timestamp(s), pd.Timestamp(e))


def previous_trading_day(day: date) -> Window:
    """Journée de trading précédente (le vendredi pour un lundi)."""
    prev = day - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return trading_day(prev)
