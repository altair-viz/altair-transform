"""Utilities for working with pandas & JS datetimes."""
import pandas as pd
from dateutil.tz import tzlocal


def date_to_timestamp(date: pd.DatetimeIndex) -> float:
    if date.tzinfo is None:
        date = date.tz_localize(tzlocal())
    return date.astype('int64') * 1E-6


def timestamp_to_date(timestamp: float,
                      tz: bool = False,
                      utc: bool = False) -> pd.DatetimeIndex:
    dates = pd.to_datetime(timestamp, unit='ms').tz_localize('UTC')
    if utc:
        return dates
    if tz:
        return dates.tz_convert(tzlocal())
    return dates.tz_convert(tzlocal()).tz_localize(None)
