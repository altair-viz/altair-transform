"""Utilities for working with pandas & JS datetimes."""
from typing import Union
import pandas as pd
from dateutil.tz import tzlocal
from functools import wraps

Date = Union[pd.Series, pd.DatetimeIndex, pd.Timestamp]


def date_to_timestamp(date: pd.DatetimeIndex):
    """Convert a pandas datetime to a javascript timestamp.

    This aims to match the timezone handling semantics
    used in Vega and Vega-Lite.

    Parameters
    ----------
    timestamp : float
        The unix epoch timestamp.

    Returns
    -------
    date : pd.DatetimeIndex
        The timestamps to be converted

    See Also
    --------
    date_to_timestamp : opposite of this function
    """
    if date.tzinfo is None:
        date = date.tz_localize(tzlocal())
    try:
        # Works for pd.Timestamp
        return date.timestamp() * 1000
    except AttributeError:
        # Works for pd.DatetimeIndex
        return date.astype('int64') * 1E-6


def timestamp_to_date(timestamp: float,
                      tz: bool = False,
                      utc: bool = False) -> pd.DatetimeIndex:
    """Convert javascript timestamp to a pandas datetime.

    This aims to match the timezone handling semantics
    used in Vega and Vega-Lite.

    Parameters
    ----------
    date : pd.DatetimeIndex
        The timestamps to be converted

    Returns
    -------
    timestamp : float
        The unix epoch timestamp.

    See Also
    --------
    timestamp_to_date : opposite of this function
    """
    dates = pd.to_datetime(timestamp, unit='ms').tz_localize('UTC')
    if utc:
        return dates
    if tz:
        return dates.tz_convert(tzlocal())
    return dates.tz_convert(tzlocal()).tz_localize(None)


def _timeunit(arg):
    if callable(arg):
        timezone = tzlocal()
    else:
        timezone = arg

    def wrapper(func, timezone=timezone):
        @wraps(func)
        def wrapped(date: Date) -> Date:
            date = date.dt if isinstance(date, pd.Series) else date
            if date.tz is None:
                date = date.tz_localize(tzlocal())
                date = date.dt if isinstance(date, pd.Series) else date
            date = date.tz_convert(timezone)

            if isinstance(date, pd.Series):
                return pd.Series(func(date.dt))
            elif isinstance(date, pd.Timestamp):
                return func(pd.DatetimeIndex([date]))[0]
            else:
                return func(date)
        return wrapped

    if callable(arg):
        return wrapper(arg)
    else:
        return wrapper


@_timeunit
def year(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'year' timeUnit."""
    return pd.to_datetime(date.year.astype(str))


@_timeunit('utc')
def utcyear(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcyear' timeUnit."""
    return pd.to_datetime(date.year.astype(str))


@_timeunit
def yearmonth(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'yearmonth' timeUnit."""
    return pd.to_datetime(date.year.astype(str) +
                          '-' + date.month.astype(str))


@_timeunit('utc')
def utcyearmonth(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcyearmonth' timeUnit."""
    return pd.to_datetime(date.year.astype(str) +
                          '-' + date.month.astype(str))


@_timeunit
def yearmonthdate(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'yearmonthdate' timeUnit."""
    return pd.to_datetime(date.year.astype(str) +
                          '-' + date.month.astype(str) +
                          '-' + date.day.astype(str))


@_timeunit('utc')
def utcyearmonthdate(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcyearmonthdate' timeUnit."""
    return pd.to_datetime(date.year.astype(str) +
                          '-' + date.month.astype(str) +
                          '-' + date.day.astype(str))
