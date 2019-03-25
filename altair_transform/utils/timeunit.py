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


def _timeunit(name):
    """Decorator for timeunit transforms"""
    def wrapper(func, timezone='UTC' if name.startswith('utc') else tzlocal()):
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

    return wrapper


def _standard_timeunit(name, date):
    Y = date.year.astype(str) if 'year' in name else '1900'
    M = date.month.astype(str).str.zfill(2) if 'month' in name else '01'
    D = date.day.astype(str).str.zfill(2) if 'date' in name else '01'
    h = date.hour.astype(str).str.zfill(2) if 'hours' in name else '00'
    m = date.minute.astype(str).str.zfill(2) if 'minutes' in name else '00'
    s = date.second.astype(str).str.zfill(2) if 'seconds' in name else '00'
    ms = date.microsecond.astype(str).str.zfill(6) if 'milliseconds' in name else '00'
    return pd.to_datetime(Y + '-' + M + '-' + D + ' ' +
                          h + ':' + m + ':' + s + '.' + ms)


@_timeunit('year')
def year(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'year' timeUnit."""
    return _standard_timeunit('year', date)


@_timeunit('utcyear')
def utcyear(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcyear' timeUnit."""
    return _standard_timeunit('utcyear', date)


@_timeunit('month')
def month(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'month' timeUnit."""
    return _standard_timeunit('month', date)


@_timeunit('utcmonth')
def utcmonth(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcmonth' timeUnit."""
    return _standard_timeunit('utcmonth', date)


@_timeunit('date')
def date(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'date' timeUnit."""
    return _standard_timeunit('date', date)


@_timeunit('utcdate')
def utcdate(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcdate' timeUnit."""
    return _standard_timeunit('utcdate', date)


@_timeunit('hours')
def hours(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'hours' timeUnit."""
    return _standard_timeunit('hours', date)


@_timeunit('utchours')
def utchours(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utchours' timeUnit."""
    return _standard_timeunit('utchours', date)


@_timeunit('minutes')
def minutes(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'minutes' timeUnit."""
    return _standard_timeunit('minutes', date)


@_timeunit('utcminutes')
def utcminutes(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcminutes' timeUnit."""
    return _standard_timeunit('utcminutes', date)


@_timeunit('seconds')
def seconds(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'seconds' timeUnit."""
    return _standard_timeunit('seconds', date)


@_timeunit('utcseconds')
def utcseconds(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcseconds' timeUnit."""
    return _standard_timeunit('utcseconds', date)


@_timeunit('milliseconds')
def milliseconds(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'seconds' timeUnit."""
    return _standard_timeunit('milliseconds', date)


@_timeunit('utcmilliseconds')
def utcmilliseconds(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcseconds' timeUnit."""
    return _standard_timeunit('utcmilliseconds', date)


@_timeunit('yearmonth')
def yearmonth(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'yearmonth' timeUnit."""
    return _standard_timeunit('yearmonth', date)


@_timeunit('utcyearmonth')
def utcyearmonth(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcyearmonth' timeUnit."""
    return _standard_timeunit('utcyearmonth', date)


@_timeunit('yearmonthdate')
def yearmonthdate(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'yearmonthdate' timeUnit."""
    return _standard_timeunit('yearmonthdate', date)


@_timeunit('utcyearmonthdate')
def utcyearmonthdate(date: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Implement vega-lite's 'utcyearmonthdate' timeUnit."""
    return _standard_timeunit('utcyearmonthdate', date)
