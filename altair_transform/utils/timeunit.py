"""Utilities for working with pandas & JS datetimes."""
import re
from typing import Union, Set
import pandas as pd
from dateutil.tz import tzlocal

__all__ = ['date_to_timestamp', 'timestamp_to_date', 'compute_timeunit']

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


_simple_timeunits = ['utc', 'year', 'quarter', 'month', 'day', 'date',
                     'hours', 'minutes', 'seconds', 'milliseconds']
_elements = ''.join(f'(?P<{name}>{name})?' for name in _simple_timeunits)
_timeunit_regex = re.compile(f'^{_elements}$')


def _parse_timeunit_string(s: str) -> Set[str]:
    """Return the set of timeunit keys in a specification string."""
    match = _timeunit_regex.match(s)
    if not match:
        return set()
    return {k for k, v in match.groupdict().items() if v}


def _standard_timeunit(name: str, date: Date) -> Date:
    units = _parse_timeunit_string(name)
    if 'day' in units:
        raise NotImplementedError('quarter and day timeunit')
    if not units:
        raise ValueError(f"{0!r} is not a recognized timeunit")

    def quarter(month):
        return month - (month - 1) % 3

    Y = date.year.astype(str) if 'year' in units else '1900'
    M = date.month.astype(str).str.zfill(2) if 'month' in units else (
        quarter(date.month).astype(str).str.zfill(2)
        if 'quarter' in units else '01'
    )
    D = date.day.astype(str).str.zfill(2) if 'date' in units else '01'
    h = date.hour.astype(str).str.zfill(2) if 'hours' in units else '00'
    m = date.minute.astype(str).str.zfill(2) if 'minutes' in units else '00'
    s = date.second.astype(str).str.zfill(2) if 'seconds' in units else '00'
    ms = ((date.microsecond // 1000).astype(str).str.zfill(3)
          if 'milliseconds' in units else '00')
    return pd.to_datetime(Y + '-' + M + '-' + D + ' ' +
                          h + ':' + m + ':' + s + '.' + ms)


def compute_timeunit(date: Date, timeunit: str) -> Date:
    """Compute a timeUnit transform"""
    # Convert to either UTC or localtime as appropriate.
    def dt(date):
        return date.dt if isinstance(date, pd.Series) else date
    if dt(date).tz is None:
        date = dt(date).tz_localize(tzlocal())
    timezone = 'UTC' if timeunit.startswith('utc') else tzlocal()
    date = dt(date).tz_convert(timezone)

    def func(date):
        if timeunit in ['day', 'utcday']:
            return (pd.to_datetime('2006-01-01') +
                    pd.to_timedelta((date.dayofweek + 1) % 7, 'D'))
        return _standard_timeunit(timeunit, date)

    if isinstance(date, pd.Series):
        return pd.Series(func(date.dt))
    elif isinstance(date, pd.Timestamp):
        return func(pd.DatetimeIndex([date]))[0]
    else:
        return func(date)
