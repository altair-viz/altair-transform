"""Utilities for working with pandas & JS datetimes."""
import pandas as pd
from dateutil.tz import tzlocal


def date_to_timestamp(date: pd.DatetimeIndex) -> float:
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
