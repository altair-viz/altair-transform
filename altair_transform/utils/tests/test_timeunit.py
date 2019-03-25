"""Tests of the timeunit utilities"""
from dateutil.tz import tzlocal
import pytest
import pytz

import pandas as pd

from altair_transform.utils import timeunit


TIMEUNITS = [
    'year', 'utcyear',
    'month', 'utcmonth',
    'date', 'utcdate',
    'hours', 'utchours',
    'minutes', 'utcminutes',
    'seconds', 'utcseconds',
    'milliseconds', 'utcmilliseconds',
    'yearmonth', 'utcyearmonth',
    'yearmonthdate', 'utcyearmonthdate',
    'yearmonthdatehours', 'utcyearmonthdatehours',
    'yearmonthdatehoursminutes', 'utcyearmonthdatehoursminutes',
    'yearmonthdatehoursminutesseconds', 'utcyearmonthdatehoursminutesseconds',
]
TIMEZONES = [None, tzlocal(), 'UTC', 'US/Pacific', 'US/Eastern']


@pytest.fixture
def dates():
    return pd.date_range('1999-12-31 12:00',
                         '2000-01-01 12:00',
                         freq='H')


@pytest.mark.parametrize('timezone', TIMEZONES[:3])
def test_datetimeindex_roundtrip(dates, timezone):
    dates = dates.tz_localize(timezone)
    timestamp = timeunit.date_to_timestamp(dates)
    dates2 = timeunit.timestamp_to_date(timestamp,
                                        tz=(dates.tz is not None),
                                        utc=(dates.tz is pytz.UTC))
    assert dates2.equals(dates)


@pytest.mark.parametrize('timezone', TIMEZONES[:3])
def test_timestamp_roundtrip(dates, timezone):
    date = dates.tz_localize(timezone)[0]
    timestamp = timeunit.date_to_timestamp(date)
    date2 = timeunit.timestamp_to_date(timestamp,
                                       tz=(date.tz is not None),
                                       utc=(date.tz is pytz.UTC))
    assert date2 == date


@pytest.mark.parametrize('timezone', TIMEZONES)
@pytest.mark.parametrize('unit', TIMEUNITS)
def test_timeunit_input_types(dates, timezone, unit):
    dates = dates.tz_localize(timezone)
    unit = getattr(timeunit, unit)

    timestamps = [unit(d) for d in dates]
    series = unit(pd.Series(dates))
    datetimeindex = unit(dates)

    assert datetimeindex.equals(pd.DatetimeIndex(series))
    assert datetimeindex.equals(pd.DatetimeIndex(timestamps))


@pytest.mark.parametrize('timezone', TIMEZONES)
@pytest.mark.parametrize('timeunit_name', TIMEUNITS)
def test_timeunit(dates, timezone, timeunit_name):
    timeunit_func = getattr(timeunit, timeunit_name)
    timeunit_calc = timeunit_func(dates.tz_localize(timezone))

    tz = 'UTC' if timeunit_name.startswith('utc') else tzlocal()
    dates = dates.tz_localize(timezone or tzlocal()).tz_convert(tz)

    to_check = [
        ('year', 'year', 1900),
        ('month', 'month', 1),
        ('date', 'day', 1),
        ('hours', 'hour', 0),
        ('minutes', 'minute', 0),
        ('seconds', 'second', 0),
        ('milliseconds', 'microsecond', 0)
    ]

    for name, attr, default in to_check:
        if name in timeunit_name:
            assert getattr(dates, attr).equals(getattr(timeunit_calc, attr))
        else:
            assert (getattr(timeunit_calc, attr) == default).all()
    assert (timeunit_calc.nanosecond == 0).all()
