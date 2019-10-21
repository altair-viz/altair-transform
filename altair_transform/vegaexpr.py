"""
Evaluate vega expressions language
"""
import datetime as dt
import math
import random
import sys
import pandas as pd
from dateutil import tz
import time as timemod
from typing import Any, Dict, Optional, Pattern, overload

from altair_transform.utils import evaljs


class _UndefinedType(object):
    def __repr__(self):
        return "undefined"


undefined = _UndefinedType()


def eval_vegajs(expression: str, datum: pd.DataFrame = None) -> pd.DataFrame:
    """Evaluate a vega expression"""
    namespace = {"datum": datum} if datum is not None else {}
    namespace.update(VEGAJS_NAMESPACE)
    return evaljs(expression, namespace)


# Type Coercion Functions
def isArray(value: Any) -> bool:
    """Returns true if value is an array, false otherwise."""
    return isinstance(value, list)


def isBoolean(value: Any) -> bool:
    """Returns true if value is a boolean (true or false), false otherwise."""
    return isinstance(value, bool)


def isDate(value: Any) -> bool:
    """Returns true if value is a Date object, false otherwise.

    This method will return false for timestamp numbers or
    date-formatted strings; it recognizes Date objects only.
    """
    return isinstance(value, dt.datetime)


def isDefined(value: Any) -> bool:
    """Returns true if value is a defined value, false if value equals undefined.

    This method will return true for null and NaN values.
    """
    # TODO: support implicitly undefined values?
    return value is not undefined


def isNumber(value: Any) -> bool:
    """Returns true if value is a number, false otherwise.

    NaN and Infinity are considered numbers.
    """
    return isinstance(value, (int, float))


def isObject(value: Any) -> bool:
    """Returns true if value is an object, false otherwise.

    Following JavaScript typeof convention, null values are considered objects.
    """
    return value is None or isinstance(value, dict)


def isRegExp(value: Any) -> bool:
    """
    Returns true if value is a RegExp (regular expression)
    object, false otherwise.
    """
    return isinstance(value, Pattern)


def isString(value: Any) -> bool:
    """Returns true if value is a string, false otherwise."""
    return isinstance(value, str)


def isValid(value: Any) -> bool:
    """Returns true if value is not null, undefined, or NaN."""
    return not (value is undefined or pd.isna(value))


# Type Coercion Functions
def toBoolean(value: Any) -> bool:
    """
    Coerces the input value to a boolean.
    Null values and empty strings are mapped to null.
    """
    return bool(value)


def toDate(value: Any) -> Optional[float]:
    """
    Coerces the input value to a Date instance.
    Null values and empty strings are mapped to null.
    If an optional parser function is provided, it is used to
    perform date parsing, otherwise Date.parse is used.
    """
    if isinstance(value, (float, int)):
        return value
    if value is None or value == "":
        return None
    return pd.to_datetime(value).timestamp() * 1000


def toNumber(value: Any) -> Optional[float]:
    """
    Coerces the input value to a number.
    Null values and empty strings are mapped to null.
    """
    if value is None or value == "":
        return None
    return float(value)


def toString(value: Any) -> Optional[str]:
    """
    Coerces the input value to a string.
    Null values and empty strings are mapped to null.
    """
    if value is None or value == "":
        return None
    if isinstance(value, float) and value % 1 == 0:
        return str(int(value))
    return str(value)


# Date/Time Functions
def now() -> float:
    """Returns the timestamp for the current time."""
    return round(timemod.time() * 1000, 0)


@overload
def datetime() -> dt.datetime:
    ...


@overload
def datetime(timestamp: float) -> dt.datetime:
    ...


@overload
def datetime(
    year: float,
    month: int,
    day: int = 0,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    millisecond: float = 0,
) -> dt.datetime:
    ...


def datetime(*args):
    """Returns a new Date instance.

    datetime()  # current time
    datetime(timestamp)
    datetime(year, month[, day, hour, min, sec, millisec])

    The month is 0-based, such that 1 represents February.
    """
    if len(args) == 0:
        return dt.datetime.now()
    elif len(args) == 1:
        return dt.datetime.fromtimestamp(0.001 * args[0])
    elif len(args) == 2:
        return dt.datetime(*args, 1)
    elif len(args) <= 7:
        args = list(map(int, args))
        args[1] += 1  # JS month is zero-based
        if len(args) == 2:
            args.append(0)  # Day is required in Python
        if len(args) == 7:
            args[6] = args[6] * 1000  # milliseconds to microseconds
        return dt.datetime(*args)
    else:
        raise ValueError("Too many arguments")


def date(datetime: dt.datetime) -> int:
    """
    Returns the day of the month for the given datetime value, in local time.
    """
    return datetime.day


def day(datetime: dt.datetime) -> int:
    """
    Returns the day of the week for the given datetime value, in local time.
    """
    return (datetime.weekday() + 1) % 7


def year(datetime: dt.datetime) -> int:
    """Returns the year for the given datetime value, in local time."""
    return datetime.year


def quarter(datetime: dt.datetime) -> int:
    """
    Returns the quarter of the year (0-3) for the given datetime value,
    in local time.
    """
    return (datetime.month - 1) // 3


def month(datetime: dt.datetime) -> int:
    """
    Returns the (zero-based) month for the given datetime value, in local time.
    """
    return datetime.month - 1


def hours(datetime: dt.datetime) -> int:
    """
    Returns the hours component for the given datetime value, in local time.
    """
    return datetime.hour


def minutes(datetime: dt.datetime) -> int:
    """
    Returns the minutes component for the given datetime value, in local time.
    """
    return datetime.minute


def seconds(datetime: dt.datetime) -> int:
    """
    Returns the seconds component for the given datetime value, in local time.
    """
    return datetime.second


def milliseconds(datetime: dt.datetime) -> float:
    """
    Returns the milliseconds component for the given datetime value,
    in local time.
    """
    return datetime.microsecond / 1000


def time(datetime: dt.datetime) -> float:
    """Returns the epoch-based timestamp for the given datetime value."""
    return datetime.timestamp() * 1000


def timezoneoffset(datetime):
    # TODO: use tzlocal?
    raise NotImplementedError("timezoneoffset()")


def utc(
    year: int,
    month: int = 0,
    day: int = 1,
    hour: int = 0,
    min: int = 0,
    sec: int = 0,
    millisec: int = 0,
) -> float:
    """
    Returns a timestamp for the given UTC date.
    The month is 0-based, such that 1 represents February.
    """
    return (
        dt.datetime(
            int(year),
            int(month) + 1,
            int(day),
            int(hour),
            int(min),
            int(sec),
            int(millisec * 1000),
            tzinfo=dt.timezone.utc,
        ).timestamp()
        * 1000
    )


def utcdate(datetime):
    """Returns the day of the month for the given datetime value, in UTC time."""
    return date(datetime.astimezone(tz.UTC))


def utcday(datetime):
    """Returns the day of the week for the given datetime value, in UTC time."""
    return day(datetime.astimezone(tz.UTC))


def utcyear(datetime):
    """Returns the year for the given datetime value, in UTC time."""
    return year(datetime.astimezone(tz.UTC))


def utcquarter(datetime):
    """Returns the quarter of the year (0-3) for the given datetime value, in UTC time."""
    return quarter(datetime.astimezone(tz.UTC))


def utcmonth(datetime):
    """Returns the (zero-based) month for the given datetime value, in UTC time."""
    return month(datetime.astimezone(tz.UTC))


def utchours(datetime):
    """Returns the hours component for the given datetime value, in UTC time."""
    return hours(datetime.astimezone(tz.UTC))


def utcminutes(datetime):
    """Returns the minutes component for the given datetime value, in UTC time."""
    return minutes(datetime.astimezone(tz.UTC))


def utcseconds(datetime):
    """Returns the seconds component for the given datetime value, in UTC time."""
    return seconds(datetime.astimezone(tz.UTC))


def utcmilliseconds(datetime):
    """Returns the milliseconds component for the given datetime value, in UTC time."""
    return milliseconds(datetime.astimezone(tz.UTC))


# From https://vega.github.io/vega/docs/expressions/
VEGAJS_NAMESPACE: Dict[str, Any] = {
    # Constants
    "null": None,
    "true": True,
    "false": False,
    "undefined": undefined,
    "NaN": math.nan,
    "E": math.e,
    "LN2": math.log(2),
    "LN10": math.log(10),
    "LOG2E": math.log2(math.e),
    "LOG10E": math.log10(math.e),
    "MAX_VALUE": sys.float_info.max,
    "MIN_VALUE": sys.float_info.min,
    "PI": math.pi,
    "SQRT1_2": math.sqrt(0.5),
    "SQRT2": math.sqrt(2),
    # Type Checking
    "isArray": isArray,
    "isBoolean": isBoolean,
    "isDate": isDate,
    "isDefined": isDefined,
    "isNumber": isNumber,
    "isObject": isObject,
    "isRegExp": isRegExp,
    "isString": isString,
    "isValid": isValid,
    # Type Coercion
    "toBoolean": toBoolean,
    "toDate": toDate,
    "toNumber": toNumber,
    "toString": toString,
    # Control Flow Functions
    "if": lambda test, if_value, else_value: if_value if test else else_value,
    # Math Functions
    "isNaN": math.isnan,
    "isFinite": math.isfinite,
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "clamp": lambda val, low, hi: max(min(val, hi), low),
    "cos": math.cos,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "max": max,
    "min": min,
    "pow": math.pow,
    "random": random.random,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
    # Date/Time Functions
    "now": now,
    "datetime": datetime,
    "date": date,
    "day": day,
    "year": year,
    "quarter": quarter,
    "month": month,
    "hours": hours,
    "minutes": minutes,
    "seconds": seconds,
    "milliseconds": milliseconds,
    "time": time,
    "timezoneoffset": timezoneoffset,
    "utc": utc,
    "utcdate": utcdate,
    "utcday": utcday,
    "utcyear": utcyear,
    "utcquarter": utcquarter,
    "utcmonth": utcmonth,
    "utchours": utchours,
    "utcminutes": utcminutes,
    "utcseconds": utcseconds,
    "utcmilliseconds": utcmilliseconds,
    # TODOs:
    # Statistical Functions
    # Remaining Date/Time Functions
    # Array Functions
    # String Functions
    # Object Functions
    # Formatting Functions
    # RegExp Functions
}
