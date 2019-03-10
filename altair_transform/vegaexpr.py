"""
Evaluate vega expressions language
"""
import datetime as dtmodule
import math
import random
import sys
import pandas as pd
import time as timemod
from typing import Pattern

import pytz

from altair_transform.utils import evaljs


def eval_vegajs(expression, datum=None):
    """Evaluate a vega expression"""
    namespace = {"datum": datum} if datum is not None else {}
    namespace.update(VEGAJS_NAMESPACE)
    return evaljs(expression, namespace)


# Type Coercion Functions
def isArray(value):
    """Returns true if value is an array, false otherwise."""
    return isinstance(value, list),


def isBoolean(value):
    """Returns true if value is a boolean (true or false), false otherwise."""
    return isinstance(value, bool)


def isDate(value):
    """Returns true if value is a Date object, false otherwise.

    This method will return false for timestamp numbers or
    date-formatted strings; it recognizes Date objects only.
    """
    return isinstance(value, dtmodule.datetime)


def isNumber(value):
    """Returns true if value is a number, false otherwise.

    NaN and Infinity are considered numbers.
    """
    return isinstance(value, (int, float))


def isObject(value):
    """Returns true if value is an object, false otherwise.

    Following JavaScript typeof convention, null values are considered objects.
    """
    return value is None or isinstance(value, dict)


def isRegExp(value):
    """
    Returns true if value is a RegExp (regular expression)
    object, false otherwise.
    """
    return isinstance(value, Pattern)


def isString(value):
    """Returns true if value is a string, false otherwise."""
    return isinstance(value, str)


# Type Coercion Functions
def toBoolean(value):
    """
    Coerces the input value to a boolean.
    Null values and empty strings are mapped to null.
    """
    return bool(value)


def toDate(value):
    """
    Coerces the input value to a Date instance.
    Null values and empty strings are mapped to null.
    If an optional parser function is provided, it is used to
    perform date parsing, otherwise Date.parse is used.
    """
    if value is None or value == "":
        return None
    return pd.to_datetime(value)


def toNumber(value):
    """
    Coerces the input value to a number.
    Null values and empty strings are mapped to null.
    """
    if value is None or value == "":
        return None
    return float(value)


def toString(value):
    """
    Coerces the input value to a string.
    Null values and empty strings are mapped to null.
    """
    if value is None:
        return None
    return str(value)


# Date/Time Functions
def now():
    """Returns the timestamp for the current time."""
    return round(timemod.time() * 1000, 0)


def datetime(year, month, day=0, hour=0, min=0, sec=0, millisec=0):
    """Returns a new Date instance.
    The month is 0-based, such that 1 represents February.
    """
    # TODO: do we need a local timezone?
    return dtmodule.datetime(year, month + 1, day, hour,
                             min, sec, millisec * 1000)


def date(datetime):
    """
    Returns the day of the month for the given datetime value, in local time.
    """
    return datetime.day


def day(datetime):
    """
    Returns the day of the week for the given datetime value, in local time.
    """
    return (datetime.weekday() + 1) % 7


def year(datetime):
    """Returns the year for the given datetime value, in local time."""
    return datetime.year


def quarter(datetime):
    """
    Returns the quarter of the year (0-3) for the given datetime value,
    in local time.
    """
    return (datetime.month - 1) // 3


def month(datetime):
    """
    Returns the (zero-based) month for the given datetime value, in local time.
    """
    return datetime.month - 1


def hours(datetime):
    """
    Returns the hours component for the given datetime value, in local time.
    """
    return datetime.hour


def minutes(datetime):
    """
    Returns the minutes component for the given datetime value, in local time.
    """
    return datetime.minute


def seconds(datetime):
    """
    Returns the seconds component for the given datetime value, in local time.
    """
    return datetime.second


def milliseconds(datetime):
    """
    Returns the milliseconds component for the given datetime value,
    in local time.
    """
    return datetime.microsecond / 1000


def time(datetime):
    """Returns the epoch-based timestamp for the given datetime value."""
    return datetime.timestamp() * 1000


def timezoneoffset(datetime):
    # TODO: use tzlocal?
    raise NotImplementedError("timezoneoffset()")


def utc(year, month, day=0, hour=0, min=0, sec=0, millisec=0):
    """
    Returns a timestamp for the given UTC date.
    The month is 0-based, such that 1 represents February.
    """
    return dtmodule.datetime(year, month + 1, day, hour, min, sec,
                             millisec * 1000, tzinfo=pytz.utc)


# From https://vega.github.io/vega/docs/expressions/
VEGAJS_NAMESPACE = {
    # Constants
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
    "isNumber": isNumber,
    "isObject": isObject,
    "isRegExp": isRegExp,
    "isString": isString,

    # Type Coercion
    "toBoolean": toBoolean,
    "toDate": toDate,
    "toNumber": toNumber,
    "toString": toString,

    # Control Flow Functions
    "if": lambda test, if_value, else_value: if_value if test else else_value,

    # Math Functions
    "isNan": math.isnan,
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

    # TODOs:
    # Remaining Date/Time Functions
    # Array Functions
    # String Functions
    # Object Functions
    # Formatting Functions
    # RegExp Functions
}
