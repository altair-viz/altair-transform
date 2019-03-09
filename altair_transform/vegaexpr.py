"""
Evaluate vega expressions language
"""
import datetime
import math
import random
import sys
import pandas as pd
from typing import Pattern

from altair_transform.utils import evaljs


def eval_vegajs(expression, datum=None):
    """Evaluate a vega expression"""
    namespace = {"datum": datum} if datum is not None else {}
    namespace.update(VEGAJS_NAMESPACE)
    return evaljs(expression, namespace)


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
    "isArray": lambda x: isinstance(x, list),
    "isBoolean": lambda x: isinstance(x, bool),
    "isDate": lambda x: isinstance(x, datetime.datetime),
    "isNumber": lambda x: isinstance(x, (int, float)),
    "isObject": lambda x: isinstance(x, dict),
    "isRegExp": lambda x: isinstance(x, Pattern),
    "isString": lambda x: isinstance(x, str),

    # Type Coercion
    "toBoolean": bool,
    "toDate": pd.to_datetime,
    "toNumber": float,
    "toString": str,

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

    # TODOs:
    # Date/Time Functions
    # Array Functions
    # String Functions
    # Object Functions
    # Formatting Functions
    # RegExp Functions
}
