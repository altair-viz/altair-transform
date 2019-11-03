import datetime as dt
import pytest
import numpy as np
from altair_transform.vegaexpr import eval_vegajs, undefined

# Most parsing is tested in the parser; here we just test a sampling of the
# variables and functions defined in the vegaexpr namespace.

EXPRESSIONS = {
    "null": None,
    "true": True,
    "false": False,
    "{}[1]": undefined,
    "{}.foo": undefined,
    "[][0]": undefined,
    "2 * PI": 2 * np.pi,
    "1 / SQRT2": 1.0 / np.sqrt(2),
    "LOG2E + LN10": np.log2(np.e) + np.log(10),
    "isArray([1, 2, 3])": True,
    "isBoolean(false)": True,
    "isBoolean(true)": True,
    "isBoolean(1)": False,
    "isDate(datetime(2019, 1, 1))": True,
    "isDate('2019-01-01')": False,
    "isDefined(null)": True,
    "isDefined({}[1])": False,
    "isNumber(3.5)": True,
    "isNumber(now())": True,
    "isString('abc')": True,
    'isString("abc")': True,
    "isObject({a:2})": True,
    "isObject({'a':2})": True,
    "isValid(null)": False,
    "isValid(NaN)": False,
    "isValid({}[1])": False,
    "isValid(0)": True,
    "toBoolean(1)": True,
    "toBoolean(0)": False,
    "toDate('')": None,
    "toDate(null)": None,
    "toDate(1547510400000)": 1547510400000,
    "toDate('2019-01-15')": 1547510400000,
    "toNumber('1234.5')": 1234.5,
    "toNumber('')": None,
    "toNumber(null)": None,
    "toString(123)": "123",
    "toString(0.5)": "0.5",
    "toString('')": None,
    "toString(null)": None,
    "toString(123)": "123",
    "toString('123')": "123",
    'if(4 > PI, "yes", "no")': "yes",
    "pow(sin(PI), 2) + pow(cos(PI), 2)": 1,
    "floor(1.5) == ceil(0.5)": True,
    "max(1, 2, 3) == min(3, 4, 5)": True,
    "time(datetime(1546338896789))": 1546338896789,
    "isDate(datetime())": True,
    "datetime(1546329600000)": dt.datetime.fromtimestamp(1546329600),
    "datetime(2019, 0, 1)": dt.datetime(2019, 1, 1),
    "year(datetime(2019, 0, 1, 2, 34, 56, 789))": 2019,
    "quarter(datetime(2019, 0, 1, 2, 34, 56, 789))": 0,
    "month(datetime(2019, 0, 1, 2, 34, 56, 789))": 0,
    "date(datetime(2019, 0, 1, 2, 34, 56, 789))": 1,
    "day(datetime(2019, 0, 1, 2, 34, 56, 789))": 2,
    "hours(datetime(2019, 0, 1, 2, 34, 56, 789))": 2,
    "minutes(datetime(2019, 0, 1, 2, 34, 56, 789))": 34,
    "seconds(datetime(2019, 0, 1, 2, 34, 56, 789))": 56,
    "milliseconds(datetime(2019, 0, 1, 2, 34, 56, 789))": 789,
    "utc(2019, 0, 1, 2, 34, 56, 789)": 1546310096789,
    "utcyear(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 2019,
    "utcquarter(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 0,
    "utcmonth(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 0,
    "utcdate(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 1,
    "utcday(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 2,
    "utchours(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 2,
    "utcminutes(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 34,
    "utcseconds(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 56,
    "utcmilliseconds(datetime(utc(2019, 0, 1, 2, 34, 56, 789)))": 789,
    "parseInt('1234 years')": 1234,
    "parseInt('2A', 16)": 42,
    "parseFloat('  3.125 is close to pi')": 3.125,
    "indexof('ABCABC', 'C')": 2,
    "lastindexof('ABCABC', 'C')": 5,
    "length('ABCABC')": 6,
    "lower('AbC')": "abc",
    "pad('abc', 6, 'x', 'left')": "xxxabc",
    "pad('abc', 6, 'x', 'right')": "abcxxx",
    "pad('abc', 6, 'x', 'center')": "xabcxx",
    "replace('ABCDABCD', 'BC', 'xx')": "AxxDABCD",
    "split('AB CD EF', ' ')": ["AB", "CD", "EF"],
    "substring('ABCDEF', 3, -1)": "ABC",
    "slice('ABCDEF', 3, -1)": "DE",
    "trim('   ABC   ')": "ABC",
    "truncate('1234567', 4, 'right', 'x')": "123x",
    "truncate('1234567', 4, 'left', 'x')": "x567",
    "truncate('1234567', 4, 'center', 'x')": "12x7",
    "upper('AbC')": "ABC",
    "extent([5, {}[1], 2, null, 4, NaN, 1])": [1, 5],
    "clampRange([5, 2], 1, 7)": [2, 5],
    "clampRange([5, 2], 3, 7)": [3, 6],
    "clampRange([5, 2], 0, 4)": [1, 4],
    "clampRange([5, 2], 3, 4)": [3, 4],
    "inrange(4, [3, 4])": True,
    "inrange(4, [4, 5])": True,
    "inrange(4, [5, 7])": False,
    "join(['a', 'b', 'c'])": "a,b,c",
    "join(['a', 'b', 'c'], '-')": "a-b-c",
    "lerp([0, 50], 0.5)": 25.0,
    "peek([1, 2, 3])": 3,
    "reverse([1, 2, 3])": [3, 2, 1],
    "sequence(3)": [0, 1, 2],
    "sequence(1, 4)": [1, 2, 3],
    "sequence(0, 2, 0.5)": [0, 0.5, 1, 1.5],
    "slice([1, 2, 3, 4], 1, 3)": [2, 3],
    "span([0, 2, 4])": 4,
}


@pytest.mark.parametrize("expression,expected", EXPRESSIONS.items())
def test_vegajs_expressions(expression, expected):
    result = eval_vegajs(expression)
    if isinstance(result, float):
        assert np.allclose(result, expected)
    else:
        assert result == expected
