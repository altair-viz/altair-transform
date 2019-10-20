import pytest
import numpy as np
from altair_transform.vegaexpr import eval_vegajs

# Most parsing is tested in the parser; here we just test a sampling of the
# variables and functions defined in the vegaexpr namespace.

EXPRESSIONS = {
    "2 * PI": 2 * np.pi,
    "1 / SQRT2": 1.0 / np.sqrt(2),
    "LOG2E + LN10": np.log2(np.e) + np.log(10),
    "isArray([1, 2, 3])": True,
    "isNumber(3.5)": True,
    'isString("abc")': True,
    "toBoolean(1)": True,
    "toBoolean(0)": False,
    'if(4 > PI, "yes", "no")': "yes",
    "pow(sin(PI), 2) + pow(cos(PI), 2)": 1,
    "floor(1.5) == ceil(0.5)": True,
    "max(1, 2, 3) == min(3, 4, 5)": True,
    "isDate(datetime(2019, 0, 1, 2, 34, 56, 789))": True,
    "year(datetime(2019, 0, 1, 2, 34, 56, 789))": 2019,
    "quarter(datetime(2019, 0, 1, 2, 34, 56, 789))": 0,
    "month(datetime(2019, 0, 1, 2, 34, 56, 789))": 0,
    "date(datetime(2019, 0, 1, 2, 34, 56, 789))": 1,
    "hours(datetime(2019, 0, 1, 2, 34, 56, 789))": 2,
    "minutes(datetime(2019, 0, 1, 2, 34, 56, 789))": 34,
    "seconds(datetime(2019, 0, 1, 2, 34, 56, 789))": 56,
    "milliseconds(datetime(2019, 0, 1, 2, 34, 56, 789))": 789,
}


@pytest.mark.parametrize("expression,expected", EXPRESSIONS.items())
def test_vegajs_expressions(expression, expected):
    result = eval_vegajs(expression)
    if isinstance(result, float):
        assert np.allclose(result, expected)
    else:
        assert result == expected
