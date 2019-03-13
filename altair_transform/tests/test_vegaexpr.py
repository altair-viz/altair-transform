import pytest
import numpy as np
from altair_transform.vegaexpr import eval_vegajs

# Most parsing is tested in the parser; here we just test a sampling of the
# variables and functions defined in the vegaexpr namespace.

EXPRESSIONS = {
    '2 * PI': 2 * np.pi,
    '1 / SQRT2': 1. / np.sqrt(2),
    'LOG2E + LN10': np.log2(np.e) + np.log(10),
    'isArray([1, 2, 3])': True,
    'isNumber(3.5)': True,
    'isString("abc")': True,
    'toBoolean(1)': True,
    'toBoolean(0)': False,
    'if(4 > PI, "yes", "no")': 'yes',
    'isDate(datetime(2019, 0, 1))': True,
    'pow(sin(PI), 2) + pow(cos(PI), 2)': 1,
    'year(datetime(2019, 0, 15))': 2019,
    'month(datetime(2019, 0, 15))': 0,
    'date(datetime(2019, 0, 15))': 15,
    'floor(1.5) == ceil(0.5)': True,
    'max(1, 2, 3) == min(3, 4, 5)': True,
}


@pytest.mark.parametrize('expression,expected', EXPRESSIONS.items())
def test_vegajs_expressions(expression, expected):
    result = eval_vegajs(expression)
    if isinstance(result, float):
        assert np.allclose(result, expected)
    else:
        assert result == expected
