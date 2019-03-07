import operator
import functools

import pytest

from altair_transform.utils import evaljs
from ._testcases import extract, EXPRESSIONS, JSONLY_EXPRESSIONS, NAMES

@pytest.fixture
def names():
    return NAMES

@pytest.mark.parametrize('expression', extract(EXPRESSIONS))
def test_expressions(expression, names):
    assert eval(expression, names) == evaljs(expression, names)

@pytest.mark.parametrize('expression,output', JSONLY_EXPRESSIONS)
def test_jsonly_expressions(expression, output, names):
    assert evaljs(expression, names) == output