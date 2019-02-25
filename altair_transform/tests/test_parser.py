import pytest

from altair_transform.parser import Parser

EXPRESSIONS = [
    "1 + 1",
    "2 * 4",
    "3 / 5",
    "4 - 6",
    "2 ** 3",
    "2 * 3 + 4 / 5",
    "2 + 3 * 4 - 5",
    "2 * (3 + 4)",
    "(2 * 3) + 4",
    "2 * 3 ** 4"
]

EXPRESSIONS_WITH_NAMES = [
    "A * 1",
    "B + 2",
    "3 * (4 + C)"
]

class Bunch(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


@pytest.fixture
def parser():
    return Parser()


@pytest.mark.parametrize('expression', EXPRESSIONS)
def test_simple_eval(expression, parser):
    assert eval(expression) == parser.parse(expression)


@pytest.mark.parametrize('expression', EXPRESSIONS_WITH_NAMES)
def test_name_eval(expression, parser):
    names = {'A': 5, 'B': 6, 'C': 7}
    assert eval(expression, names) == parser.parse(expression, names)


def test_attribute_access(parser):
    names = {'A': Bunch(B=4), 'C': 5}

    expression = 'A.B + C / 5'
    assert eval(expression, names) == parser.parse(expression, names)

    expression = '(A).B + C * 10'
    assert eval(expression, names) == parser.parse(expression, names)

    with pytest.raises(ValueError) as err:
        parser.parse("1.B")