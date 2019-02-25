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


@pytest.fixture
def parser():
    return Parser()


@pytest.mark.parametrize('expression', EXPRESSIONS)
def test_simple_eval(expression, parser):
    assert eval(expression) == parser.parse(expression)