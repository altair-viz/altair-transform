import pytest

from altair_transform.calculate import Parser

EXPRESSIONS = [
    "1 + 1",
    "2 * 4",
    "3 / 5",
    "4 - 6",
    "2 ** 3",
    "2.5 * 3 + 4 / 5.2",
    "2.5 + 3 * 4 - 5.0",
    "2.5 * (3 + 4)",
    "(2 * 3) + 4",
    "2 * 3 ** 4",
    "1.5 + 2. * .3",
    "-0.6 * (2 / 1.5)"
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

    def __getitem__(self, item):
        return getattr(self, item)


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
    names = {'A': Bunch(B=4), 'C': 5, 'f': lambda x: x ** 2}

    expression = 'A.B + C / 5'
    assert eval(expression, names) == parser.parse(expression, names)

    expression = 'A["B"] + C / 5'
    assert eval(expression, names) == parser.parse(expression, names)

    expression = '(A).B + C * 10'
    assert eval(expression, names) == parser.parse(expression, names)

    with pytest.raises(ValueError) as err:
        parser.parse("1.B")


def test_function_calls(parser):
    names = {'A': 3, 'C': 5, 'f': lambda x=2: x ** 2, 'mul': lambda x, y: x * y}

    expression = 'A * f()'
    assert eval(expression, names) == parser.parse(expression, names)

    expression = 'A * f(C)'
    assert eval(expression, names) == parser.parse(expression, names)

    expression = 'mul(A, 2)'
    assert eval(expression, names) == parser.parse(expression, names)


def test_string_variants(parser):
    names = {'f': lambda *args: args}
    expression = r"""f("abc", 'abc', "a\"b\"c", 'a\'b\'c', 'abc\\', "abc\\")"""
    assert eval(expression, names) == parser.parse(expression, names)


@pytest.mark.parametrize('expression', ["[]", "[1 + 1]", "[A, 'foo', 23 * B]"])
def test_lists(parser, expression):
    names = {'A': 1, 'B': 2}
    assert eval(expression, names) == parser.parse(expression, names)


@pytest.mark.parametrize('expression', ["{}", "{'a': 4}", "{'a': 5, 'b': 5}"])
def test_objects(parser, expression):
    assert eval(expression) == parser.parse(expression)


def test_jsobject(parser):
    names = {'a': 1, 'b': 2}
    js_obj = "{a, b, c: 3, 'd': 4, 1: 5}"
    assert parser.parse(js_obj, names) == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 1: 5}