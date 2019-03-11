"""Functionality to evaluate contents of the ast"""
from functools import singledispatch, wraps
import operator
from typing import Any, Union

from altair_transform.utils import ast, Parser


def evaljs(expression: Union[str, ast.Expr], namespace: dict = None):
    """Evaluate a javascript expression, optionally with a namespace."""
    if isinstance(expression, str):
        parser = Parser()
        expression = parser.parse(expression)
    return visit(expression, namespace or {})


@singledispatch
def visit(obj: Any, namespace: dict):
    return obj


@visit.register
def _visit_expr(obj: ast.Expr, namespace: dict):
    return obj.value


@visit.register
def _visit_binop(obj: ast.BinOp, namespace: dict):
    if obj.op not in BINARY_OPERATORS:
        raise NotImplementedError(f"Binary Operator A {obj.op} B")
    op = BINARY_OPERATORS[obj.op]
    return op(visit(obj.lhs, namespace), visit(obj.rhs, namespace))


@visit.register
def _visit_unop(obj: ast.UnOp, namespace: dict):
    if obj.op not in UNARY_OPERATORS:
        raise NotImplementedError(f"Unary Operator {obj.op}x")
    op = UNARY_OPERATORS[obj.op]
    return op(visit(obj.rhs, namespace))


@visit.register
def _visit_ternop(obj: ast.TernOp, namespace: dict):
    if obj.op not in TERNARY_OPERATORS:
        raise NotImplementedError(
            f"Ternary Operator A {obj.op[0]} B {obj.op[1]} C")
    op = TERNARY_OPERATORS[obj.op]
    return op(visit(obj.lhs, namespace),
              visit(obj.mid, namespace),
              visit(obj.rhs, namespace))


@visit.register
def _visit_number(obj: ast.Number, namespace: dict):
    return obj.value


@visit.register
def _visit_string(obj: ast.String, namespace: dict):
    return obj.value


@visit.register
def _visit_global(obj: ast.Global, namespace: dict):
    if obj.name not in namespace:
        raise NameError("{0} is not a valid name".format(obj.name))
    return namespace[obj.name]


@visit.register
def _visit_name(obj: ast.Name, namespace: dict):
    return obj.name


@visit.register
def _visit_list(obj: ast.List, namespace: dict):
    return [visit(entry, namespace) for entry in obj.entries]


@visit.register
def _visit_object(obj: ast.Object, namespace: dict):
    def _visit(entry):
        if isinstance(entry, tuple):
            return tuple(visit(e, namespace) for e in entry)
        if isinstance(entry, ast.Name):
            return (visit(entry, namespace),
                    visit(ast.Global(entry.name), namespace))
    return dict(_visit(entry) for entry in obj.entries)


@visit.register
def _visit_attr(obj: ast.Attr, namespace: dict):
    obj_ = visit(obj.obj, namespace)
    attr = visit(obj.attr, namespace)
    if isinstance(obj_, dict):
        return obj_[attr]
    return getattr(obj_, attr)


@visit.register
def _visit_item(obj: ast.Item, namespace: dict):
    obj_ = visit(obj.obj, namespace)
    item = visit(obj.item, namespace)
    if isinstance(obj_, list) and isinstance(item, float):
        item = int(item)
    return obj_[item]


@visit.register
def _visit_func(obj: ast.Func, namespace: dict):
    func = visit(obj.func, namespace)
    args = [visit(arg, namespace) for arg in obj.args]
    return func(*args)


def int_inputs(func):
    @wraps(func)
    def wrapper(*args):
        return float(func(*map(int, args)))
    return wrapper


@int_inputs
def zerofill_rshift(lhs, rhs):
    if lhs < 0:
        lhs = lhs + 0x100000000
    return lhs >> rhs


# TODO: do implicit type conversions ugh...
UNARY_OPERATORS = {
    '~': int_inputs(operator.inv),
    '-': operator.neg,
    '+': operator.pos,
    '!': operator.not_,
}


BINARY_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "%": operator.mod,
    "&": int_inputs(operator.and_),
    "|": int_inputs(operator.or_),
    "^": int_inputs(operator.xor),
    "<<": int_inputs(operator.lshift),
    ">>": int_inputs(operator.rshift),
    ">>>": zerofill_rshift,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "===": operator.eq,
    "!=": operator.ne,
    "!==": operator.ne,
    "&&": lambda a, b: a and b,
    "||": lambda a, b: a or b,
}


TERNARY_OPERATORS = {
    ("?", ":"): lambda a, b, c: b if a else c
}
