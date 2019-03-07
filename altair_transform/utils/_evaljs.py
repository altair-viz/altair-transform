"""Functionality to evaluate contents of the ast"""
from functools import singledispatch, wraps
import operator

from altair_transform.utils import ast, Parser


def evaljs(expression, namespace=None):
    if isinstance(expression, str):
        parser = Parser()
        expression = parser.parse(expression)
    return EvalJS(namespace).visit(expression)


class EvalJS():
    """Visitor pattern to evaluate ASTs of javascript expressions."""
    def __init__(self, namespace=None):
        self.namespace = namespace or {}

    def __dispatch(func):
        """single dispatch decorator for class methods"""
        disp = singledispatch(func)
        @wraps(func)
        def wrapper(*args, **kw):
            return disp.dispatch(type(args[1]))(*args, **kw)
        wrapper.register = disp.register
        return wrapper

    @__dispatch
    def visit(self, obj):
        return obj

    @visit.register(ast.Expr)
    def _(self, obj):
        return obj.value

    @visit.register(ast.BinOp)
    def _(self, obj):
        if obj.op not in BINARY_OPERATORS:
            raise NotImplementedError("Binary Operator A{0}B".format(obj.op))
        op = BINARY_OPERATORS[obj.op]
        return op(self.visit(obj.lhs), self.visit(obj.rhs))

    @visit.register(ast.UnOp)
    def _(self, obj):
        if obj.op not in UNARY_OPERATORS:
            raise NotImplementedError("Unary Operator {0}x".format(obj.op))
        op = UNARY_OPERATORS[obj.op]
        return op(self.visit(obj.rhs))

    @visit.register(ast.TernOp)
    def _(self, obj):
        if obj.op != ('?', ':'):
            raise NotImplementedError("Ternary Operator A {0} B {1} C",
                                      *obj.op)
        return (self.visit(obj.mid) if self.visit(obj.lhs)
                else self.visit(obj.rhs))

    @visit.register(ast.Number)
    def _(self, obj):
        return obj.value

    @visit.register(ast.String)
    def _(self, obj):
        return obj.value

    @visit.register(ast.Global)
    def _(self, obj):
        if obj.name not in self.namespace:
            raise NameError("{0} is not a valid name".format(obj.name))
        return self.namespace[obj.name]

    @visit.register(ast.Name)
    def _(self, obj):
        return obj.name

    @visit.register(ast.List)
    def _(self, obj):
        return [self.visit(entry) for entry in obj.entries]

    @visit.register(ast.Object)
    def _(self, obj):
        def visit(entry):
            if isinstance(entry, tuple):
                return tuple(self.visit(e) for e in entry)
            elif isinstance(entry, ast.Name):
                return (self.visit(entry), self.visit(ast.Global(entry.name)))
        return dict(visit(entry) for entry in obj.entries)

    @visit.register(ast.Attr)
    def _(self, obj):
        obj_ = self.visit(obj.obj)
        attr = self.visit(obj.attr)
        if isinstance(obj_, dict):
            return obj_[attr]
        else:
            return getattr(obj_, attr)

    @visit.register(ast.Item)
    def _(self, obj):
        obj_ = self.visit(obj.obj)
        item = self.visit(obj.item)
        if isinstance(obj_, list) and isinstance(item, float):
            item = int(item)
        return obj_[item]

    @visit.register(ast.Func)
    def _(self, obj):
        func = self.visit(obj.func)
        args = [self.visit(arg) for arg in obj.args]
        return func(*args)


def int_inputs(func):
    @wraps(func)
    def wrapper(*args):
        return float(func(*map(int, args)))
    return wrapper


@int_inputs
def zerofill_rshift(a, b):
    # TODO: make this work correctly
    return operator.rshift(a, b)


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
