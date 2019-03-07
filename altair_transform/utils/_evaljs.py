"""Functionality to evaluate contents of the ast"""
from functools import wraps
import operator

from altair_transform.utils import ast, Parser


def evaljs(expression, namespace=None):
    if isinstance(expression, str):
        parser = Parser()
        expression = parser.parse(expression)
    return Evaluate(namespace).visit(expression)


class Visitor:
    """Class implementing the external visitor pattern"""
    def visit(self, obj, *args, **kwargs):
        methods = (getattr(self, 'visit_' + cls.__name__, None)
                   for cls in obj.__class__.__mro__)
        method = next((m for m in methods if m), self.generic_visit)
        return method(obj, *args, **kwargs)

    def generic_visit(self, obj, *args, **kwargs):
        raise NotImplementedError("visitor for {0}".format(obj))


class Evaluate(Visitor):
    def __init__(self, namespace=None):
        self.namespace = namespace or {}

    def visit_Expr(self, obj):
        return obj.value

    def visit_BinOp(self, obj):
        if obj.op not in BINARY_OPERATORS:
            raise NotImplementedError("Binary Operator A{0}B".format(obj.op))
        op = BINARY_OPERATORS[obj.op]
        return op(self.visit(obj.lhs), self.visit(obj.rhs))

    def visit_UnOp(self, obj):
        if obj.op not in UNARY_OPERATORS:
            raise NotImplementedError("Unary Operator {0}x".format(obj.op))
        op = UNARY_OPERATORS[obj.op]
        return op(self.visit(obj.rhs))

    def visit_TernOp(self, obj):
        if obj.op != ('?', ':'):
            raise NotImplementedError("Ternary Operator A {0} B {1} C",
                                      *obj.op)
        return (self.visit(obj.mid) if self.visit(obj.lhs)
                else self.visit(obj.rhs))

    def visit_Number(self, obj):
        return obj.value

    def visit_String(self, obj):
        return obj.value

    def visit_Global(self, obj):
        if obj.name not in self.namespace:
            raise NameError("{0} is not a valid name".format(obj.name))
        return self.namespace[obj.name]

    def visit_Name(self, obj):
        return obj.name

    def visit_List(self, obj):
        return [self.visit(entry) for entry in obj.entries]

    def visit_Object(self, obj):
        def visit(entry):
            if isinstance(entry, tuple):
                return tuple(self.visit(e) for e in entry)
            elif isinstance(entry, ast.Name):
                return (self.visit(entry), self.visit_Global(entry))
        return dict(visit(entry) for entry in obj.entries)

    def visit_Attr(self, obj):
        obj_ = self.visit(obj.obj)
        attr = self.visit(obj.attr)
        if isinstance(obj_, dict):
            return obj_[attr]
        else:
            return getattr(obj_, attr)

    def visit_Item(self, obj):
        obj_ = self.visit(obj.obj)
        item = self.visit(obj.item)
        if isinstance(obj_, list) and isinstance(item, float):
            item = int(item)
        return obj_[item]

    def visit_Func(self, obj):
        func = self.visit(obj.func)
        args = [self.visit(arg) for arg in obj.args]
        return func(*args)

    def generic_visit(self, obj):
        return obj


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
