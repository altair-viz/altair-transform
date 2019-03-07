from functools import singledispatch, wraps

def singledispatch_method(method):
    """single dispatch decorator for class methods"""
    d = singledispatch(method)
    @wraps(method)
    def wrapper(*args, **kw):
        return d.dispatch(type(args[1]))(*args, **kw)
    wrapper.register = d.register
    return wrapper