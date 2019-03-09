from functools import singledispatch, wraps


def singledispatch_method(method):
    """single dispatch decorator for class methods"""
    disp = singledispatch(method)

    @wraps(method)
    def wrapper(*args, **kw):
        return disp.dispatch(type(args[1]))(*args, **kw)

    wrapper.register = disp.register
    return wrapper
