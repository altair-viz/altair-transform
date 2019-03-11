from functools import singledispatch
from typing import Any

import altair as alt
import pandas as pd


@singledispatch
def visit(transform: Any, df: pd.DataFrame):
    raise NotImplementedError("transform of type {0}".format(type(transform)))


@visit.register
def visit_list(transform: list, df: pd.DataFrame):
    for t in transform:
        df = visit(t, df)
    return df


@visit.register
def visit_dict(transform: dict, df: pd.DataFrame):
    transform = alt.Transform.from_dict(transform)
    return visit(transform, df)
