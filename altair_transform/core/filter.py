from functools import singledispatch
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from .visitor import visit
from ..vegaexpr import eval_vegajs


@visit.register
def visit_filter(transform: alt.FilterTransform, df: pd.DataFrame):
    mask = eval_predicate(transform['filter'], df).astype(bool)
    return df[mask]


def get_column(df: pd.DataFrame, predicate: Any):
    """Get the transformed column from the predicate."""
    if predicate.timeUnit is not alt.Undefined:
        raise NotImplementedError("timeUnit Trnasform in Predicates")
    return df[predicate.field]


@singledispatch
def eval_predicate(predicate: Any, df: pd.DataFrame):
    raise NotImplementedError(
        f"Evaluating predicate of type {type(predicate)}")


@eval_predicate.register
def eval_string(predicate: str, df: pd.DataFrame):
    return df.apply(lambda datum: eval_vegajs(predicate, datum), axis=1)


@eval_predicate.register
def eval_field_equal(predicate: alt.FieldEqualPredicate, df: pd.DataFrame):
    return get_column(df, predicate) == eval_value(predicate.equal)


@eval_predicate.register
def eval_field_range(predicate: alt.FieldRangePredicate, df: pd.DataFrame):
    min_, max_ = [eval_value(val) for val in predicate.range]
    column = get_column(df, predicate)
    if min_ is None:
        min_ = column.min()
    if max_ is None:
        max_ = column.max()
    return column.between(min_, max_, inclusive=True)


@eval_predicate.register
def eval_field_oneof(predicate: alt.FieldOneOfPredicate, df: pd.DataFrame):
    options = [eval_value(val) for val in predicate.oneOf]
    return get_column(df, predicate).isin(options)


@eval_predicate.register
def eval_field_lt(predicate: alt.FieldLTPredicate, df: pd.DataFrame):
    return get_column(df, predicate) < eval_value(predicate.lt)


@eval_predicate.register
def eval_field_lte(predicate: alt.FieldLTEPredicate, df: pd.DataFrame):
    return get_column(df, predicate) <= eval_value(predicate.lte)


@eval_predicate.register
def eval_field_gt(predicate: alt.FieldGTPredicate, df: pd.DataFrame):
    return get_column(df, predicate) > eval_value(predicate.gt)


@eval_predicate.register
def eval_field_gte(predicate: alt.FieldGTEPredicate, df: pd.DataFrame):
    return get_column(df, predicate) >= eval_value(predicate.gte)


@eval_predicate.register
def eval_logical_not(predicate: alt.LogicalNotPredicate, df: pd.DataFrame):
    return ~eval_predicate(predicate['not'], df)


@eval_predicate.register
def eval_logical_and(predicate: alt.LogicalAndPredicate, df: pd.DataFrame):
    return np.logical_and.reduce([eval_predicate(p, df)
                                  for p in predicate['and']])


@eval_predicate.register
def eval_logical_or(predicate: alt.LogicalOrPredicate, df: pd.DataFrame):
    return np.logical_or.reduce([eval_predicate(p, df)
                                 for p in predicate['or']])


@singledispatch
def eval_value(value: Any):
    return value


@eval_value.register
def eval_datetime(value: alt.DateTime):
    # TODO: implement datetime conversion & comparison
    raise NotImplementedError("Evaluating alt.DateTime object")
