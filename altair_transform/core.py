from functools import singledispatch

import pandas as pd

import altair as alt
from altair_transform.vegaexpr import eval_vegajs

__all__ = ['apply_transform']


def apply_transform(df, transform, inplace=False):
    """Apply transform or transforms to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    transform : list|dict
        A transform specification or list of transform specifications.
        Each specification must be valid according to Altair's transform
        schema.
    inplace : bool
        If True, then dataframe may be modified in-place. Default: False.

    Returns
    -------
    df_transform : pd.DataFrame
        The transformed dataframe.
    """
    if not inplace:
        df = df.copy()
    return visit(transform, df)


@singledispatch
def visit(transform, df):
    raise NotImplementedError("transform of type {0}".format(type(transform)))


@visit.register(list)
def visit_list(transform, df):
    for t in transform:
        df = visit(t, df)
    return df


@visit.register(dict)
def visit_dict(transform, df):
    transform = alt.Transform.from_dict(transform)
    return visit(transform, df)


@visit.register(alt.CalculateTransform)
def _3(transform, df):
    col = transform['as']
    df[col] = df.apply(
        lambda datum: eval_vegajs(transform.calculate, datum),
        axis=1)
    return df


@visit.register(alt.FilterTransform)
def visit_filter(transform, df):
    if not isinstance(transform.filter, str):
        raise NotImplementedError("non-string filter")
    mask = df.apply(
        lambda datum: eval_vegajs(transform.filter, datum),
        axis=1).astype(bool)
    return df[mask]


@visit.register(alt.AggregateTransform)
def visit_agg(transform, df):
    groupby = transform['groupby']
    for aggregate in transform['aggregate']:
        op = aggregate['op'].to_dict()
        field = aggregate['field']
        col = aggregate['as']

        op = AGG_REPLACEMENTS.get(op, op)

        if groupby is alt.Undefined:
            df[col] = df[field].aggregate(op)
        else:
            result = df.groupby(groupby)[field].aggregate(op)
            result.name = col
            df = df.join(result, on=groupby)
    return df


@visit.register(alt.LookupTransform)
def visit_lookup(transform, df):
    lookup_data = transform['from']
    data = lookup_data.data
    key = lookup_data.key
    # TODO: handle null fields
    fields = lookup_data.fields

    if not isinstance(data, alt.InlineData):
        raise NotImplementedError(f"Lookup data of type {type(data)}")
    other_df = pd.DataFrame(data.values)
    other_df = other_df[[key] + fields]

    lookup = transform.lookup
    default = transform.default
    # TODO: use as_ if fields are not specified
    # as_ = transform['as']

    if default is not alt.Undefined:
        # TODO: make sure this doesn't conflict
        indicator = "__merge_indicator"
    else:
        indicator = False

    merged = pd.merge(df, other_df, left_on=lookup,
                      right_on=key, how='left',
                      indicator=indicator)
    # TODO: don't drop if in fields
    if key != lookup:
        merged = merged.drop(key, axis=1)
    if indicator:
        merged.loc[indicator == "left_only", fields] = default
        merged = merged.drop(indicator, axis=1)
    return merged


def confidence_interval(x, level):
    from scipy import stats
    return stats.t.interval(level, len(x)-1, loc=x.mean(), scale=x.sem())


AGG_REPLACEMENTS = {
    'argmin': 'idxmin',
    'argmax': 'idxmax',
    'average': 'mean',
    'ci0': lambda x: confidence_interval(x, 0.05),
    'ci1': lambda x: confidence_interval(x, 0.95),
    'distinct': 'nunique',
    'stderr': 'sem',
    'stdev': 'std',
    'stdevp': lambda x: x.std(ddof=0),
    'missing': lambda x: x.isnull().sum(),
    'q1': lambda x: x.quantile(0.25),
    'q3': lambda x: x.quantile(0.75),
    'valid': 'count',
    'values': 'count',
    'variance': 'var',
    'variancep': lambda x: x.var(ddof=0)
}
