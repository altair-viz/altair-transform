import altair as alt
import pandas as pd
from .visitor import visit
from .aggregate import AGG_REPLACEMENTS


@visit.register
def visit_joinaggregate(transform: alt.JoinAggregateTransform,
                        df: pd.DataFrame) -> pd.DataFrame:
    groupby = transform['groupby']
    for aggregate in transform['joinaggregate']:
        op = aggregate['op'].to_dict()
        field = aggregate['field']
        col = aggregate['as']

        op = AGG_REPLACEMENTS.get(op, op)
        if field == "*" and field not in df.columns:
            field = df.columns[0]

        if groupby is alt.Undefined:
            df[col] = df[field].aggregate(op)
        else:
            result = df.groupby(groupby)[field].aggregate(op)
            result.name = col
            df = df.join(result, on=groupby)
    return df
