import altair as alt
import pandas as pd
from .visitor import visit
from .aggregate import AGG_REPLACEMENTS


@visit.register(alt.PivotTransform)
def visit_pivot(transform: alt.PivotTransform, df: pd.DataFrame) -> pd.DataFrame:
    transform = transform.to_dict()
    pivot = transform["pivot"]
    limit = transform.get("limit")
    if limit:
        vals = sorted(df[pivot].unique())[:limit]
        df = df[df[pivot].isin(vals)]
    groupby = transform.get("groupby")
    agg = transform.get("op", "sum")
    agg = AGG_REPLACEMENTS.get(agg, agg)
    return df.pivot_table(
        columns=pivot, values=transform["value"], index=groupby, aggfunc=agg,
    ).reset_index(drop=not groupby)
