from typing import Union

import altair as alt
import pandas as pd
from .visitor import visit


@visit.register
def visit_lookup(transform: alt.LookupTransform, df: pd.DataFrame):
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

    indicator: Union[str, bool]
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
