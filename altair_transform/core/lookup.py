from typing import Union

import altair as alt
import pandas as pd
from .visitor import visit


def _load_data(data: dict):
    if 'values' in data:
        return pd.DataFrame(data['values'])
    elif 'url' in data:
        url = data['url']
        fmt = data.get('format', url.split('.')[-1])
        if fmt == 'csv':
            return pd.read_csv(url)
        elif fmt == 'json':
            return pd.read_csv(url)
        else:
            raise ValueError(f"Unknown format for UrlData: '{fmt}'")
    else:
        data = alt.Data.from_dict(data)
        raise NotImplementedError(
            f"Data of type {type(data)}.")


@visit.register
def visit_lookup(transform: alt.LookupTransform,
                 df: pd.DataFrame) -> pd.DataFrame:
    transform = transform.to_dict()
    lookup_data = transform['from']
    data = lookup_data['data']
    key = lookup_data['key']
    fields = lookup_data['fields']

    other_df = _load_data(data)
    if fields is alt.Undefined:
        fields = list(other_df.columns)

    cols_to_use = fields
    if key not in fields:
        cols_to_use = fields + [key]
    else:
        cols_to_use = fields
    other_df = other_df[cols_to_use]

    lookup = transform['lookup']
    default = transform.get('default')

    # TODO: use as_ if fields are not specified
    indicator: Union[str, bool]
    if default is None:
        indicator = False
    else:
        # TODO: make sure this doesn't conflict
        indicator = "__merge_indicator"

    # TODO: how to handle conficting fields?
    merged = pd.merge(df, other_df, left_on=lookup,
                      right_on=key, how='left',
                      indicator=indicator)

    if key != lookup and key not in fields:
        merged = merged.drop(key, axis=1)
    if indicator:
        merged.loc[merged[indicator] == "left_only", fields] = default
        merged = merged.drop(indicator, axis=1)
    return merged
