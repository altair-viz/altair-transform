"""Implementation of the bin transform."""
import altair as alt
import pandas as pd
import numpy as np

from .visitor import visit
from .vega_utils import calculate_bins


def _cut(series: pd.Series, edges: np.ndarray, return_upper: bool = False):
    """Like pd.cut(), but include outliers in the outer bins."""
    bins = pd.cut(series, edges, labels=False)
    bins[series <= edges[0]] = 0
    bins[series >= edges[-1]] = len(edges) - 2
    bins = bins.astype(int)
    bins1 = pd.Series(edges[bins.values], index=bins.index)
    if return_upper:
        bins2 = pd.Series(edges[bins.values + 1], index=bins.index)
        return bins1, bins2
    else:
        return bins1


@visit.register(alt.BinTransform)
def visit_bin(transform: alt.BinTransform, df: pd.DataFrame) -> pd.DataFrame:
    transform_dct: dict = transform.to_dict()
    col = transform_dct["as"]
    bin = transform_dct["bin"]
    field = transform_dct["field"]
    extent = df[field].min(), df[field].max()

    bins = calculate_bins(extent, **({} if bin is True else bin))

    if isinstance(col, str):
        df[col] = _cut(df[field], bins, return_upper=False)
    else:
        df[col[0]], df[col[1]] = _cut(df[field], bins, return_upper=True)

    return df
