"""Implementation of the bin transform."""
from typing import Tuple

import altair as alt
import pandas as pd
import numpy as np

from .visitor import visit
from .vega_utils import calculate_bins


def _cut(series: pd.Series, edges: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    """Like pd.cut(), but include outliers in the outer bins."""
    bins = pd.cut(series, edges, labels=False, right=False)
    out_of_range = (series < edges[0]) | (series > edges[-1])
    bins[out_of_range] = -1
    bins = bins.astype(int)
    bins1 = pd.Series(edges[bins.values], index=bins.index, dtype=float)
    bins2 = pd.Series(edges[bins.values + 1], index=bins.index, dtype=float)
    bins1[out_of_range] = np.nan
    bins2[out_of_range] = np.nan
    return bins1, bins2


@visit.register(alt.BinTransform)
def visit_bin(transform: alt.BinTransform, df: pd.DataFrame) -> pd.DataFrame:
    transform_dct: dict = transform.to_dict()
    col = transform_dct["as"]
    bin_ = {} if transform_dct["bin"] is True else transform_dct["bin"]
    field = transform_dct["field"]

    bin_.setdefault("extent", [df[field].min(), df[field].max()])
    bins = calculate_bins(**bin_)

    if isinstance(col, str):
        df[col], df[col + "_end"] = _cut(df[field], bins)
    else:
        df[col[0]], df[col[1]] = _cut(df[field], bins)

    return df
