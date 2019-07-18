from typing import Any
import pandas as pd
import altair as alt

from .visitor import visit
from ..utils import to_dataframe

# These submodules register appropriate visitors.
from . import (aggregate, bin, calculate, filter, flatten, fold,  # noqa: F401
               impute, joinaggregate, lookup, sample, timeunit, window)

__all__ = ['apply', 'extract_data', 'process_chart']


def apply(df: pd.DataFrame,
          transform: Any,
          inplace: bool = False) -> pd.DataFrame:
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
    df_transformed : pd.DataFrame
        The transformed dataframe.
    """
    if not inplace:
        df = df.copy()
    if transform is alt.Undefined:
        return df
    return visit(transform, df)


def extract_data(chart: alt.Chart) -> pd.DataFrame:
    """Extract transformed data from a chart.

    This only works with data and transform defined at the
    top level of the chart.

    Parameters
    ----------
    chart : alt.Chart
        The chart instance from which the data and transform
        will be extracted

    Returns
    -------
    df_transformed : pd.DataFrame
        The extracted and transformed dataframe.
    """
    return apply(to_dataframe(chart.data, chart), chart.transform)


def transform_chart(chart: alt.Chart) -> alt.Chart:
    """Return a chart with the transformed data

    Parameters
    ----------
    chart : alt.Chart
        The chart instance from which the data and transform
        will be extracted.

    Returns
    -------
    chart_out : alt.Chart
        A copy of the input chart with the transformed data.
    """
    chart = chart.properties(data=extract_data(chart))
    chart.transform = alt.Undefined
    return chart
