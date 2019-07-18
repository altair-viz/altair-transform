"""Implementation of the bin transform."""
import altair as alt
import pandas as pd
import numpy as np

from .visitor import visit

from typing import Union, Dict


@visit.register
def visit_bin(transform: alt.BinTransform, df: pd.DataFrame) -> pd.DataFrame:
    transform = transform.to_dict()
    col = transform['as']
    bin = transform['bin']
    field = transform['field']

    bins = calc_bins(df[field].min(), df[field].max(), bin)

    if isinstance(col, str):
        df[col] = pd.cut(df[field], bins, labels=bins[:-1])
    else:
        df[col[0]] = pd.cut(df[field], bins, labels=bins[:-1])
        df[col[1]] = pd.cut(df[field], bins, labels=bins[1:])

    return df


def calc_bins(data_min: float, data_max: float,
              bin_params: Union[bool, Dict]) -> np.ndarray:
    params: dict = {}
    if isinstance(bin_params, dict):
        params = bin_params
    elif not bin_params:
        raise ValueError("bin=False not supported")

    extent = params.get('extent')
    step = params.get('step')
    steps = params.get('steps')
    # TODO: maxbins should default to 6 for row/col encodings.
    maxbins = params.get('maxbins', 10)
    nice = params.get('nice', True)
    if extent:
        data_min, data_max = extent

    # TODO: more logic around nice
    if step:
        bins = np.arange(data_min, data_max + step, step)
        if nice:
            bins -= bins[0] % step
    elif steps:
        raise NotImplementedError("BinParams(steps)")
    else:
        bins = np.linspace(data_min, data_max, maxbins + 1)

    # TODO: support anchor, base, binned, divide, minstep
    return bins
