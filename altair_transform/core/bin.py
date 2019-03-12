"""Implementation of the bin transform."""
import altair as alt
import pandas as pd
import numpy as np

from .visitor import visit

from typing import List, Union


def _get(obj, key, default=alt.Undefined):
    val = obj[key]
    if val is alt.Undefined:
        return default
    return val


@visit.register
def visit_bin(transform: alt.BinTransform, df: pd.DataFrame):
    as_: Union[List[str], str] = transform['as']
    bin: Union[bool, alt.BinParams] = transform.bin
    field: str = transform.field
    s = df[field]

    bins = calc_bins(s.min(), s.max(), bin)

    if isinstance(as_, str):
        df[as_] = pd.cut(df[field], bins, labels=bins[:-1])
    else:
        df[as_[0]] = pd.cut(df[field], bins, labels=bins[:-1])
        df[as_[1]] = pd.cut(df[field], bins, labels=bins[1:])

    return df


def calc_bins(data_min: float, data_max: float,
              bin_params: Union[bool, alt.BinParams]):
    if bin_params is False:
        raise ValueError("bin=False not supported")
    if bin_params is True:
        # Use the defaults.
        bin_params = alt.BinParams()

    extent = _get(bin_params, 'extent', None)
    step = _get(bin_params, 'step', None)
    steps = _get(bin_params, 'steps', None)
    maxbins = _get(bin_params, 'maxbins', 10)
    nice = _get(bin_params, 'nice', True)
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

    # TODO: anchor, base, divide, minstep
    return bins
