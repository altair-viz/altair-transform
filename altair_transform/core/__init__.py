from typing import Any
import pandas as pd

from .visitor import visit

# These submodules register appropriate visitors.
from . import aggregate, calculate, filter, lookup  # noqa

__all__ = ['apply']


def apply(df: pd.DataFrame, transform: Any, inplace: bool = False):
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
