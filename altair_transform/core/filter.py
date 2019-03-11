import altair as alt
import pandas as pd
from .visitor import visit
from ..vegaexpr import eval_vegajs


@visit.register
def _(transform: alt.FilterTransform, df: pd.DataFrame):
    if not isinstance(transform.filter, str):
        raise NotImplementedError("non-string filter")
    mask = df.apply(
        lambda datum: eval_vegajs(transform.filter, datum),
        axis=1).astype(bool)
    return df[mask]
