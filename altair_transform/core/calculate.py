import altair as alt
import pandas as pd
from .visitor import visit
from ..vegaexpr import eval_vegajs


@visit.register
def _(transform: alt.CalculateTransform, df: pd.DataFrame):
    col = transform['as']
    df[col] = df.apply(
        lambda datum: eval_vegajs(transform.calculate, datum),
        axis=1)
    return df
