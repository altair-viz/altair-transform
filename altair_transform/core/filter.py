import altair as alt
import pandas as pd
from .visitor import visit
from ..vegaexpr import eval_vegajs


@visit.register
def visit_filter(transform: alt.FilterTransform, df: pd.DataFrame):
    filt = transform['filter']
    if not isinstance(filt, str):
        raise NotImplementedError("non-string filter")
    mask = df.apply(lambda datum: eval_vegajs(filt, datum), axis=1)
    return df[mask.astype(bool)]
