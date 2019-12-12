import altair as alt
import numpy as np
import pandas as pd
from .visitor import visit


@visit.register(alt.QuantileTransform)
def visit_quantile(transform: alt.QuantileTransform, df: pd.DataFrame) -> pd.DataFrame:
    transform = transform.to_dict()
    quantile = transform["quantile"]
    groupby = transform.get("groupby")
    pname, vname = transform.get("as", ["prob", "value"])
    probs = transform.get("probs")
    if probs is None:
        step = transform.get("step", 0.01)
        probs = np.arange(0.5 * step, 1.0, step)

    def qq(s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({pname: probs, vname: np.quantile(s, probs)})

    if groupby:
        return df.groupby(groupby)[quantile].apply(qq).reset_index(groupby)

    else:
        return qq(df[quantile])
