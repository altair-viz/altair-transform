import altair as alt
import numpy as np
import pandas as pd
from .visitor import visit


@visit.register(alt.RegressionTransform)
def visit_regression(
    transform: alt.RegressionTransform, df: pd.DataFrame
) -> pd.DataFrame:
    transform = transform.to_dict()
    reg = transform["regression"]
    on = transform["on"]
    extent = transform.get("extent", (df[on].min(), df[on].max()))
    as_ = transform.get("as", (on, reg))
    for key in ["groupby", "method", "order", "params"]:
        if key in transform:
            raise NotImplementedError(f"transform.{key}")

    # linear
    on_fit = list(extent)
    Xfit = np.vstack([np.ones_like(on_fit), on_fit]).T
    X = np.vstack([np.ones_like(df[on].values), df[on].values]).T
    theta = np.linalg.solve(X.T @ X, X.T @ df[reg].values)
    reg_fit = np.dot(Xfit, theta).flatten()

    return pd.DataFrame({as_[0]: on_fit, as_[1]: reg_fit})
