import abc
from typing import List, Optional, Tuple

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
    extent = transform.get("extent")
    method = transform.get("method", "linear")
    as_ = transform.get("as", (on, reg))
    groupby = transform.get("groupby")
    order = transform.get("order", 3)
    params = transform.get("params", False)

    model: Model
    if method == "linear":
        model = LinearModel(on=on, reg=reg, extent=extent, as_=as_, order=order)
    elif method == "poly":
        model = PolyModel(on=on, reg=reg, extent=extent, as_=as_, order=order)
    else:
        raise NotImplementedError(f"method={method}")

    if params:
        if groupby:
            params = df.groupby(groupby).apply(model.params)
            params["keys"] = [list(p)[:-1] for p in params.index]
            return params.reset_index(drop=True)
        else:
            return model.params(df)
    else:
        if groupby:
            return df.groupby(groupby).apply(model.predict).reset_index(groupby)
        else:
            return model.predict(df)


class Model(metaclass=abc.ABCMeta):
    def __init__(
        self,
        reg: str,
        on: str,
        extent: Optional[List[float]],
        as_: Tuple[str, str],
        order: int,
    ):
        self._reg = reg
        self._on = on
        self._extent = extent
        self._as = as_
        self._order = order

    @abc.abstractmethod
    def _design_matrix(self, x: np.array) -> np.array:
        ...

    @abc.abstractmethod
    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        ...

    def extent(self, df: pd.DataFrame) -> List[float]:
        return self._extent or [df[self._on].min(), df[self._on].max()]

    def params(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self._on].values
        y = df[self._reg].values
        X = self._design_matrix(x)
        theta = np.linalg.solve(X.T @ X, X.T @ y)
        SS_tot = ((y - y.mean()) ** 2).sum()
        SS_res = ((y - np.dot(X, theta)) ** 2).sum()
        rsquare = 1 - SS_res / SS_tot
        return pd.DataFrame({"coef": [list(theta)], "rsquared": [rsquare]})

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params(df)
        coef = p["coef"][0]
        x = self._grid(df)
        X = self._design_matrix(x)
        y = coef @ X.T
        on, reg = self._as
        return pd.DataFrame({on: x, reg: y})


class LinearModel(Model):
    def _design_matrix(self, x: np.array) -> np.array:
        return np.vstack([np.ones_like(x), x]).T

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self.extent(df), dtype=float)


class PolyModel(Model):
    def _design_matrix(self, x: np.array) -> np.array:
        return x[:, None] ** np.arange(self._order + 1)

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self.extent(df)
        if self._order == 1:
            size = 2
        elif self._order == 2:
            size = 50
        else:
            size = 100
        return np.linspace(extent[0], extent[1], size)
