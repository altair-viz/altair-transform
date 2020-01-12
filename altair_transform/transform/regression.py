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

    def extent(self, df: pd.DataFrame) -> List[float]:
        return self._extent or [df[self._on].min(), df[self._on].max()]

    @abc.abstractmethod
    def grid(self, df: pd.DataFrame) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def params(self, df: pd.DataFrame) -> pd.DataFrame:
        params, rsquare = self._fit(df)
        return pd.DataFrame({"coef": [list(params)], "rsquared": [rsquare]})


class LinearModel(Model):
    def grid(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self.extent(df), dtype=float)

    def _fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        x = df[self._on].values
        y = df[self._reg].values
        X = np.vstack([np.ones_like(x), x]).T
        theta = np.linalg.solve(X.T @ X, X.T @ y)
        SS_tot = ((y - y.mean()) ** 2).sum()
        SS_res = ((y - np.dot(X, theta)) ** 2).sum()
        rsquare = 1 - SS_res / SS_tot
        return theta, rsquare

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        theta = self._fit(df)[0]
        x = self.grid(df)
        X = np.vstack([np.ones_like(x), x])
        y = theta @ X
        on, reg = self._as
        return pd.DataFrame({on: x, reg: y})


class PolyModel(Model):
    def grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self.extent(df)
        if self._order == 1:
            size = 2
        elif self._order == 2:
            size = 50
        else:
            size = 100
        return np.linspace(extent[0], extent[1], size)

    def _fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        x = df[self._on].values
        y = df[self._reg].values
        X = x[:, None] ** np.arange(self._order + 1)
        theta = np.linalg.solve(X.T @ X, X.T @ y)
        SS_tot = ((y - y.mean()) ** 2).sum()
        SS_res = ((y - np.dot(X, theta)) ** 2).sum()
        rsquare = 1 - SS_res / SS_tot
        return theta, rsquare

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        theta = self._fit(df)[0]
        x = self.grid(df)
        X = x ** np.arange(self._order + 1)[:, None]
        y = theta @ X
        on, reg = self._as
        return pd.DataFrame({on: x, reg: y})
