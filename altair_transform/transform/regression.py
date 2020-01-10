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
    for key in ["method", "params"]:
        if key in transform:
            raise NotImplementedError(f"transform.{key}")

    model: Model
    if method == "linear":
        model = LinearModel(on=on, reg=reg, extent=extent, as_=as_, order=order)
    elif method == "poly":
        model = PolyModel(on=on, reg=reg, extent=extent, as_=as_, order=order)
    else:
        raise NotImplementedError(f"method={method}")

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
    def _fit(self, df: pd.DataFrame) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def params(self, df: pd.DataFrame) -> np.ndarray:
        return self._fit(df)


class LinearModel(Model):
    def _fit(self, df: pd.DataFrame):
        x = df[self._on].values
        y = df[self._reg].values
        X = np.vstack([np.ones_like(x), x]).T
        return np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        theta = self._fit(df)
        x = np.array(self.extent(df))
        X = np.vstack([np.ones_like(x), x])
        y = theta @ X
        on, reg = self._as
        return pd.DataFrame({on: x, reg: y})


class PolyModel(Model):
    def _fit(self, df: pd.DataFrame):
        x = df[self._on].values
        y = df[self._reg].values
        X = x[:, None] ** np.arange(self._order + 1)
        return np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        theta = self._fit(df)
        x = np.array(self.extent(df))
        X = x[:, None] ** np.arange(self._order + 1)
        y = theta @ X
        on, reg = self._as
        return pd.DataFrame({on: x, reg: y})
