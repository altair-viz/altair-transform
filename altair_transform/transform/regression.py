import abc
from typing import Dict, List, Optional, Tuple, Type

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

    models: Dict[str, Type[Model]] = {
        "linear": LinearModel,
        "log": LogModel,
        "poly": PolyModel,
        "quad": QuadModel,
    }

    if method not in models:
        raise NotImplementedError(f"method={method}")

    M = models[method]
    model = M(on=on, reg=reg, extent=extent, as_=as_, order=order)

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
    _coef: Optional[np.ndarray]

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

    def params(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with model parameters and r-square values.

        Parameters
        ----------
        df : pd.DataFrame
            The input data to which the model will be fit.

        Returns
        -------
        coef : pd.DataFrame
            DataFrame with model fit results.
        """
        x = df[self._on].values
        y = df[self._reg].values
        self._fit(x, y)
        assert self._coef is not None
        SS_tot = ((y - y.mean()) ** 2).sum()
        SS_res = ((y - self._predict(x)) ** 2).sum()
        rsquare = 1 - SS_res / SS_tot
        return pd.DataFrame({"coef": [list(self._coef)], "rSquared": [rsquare]})

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the fit model

        Parameters
        ----------
        df : pd.DataFrame
            The input data to which the model will be fit.

        Returns
        -------
        model : pd.DataFrame
            DataFrame with model fit results.
        """
        self._fit(df[self._on].values, df[self._reg].values)
        x = self._grid(df)
        y = self._predict(x)
        on, reg = self._as
        return pd.DataFrame({on: x, reg: y})

    def _extent_from_data(self, df: pd.DataFrame) -> List[float]:
        return self._extent or [df[self._on].min(), df[self._on].max()]

    @abc.abstractmethod
    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        ...


# TODO: other models
# exponential (exp): y = a + e ^ (b * x)
# power (pow): y = a * x ^ b


class LogModel(Model):
    """y = a + b * log(x)"""

    def _design_matrix(self, x: np.array) -> np.array:
        return np.vstack([np.ones_like(x), np.log(x)]).T

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self._extent_from_data(df)
        return np.linspace(extent[0], extent[1], 50)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        X = self._design_matrix(x)
        self._coef = np.linalg.solve(X.T @ X, X.T @ y)

    def _predict(self, x: np.ndarray) -> None:
        assert self._coef is not None
        X = self._design_matrix(x)
        return X @ self._coef


class LinearModel(Model):
    """y = a + b * x"""

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([np.ones_like(x), x]).T

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self._extent_from_data(df))

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        X = self._design_matrix(x)
        self._coef = np.linalg.solve(X.T @ X, X.T @ y)

    def _predict(self, x: np.ndarray) -> None:
        assert self._coef is not None
        X = self._design_matrix(x)
        return X @ self._coef


class PolyModel(Model):
    """y = a + b * x + ... + k * x^k"""

    _xmean: Optional[float]
    _ymean: Optional[float]

    def _design_matrix(self, x: np.array) -> np.array:
        assert self._xmean is not None
        x = x - self._xmean
        return x[:, None] ** np.arange(self._order + 1)

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self._extent_from_data(df)
        if self._order == 1:
            size = 2
        elif self._order == 2:
            size = 50
        else:
            size = 100
        return np.linspace(extent[0], extent[1], size)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._xmean = x.mean()
        self._ymean = y.mean()
        X = self._design_matrix(x)
        self._coef = np.linalg.solve(X.T @ X, X.T @ (y - self._ymean))

    def _predict(self, x: np.ndarray) -> None:
        assert self._coef is not None
        assert self._ymean is not None
        X = self._design_matrix(x)
        return self._ymean + X @ self._coef


class QuadModel(Model):
    """y = a + b * x + c * x^2"""

    _xmean: Optional[float]
    _ymean: Optional[float]

    def _design_matrix(self, x: np.array) -> np.array:
        assert self._xmean is not None
        x = x - self._xmean
        return np.vstack([np.ones_like(x), x, x * x]).T

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self._extent_from_data(df)
        return np.linspace(extent[0], extent[1], 50)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._xmean = x.mean()
        self._ymean = y.mean()
        X = self._design_matrix(x)
        self._coef = np.linalg.solve(X.T @ X, X.T @ (y - self._ymean))

    def _predict(self, x: np.ndarray) -> None:
        assert self._coef is not None
        assert self._ymean is not None
        X = self._design_matrix(x)
        return self._ymean + X @ self._coef
