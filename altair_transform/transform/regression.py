import abc
from typing import Dict, List, Optional, Tuple, Type

import altair as alt
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from .visitor import visit


def _ensure_length(coef: np.ndarray, k: int) -> np.ndarray:
    return np.hstack([coef, np.zeros(k - len(coef), dtype=coef.dtype)])


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
        "exp": ExpModel,
        "linear": LinearModel,
        "log": LogModel,
        "poly": PolyModel,
        "pow": PowModel,
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
        SS_tot = ((y - y.mean()) ** 2).sum()
        SS_res = ((y - self._predict(x)) ** 2).sum()
        rsquare = 1 - SS_res / SS_tot
        return pd.DataFrame({"coef": [list(self._params())], "rSquared": [rsquare]})

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
    def _params(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        ...


# TODO: other models
# power (pow):


class ExpModel(Model):
    """y = a * e ^ (b * x)"""

    _model: Optional[Polynomial]

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self._extent_from_data(df)
        return np.linspace(extent[0], extent[1], 50)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = Polynomial.fit(x, np.log(y), 1, w=np.sqrt(abs(y)))

    def _predict(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return np.exp(self._model(x))

    def _params(self) -> np.ndarray:
        assert self._model is not None
        log_a, b = _ensure_length(
            self._model.convert(domain=self._model.window).coef, 2
        )
        return np.array([np.exp(log_a), b])


class LinearModel(Model):
    """y = a + b * x"""

    _model: Optional[Polynomial]

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self._extent_from_data(df))

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = Polynomial.fit(x, y, 1)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model(x)

    def _params(self):
        assert self._model is not None
        return _ensure_length(self._model.convert(domain=self._model.window).coef, 2)


class LogModel(Model):
    """y = a + b * log(x)"""

    _model: Optional[Polynomial]

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self._extent_from_data(df)
        return np.linspace(extent[0], extent[1], 50)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = Polynomial.fit(np.log(x), y, 1)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model(np.log(x))

    def _params(self) -> np.ndarray:
        assert self._model is not None
        return _ensure_length(self._model.convert(domain=self._model.window).coef, 2)


class PolyModel(Model):
    """y = a + b * x + ... + k * x^k"""

    _model: Optional[Polynomial]

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self._extent_from_data(df))

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = Polynomial.fit(x, y, self._order)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model(x)

    def _params(self):
        assert self._model is not None
        return _ensure_length(
            self._model.convert(domain=self._model.window).coef, self._order + 1
        )


class PowModel(Model):
    """y = a * x ^ b"""

    _model: Optional[Polynomial]

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: make this match grid used in vega.
        extent = self._extent_from_data(df)
        return np.linspace(extent[0], extent[1], 50)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = Polynomial.fit(np.log(x), np.log(y), 1)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return np.exp(self._model(np.log(x)))

    def _params(self) -> np.ndarray:
        assert self._model is not None
        log_a, b = _ensure_length(
            self._model.convert(domain=self._model.window).coef, 2
        )
        return np.array([np.exp(log_a), b])


class QuadModel(Model):
    """y = a + b * x + c * x^2"""

    _model: Optional[Polynomial]

    def _grid(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self._extent_from_data(df))

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = Polynomial.fit(x, y, 2)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model(x)

    def _params(self):
        assert self._model is not None
        return _ensure_length(self._model.convert(domain=self._model.window).coef, 3)
