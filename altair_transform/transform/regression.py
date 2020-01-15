import abc
from typing import Callable, Dict, List, Optional, Tuple, Type

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
        extent: Optional[Tuple[float, float]],
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
        x, y = self._grid(df)
        on, reg = self._as
        return pd.DataFrame({on: x, reg: y})

    def _grid(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        def f(x: float) -> float:
            return self._predict(np.array([x]))[0]

        extent = self._extent_from_data(df)
        return _adaptive_sample(f, extent)

    def _extent_from_data(self, df: pd.DataFrame) -> Tuple[float, float]:
        xmin: float = df[self._on].min()
        xmax: float = df[self._on].max()
        return self._extent or (xmin, xmax)

    @abc.abstractmethod
    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def _params(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        ...


class ExpModel(Model):
    """y = a * e ^ (b * x)"""

    _model: Optional[Polynomial]

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

    def _grid(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        extent = self._extent_from_data(df)
        x = np.array(extent)
        return x, self._predict(np.array(extent))

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

    def _grid(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self._order == 1:
            extent = self._extent_from_data(df)
            x = np.array(extent)
            return x, self._predict(np.array(extent))
        else:
            return super()._grid(df)

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

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = Polynomial.fit(x, y, 2)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model(x)

    def _params(self):
        assert self._model is not None
        return _ensure_length(self._model.convert(domain=self._model.window).coef, 3)


# -----------------------------------------
# Adaptive sampling
# https://github.com/vega/vega/blob/e100874a432033b24ba687a4d2132610411da1b6/packages/vega-regression/src/Regression.js

# subdivide up to accuracy of 0.1 degrees
MIN_RADIANS = 0.1 * np.pi / 180


def _adaptive_sample(
    f: Callable[[float], float],
    extent: Tuple[float, float],
    min_steps: int = 25,
    max_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive sampling of a function.

    Code adapted from Javascript at
    https://github.com/vega/vega/blob/e100874a432033b24ba687a4d2132610411da1b6/packages/vega-regression/src/Regression.js
    """

    def point(x: float) -> Tuple[float, float]:
        return (x, f(x))

    min_x, max_x = extent
    span = max_x - min_x
    stop = span / max_steps
    prev: List[Tuple[float, float]] = [point(min_x)]
    next_: List[Tuple[float, float]] = []

    if min_steps == max_steps:
        # no adaptation, sample uniform grid directly and return
        for i in range(1, max_steps):
            prev.append(point(min_x + (i / min_steps) * span))
        x, y = np.array(prev).T
        return x, y

    # sample minimum points on uniform grid
    # then move on to perform adaptive refinement
    next_.append(point(max_x))
    for i in range(min_steps - 1, 0, -1):
        next_.append(point(min_x + (i / min_steps) * span))

    p0, p1 = prev[0], next_[-1]

    while next_:
        p1 = next_[-1]

        # midpoint for potential curve subdivision
        pm = point((p0[0] + p1[0]) / 2)

        if pm[0] - p0[0] >= stop and _angleDelta(p0, pm, p1) > MIN_RADIANS:
            # maximum resolution has not yet been met, and
            # subdivision midpoint sufficiently different from endpoint
            # save subdivision, push midpoint onto the visitation stack
            next_.append(pm)
        else:
            # subdivision midpoint sufficiently similar to endpoint
            # skip subdivision, store endpoint, move to next point on the stack
            p0 = p1
            prev.append(p1)
            next_.pop()
    x, y = np.array(prev).T
    return x, y


def _angleDelta(
    p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]
) -> float:
    a0 = np.arctan2(r[1] - p[1], r[0] - p[0])
    a1 = np.arctan2(q[1] - p[1], q[0] - p[0])
    return abs(a0 - a1)
