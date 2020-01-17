from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "x": rand.randint(0, 100, 12),
            "y": rand.randint(0, 100, 12),
            "g": list(6 * "AB"),
        }
    )
    return df


def test_linear() -> None:
    data = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [2, 4, 6, 8, 10]})
    transform = {"regression": "y", "on": "x"}
    out = altair_transform.apply(data, transform)
    assert_frame_equal(
        out, pd.DataFrame({"x": [0.0, 4.0], "y": [2.0, 10.0]}), check_dtype=False
    )


def test_linear_groupby() -> None:
    data = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 1, 2, 3],
            "y": [2, 4, 6, 8, 10, 2, 3, 4],
            "g": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    transform = {"regression": "y", "on": "x", "groupby": ["g"]}
    out = altair_transform.apply(data, transform)
    assert_frame_equal(
        out[out.g == 0].reset_index(drop=True),
        pd.DataFrame({"g": [0, 0], "x": [0.0, 4.0], "y": [2.0, 10.0]}),
        check_dtype=False,
    )
    assert_frame_equal(
        out[out.g == 1].reset_index(drop=True),
        pd.DataFrame({"g": [1, 1], "x": [1.0, 3.0], "y": [2.0, 4.0]}),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "method,coef", [("linear", [1, 2]), ("quad", [1, 2, 0]), ("poly", [1, 2, 0, 0])]
)
def test_linear_params(method: str, coef: List[int]) -> None:
    data = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [1, 3, 5, 7, 9]})
    transform = {"regression": "y", "on": "x", "params": True, "method": method}
    out = altair_transform.apply(data, transform)
    assert_frame_equal(out, pd.DataFrame({"coef": [coef], "rSquared": [1.0]}))


@pytest.mark.parametrize("groupby", [None, ["g"]])
@pytest.mark.parametrize("method,order", [("linear", 1), ("quad", 2)])
def test_poly_vs_linear(groupby: List[str], method: str, order: int) -> None:
    data = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 1, 2, 3],
            "y": [2, 4, 6, 8, 10, 2, 3, 4],
            "g": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    kwds = {} if not groupby else {"groupby": groupby}
    out1 = altair_transform.apply(
        data, {"regression": "y", "on": "x", "method": method, **kwds}
    )
    out2 = altair_transform.apply(
        data, {"regression": "y", "on": "x", "method": "poly", "order": order, **kwds}
    )
    assert_frame_equal(out1, out2, check_dtype=False)


@pytest.mark.parametrize("method", ["linear", "log", "exp", "pow", "quad", "poly"])
@pytest.mark.parametrize("params", [True, False])
@pytest.mark.parametrize("groupby", [None, ["g"]])
def test_regression_against_js(
    driver, data: pd.DataFrame, method: str, params: str, groupby: Optional[List[str]],
) -> None:
    transform: Dict[str, Any] = {
        "regression": "y",
        "on": "x",
        "method": method,
        "params": params,
    }
    if groupby:
        transform["groupby"] = groupby
    got = altair_transform.apply(data, transform)
    want = driver.apply(data, transform)

    # Account for differences in handling of undefined between browsers.
    if params and not groupby and got.shape != want.shape:
        got["keys"] = [None]

    assert_frame_equal(
        got[sorted(got.columns)],
        want[sorted(want.columns)],
        check_dtype=False,
        check_index_type=False,
        check_less_precise=True,
    )
