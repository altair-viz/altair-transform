import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    return pd.DataFrame({"x": rand.randint(0, 100, 12), "c": list("AAABBBCCCDDD")})


def test_window_transform_basic(data: pd.DataFrame) -> None:
    transform = {"window": [{"op": "sum", "field": "x", "as": "xsum"}]}
    out = altair_transform.apply(data, transform)
    expected = data["x"].cumsum()
    expected.name = "xsum"
    assert_series_equal(out["xsum"], expected.astype(float))


def test_window_transform_sorted(data: pd.DataFrame) -> None:
    transform = {
        "window": [{"op": "sum", "field": "x", "as": "xsum"}],
        "sort": [{"field": "x"}],
    }
    out = altair_transform.apply(data, transform)
    expected = data["x"].sort_values().cumsum().sort_index()
    expected.name = "xsum"
    assert_series_equal(out["xsum"], expected.astype(float))


def test_window_transform_grouped(data: pd.DataFrame) -> None:
    transform = {
        "window": [{"op": "sum", "field": "x", "as": "xsum"}],
        "groupby": ["c"],
    }
    out = altair_transform.apply(data, transform)
    expected = data.groupby("c").rolling(len(data), min_periods=1)
    expected = expected["x"].sum().reset_index("c", drop=True).sort_index()
    expected.name = "xsum"
    assert_series_equal(out["xsum"], expected)
