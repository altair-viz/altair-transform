from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(1)
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


@pytest.mark.parametrize("groupby", [None, ["c"]])
@pytest.mark.parametrize("sort", [None, "x"])
@pytest.mark.parametrize("frame", [None, [1, 1], [-2, 2], [None, None]])
def test_window_against_js(
    driver,
    data: pd.DataFrame,
    groupby: Optional[List[str]],
    sort: Optional[str],
    frame: Optional[List[Optional[int]]],
) -> None:
    transform: Dict[str, Any] = {
        "window": [{"op": "sum", "field": "x", "as": "xsum"}],
        "ignorePeers": False,
    }
    if groupby is not None:
        transform["groupby"] = groupby
    if sort is not None:
        transform["sort"] = [{"field": sort}]
    if frame is not None:
        transform["frame"] = frame
    got = altair_transform.apply(data, transform)
    want = driver.apply(data, transform)
    assert_frame_equal(
        got[sorted(got.columns)],
        want[sorted(want.columns)],
        check_dtype=False,
        check_index_type=False,
        check_less_precise=True,
    )
