from typing import Dict, List

import numpy as np
from numpy.testing import assert_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [[1, 2, 3], [4, 5, 6, 7], [8, 9]],
            "y": [[1, 2], [3, 4], [5, 6]],
            "cat": list("ABC"),
        }
    )


def test_flatten_transform(data: pd.DataFrame) -> None:
    out = altair_transform.apply(data, {"flatten": ["x"]})
    assert out.shape == (9, 3)
    assert out.columns.tolist() == ["x", "y", "cat"]
    assert_equal(out.x.values, range(1, 10))
    assert_equal(out.cat.values, list("AAABBBBCC"))

    out = altair_transform.apply(data, {"flatten": ["x", "y"]})
    assert out.shape == (9, 3)
    assert out.columns.tolist() == ["x", "y", "cat"]
    assert_equal(out.x.values, range(1, 10))
    assert_equal(out.y.values, [1, 2, np.nan, 3, 4, np.nan, np.nan, 5, 6])
    assert_equal(out.cat.values, list("AAABBBBCC"))


def test_flatten_transform_with_as(data: pd.DataFrame):
    out = altair_transform.apply(data, {"flatten": ["y"], "as": ["yflat"]})
    assert out.shape == (6, 4)
    assert out.columns.tolist() == ["yflat", "x", "y", "cat"]
    assert_equal(out.yflat.values, range(1, 7))
    assert_equal(out.cat.values, list("AABBCC"))

    out = altair_transform.apply(
        data, {"flatten": ["x", "y"], "as": ["xflat", "yflat"]}
    )
    assert out.shape == (9, 5)
    assert out.columns.tolist() == ["xflat", "yflat", "x", "y", "cat"]
    assert_equal(out.xflat.values, range(1, 10))
    assert_equal(out.yflat.values, [1, 2, np.nan, 3, 4, np.nan, np.nan, 5, 6])
    assert_equal(out.cat.values, list("AAABBBBCC"))


@pytest.mark.parametrize(
    "transform",
    [
        {"flatten": ["x"]},
        {"flatten": ["x"], "as": ["xflat"]},
        {"flatten": ["x", "y"]},
        {"flatten": ["x", "y"], "as": ["xflat"]},
        {"flatten": ["x", "y"], "as": ["xflat", "yflat"]},
    ],
)
def test_flatten_against_js(
    driver, data: pd.DataFrame, transform: Dict[str, List[str]],
) -> None:
    got = altair_transform.apply(data, transform)
    want = driver.apply(data, transform)

    print(got)
    print(want)

    assert_frame_equal(
        got[sorted(got.columns)],
        want[sorted(want.columns)],
        check_dtype=False,
        check_index_type=False,
        check_less_precise=True,
    )
