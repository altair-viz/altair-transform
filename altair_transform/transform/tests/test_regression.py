from typing import List

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from altair_transform import apply


def test_linear() -> None:
    data = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [2, 4, 6, 8, 10]})
    transform = {"regression": "y", "on": "x"}
    out = apply(data, transform)
    assert_frame_equal(out, pd.DataFrame({"x": [0.0, 4.0], "y": [2.0, 10.0]}))


def test_linear_groupby() -> None:
    data = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 1, 2, 3],
            "y": [2, 4, 6, 8, 10, 2, 3, 4],
            "g": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    transform = {"regression": "y", "on": "x", "groupby": ["g"]}
    out = apply(data, transform)
    assert_frame_equal(
        out[out.g == 0], pd.DataFrame({"g": [0, 0], "x": [0.0, 4.0], "y": [2.0, 10.0]})
    )
    assert_frame_equal(
        out[out.g == 1], pd.DataFrame({"g": [1, 1], "x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


@pytest.mark.parametrize("groupby", [None, ["g"]])
def test_poly_vs_linear(groupby: List[str]) -> None:
    data = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 1, 2, 3],
            "y": [2, 4, 6, 8, 10, 2, 3, 4],
            "g": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    kwds = {} if not groupby else {"groupby": groupby}
    out1 = apply(data, {"regression": "y", "on": "x", **kwds})
    out2 = apply(
        data, {"regression": "y", "on": "x", "method": "poly", "order": 1, **kwds}
    )
    assert_frame_equal(out1, out2)
