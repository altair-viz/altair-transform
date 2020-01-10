import pandas as pd
from pandas.testing import assert_frame_equal
from altair_transform import apply


def test_regression_transform():
    data = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [2, 4, 6, 8, 10]})
    transform = {"regression": "y", "on": "x"}
    out = apply(data, transform)
    assert_frame_equal(out, pd.DataFrame({"x": [0, 4], "y": [2.0, 10.0]}))


def test_regression_transform_groupby():
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
        out[out.g == 0], pd.DataFrame({"g": [0, 0], "x": [0, 4], "y": [2.0, 10.0]})
    )
    assert_frame_equal(
        out[out.g == 1], pd.DataFrame({"g": [1, 1], "x": [1, 3], "y": [2.0, 4.0]})
    )
