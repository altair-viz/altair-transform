import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "x": rand.randint(0, 100, 12),
            "c": list("AAABBBCCCDDD"),
            "d": list("ABCABCABCABC"),
        }
    )


def test_pivot_transform(data: pd.DataFrame) -> None:
    transform = {"pivot": "c", "value": "x"}
    expected = pd.DataFrame(
        {key: [data.x[data.c == key].sum()] for key in data.c.unique()},
    )
    out = altair_transform.apply(data, transform)
    assert_frame_equal(out, expected)


def test_pivot_transform_groupby(data: pd.DataFrame) -> None:
    transform = {"pivot": "c", "value": "x", "groupby": ["d"]}
    expected = data.pivot(values="x", index="d", columns="c").reset_index()
    expected.columns.names = [None]
    out = altair_transform.apply(data, transform)
    assert_frame_equal(out, expected)


def test_pivot_transform_limit(data: pd.DataFrame) -> None:
    transform = {"pivot": "c", "value": "x", "limit": 2}
    expected = pd.DataFrame(
        {key: [data.x[data.c == key].sum()] for key in sorted(data.c.unique())[:2]}
    )
    out = altair_transform.apply(data, transform)
    assert_frame_equal(out, expected)
