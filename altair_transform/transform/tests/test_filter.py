from typing import Any, Callable, Dict, List, Tuple, Union

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
            "y": rand.randint(0, 100, 12),
            "i": range(12),
            "c": list("AAABBBCCCDDD"),
        }
    )


FILTER_PREDICATES: List[
    Tuple[Union[str, Dict[str, Any]], Callable[[pd.DataFrame], pd.DataFrame]]
] = [
    ("datum.x < datum.y", lambda df: df[df.x < df.y]),
    ({"not": "datum.i < 5"}, lambda df: df[~(df.i < 5)]),
    (
        {"and": [{"field": "x", "lt": 50}, {"field": "i", "gte": 2}]},
        lambda df: df[(df.x < 50) & (df.i >= 2)],
    ),
    (
        {"or": [{"field": "y", "gt": 50}, {"field": "i", "lte": 4}]},
        lambda df: df[(df.y > 50) | (df.i <= 4)],
    ),
    ({"field": "c", "oneOf": ["A", "B"]}, lambda df: df[df.c.isin(["A", "B"])]),
    ({"field": "x", "range": [30, 60]}, lambda df: df[(df.x >= 30) & (df.x <= 60)]),
    ({"field": "c", "equal": "B"}, lambda df: df[df.c == "B"]),
]


@pytest.mark.parametrize("filter,calc", FILTER_PREDICATES)
def test_filter_transform(
    data: pd.DataFrame,
    filter: Union[str, Dict[str, Any]],
    calc: Callable[[pd.DataFrame], pd.DataFrame],
):
    out1 = altair_transform.apply(data, {"filter": filter})
    out2 = calc(data)
    assert_frame_equal(out1, out2)
