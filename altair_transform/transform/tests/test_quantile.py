import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    return pd.DataFrame({"x": rand.randint(0, 100, 12), "c": list("AAABBBCCCDDD")})


def test_quantile_transform(data: pd.DataFrame) -> None:
    transform = {"quantile": "x", "step": 0.1}
    out = altair_transform.apply(data, transform)
    assert list(out.columns) == ["prob", "value"]
    assert_allclose(out.prob, np.arange(0.05, 1, 0.1))
    assert_allclose(out.value, np.quantile(data.x, out.prob))


def test_quantile_transform_groupby(data: pd.DataFrame) -> None:
    group = "c"
    transform = {"quantile": "x", "step": 0.1, "groupby": [group]}
    out = altair_transform.apply(data, transform)
    assert list(out.columns) == ["c", "prob", "value"]

    for key in data[group].unique():
        out_group_1 = altair_transform.apply(data[data[group] == key], transform)
        out_group_2 = out[out[group] == key][out_group_1.columns].reset_index(drop=True)
        assert_frame_equal(out_group_1, out_group_2)
