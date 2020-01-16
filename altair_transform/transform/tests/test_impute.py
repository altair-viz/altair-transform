import altair as alt
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest

import altair_transform


@pytest.mark.parametrize("method", ["value", "mean", "median", "max", "min"])
def test_impute_transform_no_groupby(method: str) -> None:
    data = pd.DataFrame({"x": [1, 2], "y": [2, 3]})
    transform = alt.ImputeTransform(
        impute="y", key="x", keyvals={"start": 0, "stop": 5}, value=0, method=method
    )
    if method == "value":
        value = 0
    else:
        value = data.y.agg(method)
    imputed = altair_transform.apply(data, transform)

    assert_equal(imputed.x.values, range(5))
    assert_equal(imputed.y[[1, 2]].values, data.y.values)
    assert_equal(imputed.y[[0, 3, 4]].values, value)


def test_impute_transform_with_groupby() -> None:
    data = pd.DataFrame(
        {"x": [1, 2, 4, 1, 3, 4], "y": [1, 2, 4, 2, 4, 5], "cat": list("AAABBB")}
    )

    transform = alt.ImputeTransform(impute="y", key="x", method="max", groupby=["cat"])

    imputed = altair_transform.apply(data, transform)
    assert_equal(imputed.x.values, np.tile(range(1, 5), 2))
    assert_equal(imputed.y.values, [1, 2, 4, 4, 2, 5, 4, 5])
