from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 2], "y1": ["A", "B", "C"], "y2": ["D", "E", "F"]})


@pytest.mark.parametrize("as_", (None, ["name", "val"]))
def test_fold_transform(data, as_: Optional[List[str]]):
    if as_ is None:
        out = altair_transform.apply(data, {"fold": ["y1", "y2"]})
        as_ = ["key", "value"]
    else:
        out = altair_transform.apply(data, {"fold": ["y1", "y2"], "as": as_})

    expected = pd.DataFrame(
        {
            "x": np.repeat(data["x"], 2),
            as_[0]: 3 * ["y1", "y2"],
            as_[1]: np.ravel((data["y1"], data["y2"]), "F"),
            "y1": np.repeat(data["y1"], 2),
            "y2": np.repeat(data["y2"], 2),
        }
    ).reset_index(drop=True)
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("fold", [["y1"], ["y1", "y2"]])
@pytest.mark.parametrize("as_", [None, ["name", "val"]])
def test_fold_against_js(
    driver, data: pd.DataFrame, fold: List[str], as_: Optional[str]
) -> None:
    transform: Dict[str, Any] = {"fold": fold}
    if as_ is not None:
        transform["as"] = as_

    got = altair_transform.apply(data, transform)
    want = driver.apply(data, transform)

    assert_frame_equal(
        got[sorted(got.columns)],
        want[sorted(want.columns)],
        check_dtype=False,
        check_index_type=False,
        check_less_precise=True,
    )
