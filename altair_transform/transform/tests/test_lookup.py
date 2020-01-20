from typing import Any, Dict, Optional

from altair.utils.data import to_values
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    return pd.DataFrame({"x": rand.randint(0, 100, 12), "c": list("AAABBBCCCDDD")})


@pytest.fixture
def lookup_data() -> Dict[str, Any]:
    rand = np.random.RandomState(0)
    df = pd.DataFrame(
        {"y": rand.randint(0, 50, 4), "d": list("ABCD"), "e": list("ACDE")}
    )
    return to_values(df)


@pytest.mark.parametrize("lookup_key", ["c", "c2"])
def test_lookup_transform(data: pd.DataFrame, lookup_key: str) -> None:
    lookup = pd.DataFrame({lookup_key: list("ABCD"), "z": [3, 1, 4, 5]})
    transform = {
        "lookup": "c",
        "from": {"data": to_values(lookup), "key": lookup_key, "fields": ["z"]},
    }
    out1 = altair_transform.apply(data, transform)
    out2 = pd.merge(data, lookup, left_on="c", right_on=lookup_key)
    if lookup_key != "c":
        out2 = out2.drop(lookup_key, axis=1)
    assert_frame_equal(out1, out2)


@pytest.mark.parametrize("lookup_key", ["c", "c2"])
@pytest.mark.parametrize("default", [None, "missing"])
def test_lookup_transform_default(
    data: pd.DataFrame, lookup_key: str, default: Optional[str]
) -> None:
    lookup = pd.DataFrame({lookup_key: list("ABC"), "z": [3, 1, 4]})
    transform = {
        "lookup": "c",
        "from": {"data": to_values(lookup), "key": lookup_key, "fields": ["z"]},
    }
    if default is not None:
        transform["default"] = default

    out = altair_transform.apply(data, transform)
    undef = out["c"] == "D"
    if default is None:
        assert out.loc[undef, "z"].isnull().all()
    else:
        assert (out.loc[undef, "z"] == default).all()


@pytest.mark.parametrize("key", ["d", "e"])
@pytest.mark.parametrize("default", [None, "N/A"])
def test_lookup_against_js(
    driver,
    data: pd.DataFrame,
    lookup_data: Dict[str, Any],
    key: str,
    default: Optional[str],
) -> None:
    transform = {
        "lookup": "c",
        "from": {"data": lookup_data, "key": key, "fields": ["y"]},
    }
    if default is not None:
        transform["default"] = default
    got = altair_transform.apply(data, transform)
    want = driver.apply(data, transform)

    assert_frame_equal(
        got[sorted(got.columns)],
        want[sorted(want.columns)],
        check_dtype=False,
        check_index_type=False,
        check_less_precise=True,
    )
