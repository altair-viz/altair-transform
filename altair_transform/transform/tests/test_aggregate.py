from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import altair_transform
from altair_transform.transform.aggregate import AGG_REPLACEMENTS

AGGREGATES = [
    "argmax",
    "argmin",
    "average",
    "count",
    "distinct",
    "max",
    "mean",
    "median",
    "min",
    "missing",
    "q1",
    "q3",
    "ci0",
    "ci1",
    "stderr",
    "stdev",
    "stdevp",
    "sum",
    "valid",
    "values",
    "variance",
    "variancep",
]

AGG_SKIP = ["ci0", "ci1"]  # These require scipy.


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    return pd.DataFrame({"x": rand.randint(0, 100, 12), "c": list("AAABBBCCCDDD")})


@pytest.mark.parametrize("groupby", [True, False])
@pytest.mark.parametrize("op", set(AGGREGATES) - set(AGG_SKIP))
def test_aggregate_transform(data: pd.DataFrame, groupby: bool, op: Any):
    field = "x"
    col = "z"
    group = "c"

    transform: Dict[str, Any] = {"aggregate": [{"op": op, "field": field, "as": col}]}
    if groupby:
        transform["groupby"] = [group]

    if op == "argmin":

        def op(col, df=data):
            return df.loc[col.idxmin()].to_dict()

    elif op == "argmax":

        def op(col, df=data):
            return df.loc[col.idxmax()].to_dict()

    else:
        op = AGG_REPLACEMENTS.get(op, op)

    out = altair_transform.apply(data, transform)

    data = data.reset_index(drop=True)

    if op == "values":
        if groupby:
            grouped = data.groupby(group).apply(lambda x: x.to_dict(orient="records"))
            grouped.name = col
            grouped = grouped.reset_index()
        else:
            grouped = pd.DataFrame({col: [data.to_dict(orient="records")]})
    elif groupby:
        grouped = data.groupby(group)[field].aggregate(op)
        grouped.name = col
        grouped = grouped.reset_index()
    else:
        grouped = pd.DataFrame({col: [data[field].aggregate(op)]})

    assert_frame_equal(grouped, out)


@pytest.mark.parametrize("groupby", [None, ["c"]])
@pytest.mark.parametrize("op", set(AGGREGATES) - set(AGG_SKIP))
def test_aggregate_against_js(
    driver, data: pd.DataFrame, groupby: Optional[List[str]], op: str
) -> None:
    transform: Dict[str, Any] = {"aggregate": [{"op": op, "field": "x", "as": "z"}]}
    if groupby is not None:
        transform["groupby"] = groupby

    got = altair_transform.apply(data, transform)
    want = driver.apply(data, transform)

    print(data)
    print(got)
    print(want)

    assert_frame_equal(
        got[sorted(got.columns)],
        want[sorted(want.columns)],
        check_dtype=False,
        check_index_type=False,
        check_less_precise=True,
    )
