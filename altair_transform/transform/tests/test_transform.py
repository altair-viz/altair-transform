import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

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

AGG_SKIP = ["ci0", "ci1", "values"]  # These require scipy.


@pytest.fixture
def data():
    rand = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "x": rand.randint(0, 100, 12),
            "y": rand.randint(0, 100, 12),
            "t": pd.date_range("2012-01-15", freq="M", periods=12),
            "i": range(12),
            "c": list("AAABBBCCCDDD"),
            "d": list("ABCABCABCABC"),
        }
    )


def test_calculate_transform(data):
    transform = {"calculate": "datum.x + datum.y", "as": "z"}
    out1 = altair_transform.apply(data, transform)

    out2 = data.copy()
    out2["z"] = data.x + data.y

    assert_frame_equal(out1, out2)


@pytest.mark.parametrize("groupby", [True, False])
@pytest.mark.parametrize("op", set(AGGREGATES) - set(AGG_SKIP))
def test_joinaggregate_transform(data, groupby, op):
    field = "x"
    col = "z"
    group = "c"

    transform = {"joinaggregate": [{"op": op, "field": field, "as": col}]}
    if groupby:
        transform["groupby"] = [group]

    op = AGG_REPLACEMENTS.get(op, op)
    out = altair_transform.apply(data, transform)

    def validate(group):
        return np.allclose(group[field].aggregate(op), group[col])

    if groupby:
        assert out.groupby(group).apply(validate).all()
    else:
        assert validate(out)


def test_quantile_values():
    np.random.seed(0)
    data = pd.DataFrame(
        {"x": np.random.randn(12), "C": np.random.choice(["A", "B"], 12)}
    )
    transform = {"quantile": "x", "groupby": ["C"], "as": ["p", "v"], "step": 0.1}
    # Copied from vega editor for above data/transform
    expected = pd.DataFrame(
        [
            ["A", 0.05, -0.853389779139604],
            ["A", 0.15, -0.6056135776659901],
            ["A", 0.25, -0.3578373761923762],
            ["A", 0.35, -0.12325942278589436],
            ["A", 0.45, 0.04532729028492671],
            ["A", 0.55, 0.21391400335574778],
            ["A", 0.65, 0.38250071642656897],
            ["A", 0.75, 0.7489619629456958],
            ["A", 0.85, 1.1549981161544833],
            ["A", 0.95, 1.5610342693632706],
            ["B", 0.05, -0.016677003759505288],
            ["B", 0.15, 0.15684925302119532],
            ["B", 0.25, 0.336128799065637],
            ["B", 0.35, 0.6476262524884882],
            ["B", 0.45, 0.9543858525126119],
            ["B", 0.55, 0.9744405491187167],
            ["B", 0.65, 1.2402825216772193],
            ["B", 0.75, 1.5575946277597235],
            ["B", 0.85, 1.8468937659906184],
            ["B", 0.95, 2.1102258760334363],
        ],
        columns=["C", "p", "v"],
    )
    out = altair_transform.apply(data, transform)
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("N", [1, 5, 50])
def test_sample_transform(data, N):
    transform = {"sample": N}
    out = altair_transform.apply(data, transform)

    # Ensure the shape is correct
    assert out.shape == (min(N, data.shape[0]), data.shape[1])

    # Ensure the content are correct
    assert_frame_equal(out, data.iloc[out.index])


def test_multiple_transforms(data):
    transform = [
        {"calculate": "0.5 * (datum.x + datum.y)", "as": "xy_mean"},
        {"filter": "datum.x < datum.xy_mean"},
    ]
    out1 = altair_transform.apply(data, transform)
    out2 = data.copy()
    out2["xy_mean"] = 0.5 * (data.x + data.y)
    out2 = out2[out2.x < out2.xy_mean].reset_index(drop=True)

    assert_frame_equal(out1, out2)
