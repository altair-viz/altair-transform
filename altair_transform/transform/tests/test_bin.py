from typing import List

import numpy as np
import pandas as pd
import pytest

import altair_transform


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    return pd.DataFrame({"x": rand.randint(0, 100, 12)})


def test_bin_transform_simple(data: pd.DataFrame) -> None:
    transform = {"bin": True, "field": "x", "as": "xbin"}
    out = altair_transform.apply(data, transform)
    assert "xbin" in out.columns

    transform = {"bin": True, "field": "x", "as": ["xbin1", "xbin2"]}
    out = altair_transform.apply(data, transform)
    assert "xbin1" in out.columns
    assert "xbin2" in out.columns


@pytest.mark.parametrize("maxbins", [5, 10, 20])
@pytest.mark.parametrize("nice", [True, False])
def test_bin_transform_maxbins(nice: bool, maxbins: int) -> None:
    data = pd.DataFrame({"x": np.arange(100)})
    transform = {"bin": {"maxbins": maxbins, "nice": nice}, "field": "x", "as": "xbin"}
    out = altair_transform.apply(data, transform)
    assert "xbin" in out.columns
    assert "xbin_end" in out.columns
    bins = np.sort(out["xbin"].unique())
    assert len(bins) - 1 <= maxbins
    assert not out.xbin.isnull().any()


@pytest.mark.parametrize("step", [5, 10, 20])
@pytest.mark.parametrize("nice", [True, False])
def test_bin_transform_step(nice: bool, step: int) -> None:
    data = pd.DataFrame({"x": np.arange(100)})
    transform = {"bin": {"step": step, "nice": nice}, "field": "x", "as": "xbin"}
    out = altair_transform.apply(data, transform)
    bins = np.sort(out.xbin.unique())
    assert np.allclose(bins[1:] - bins[:-1], step)
    assert not out.xbin.isnull().any()


@pytest.mark.parametrize("nice", [True, False])
def test_bin_transform_steps(nice: bool, steps: List[int] = [5, 10, 20]) -> None:
    data = pd.DataFrame({"x": range(100)})
    transform = {"bin": {"steps": steps, "nice": nice}, "field": "x", "as": "xbin"}
    out = altair_transform.apply(data, transform)
    bins = np.sort(out.xbin.unique())
    assert bins[1] - bins[0] in steps
    assert not out.xbin.isnull().any()
