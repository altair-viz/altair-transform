import altair as alt
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from altair_transform.driver import _extract_data, apply

try:
    import altair_viewer  # noqa: F401
    import selenium  # noqa: F401
except (ImportError, ModuleNotFoundError):
    driver_available = False
else:
    driver_available = True


@pytest.mark.skipif(not driver_available, reason="Driver tools not available.")
def test_extract_data_source():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["A", "B", "C"]})
    chart = alt.Chart(df).mark_point()
    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = chart.to_dict()
    df_out = _extract_data(spec, "source_0")
    assert_frame_equal(df, df_out)


@pytest.mark.skipif(not driver_available, reason="Driver tools not available.")
def test_driver_apply():
    df = pd.DataFrame({"x": [1, 2, 3]})
    transform = {"calculate": "2 * datum.x", "as": "y"}
    df_out = apply(df, transform)

    df["y"] = 2 * df["x"]
    assert_frame_equal(df, df_out)
