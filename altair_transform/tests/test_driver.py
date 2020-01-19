import altair as alt
import pandas as pd
from pandas.testing import assert_frame_equal

# Note: driver fixture here comes from conftest.py
# These tests will be skipped if selenium driver is not available.


def test_extract_data_source(driver):
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["A", "B", "C"]})
    chart = alt.Chart(df).mark_point()
    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = chart.to_dict()
    df_out = driver._extract_data(spec, "source_0")
    assert_frame_equal(df, df_out)


def test_driver_apply(driver):
    df = pd.DataFrame({"x": [1, 2, 3]})
    transform = {"calculate": "2 * datum.x", "as": "y"}
    df_out = driver.apply(df, transform)

    df["y"] = 2 * df["x"]
    assert_frame_equal(df, df_out)


def test_utc_tz_offset(driver):
    # Test that UTC dates have zero offset between Python and Javascript.
    assert driver.get_tz_offset("UTC") == pd.Timedelta(0)


def test_tz_code(driver):
    code = driver.get_tz_code()
    pd.to_datetime("2012-01-01").tz_localize(code)
