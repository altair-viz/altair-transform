import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import altair_transform

TIMEUNITS = [
    "year",
    "quarter",
    "month",
    "day",
    "date",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "yearquarter",
    "yearquartermonth",
    "yearmonth",
    "yearmonthdate",
    "yearmonthdatehours",
    "yearmonthdatehoursminutes",
    "yearmonthdatehoursminutesseconds",
    "quartermonth",
    "monthdate",
    "hoursminutes",
    "hoursminutesseconds",
    "minutesseconds",
    "secondsmilliseconds",
]


@pytest.fixture
def timezone(driver) -> str:
    return driver.get_tz_code()


@pytest.fixture
def data() -> pd.DataFrame:
    rand = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "t": (
                pd.to_datetime("2020-01-01")
                + pd.to_timedelta(rand.randint(0, 60_000_000, 50), unit="s")
            ).sort_values()
        }
    )


@pytest.mark.parametrize(
    "timeUnit,fmt",
    [
        ("year", "%Y"),
        ("yearmonth", "%Y-%m"),
        ("yearmonthdate", "%Y-%m-%d"),
        ("monthdate", "2012-%m-%d"),
        ("date", "2012-01-%d"),
    ],
)
def test_timeunit_transform(data: pd.DataFrame, timeUnit: str, fmt: str) -> None:
    transform = {"timeUnit": timeUnit, "field": "t", "as": "unit"}
    out = altair_transform.apply(data, transform)
    unit = pd.to_datetime(data.t.dt.strftime(fmt))
    assert (out.unit == unit).all()


@pytest.mark.parametrize("timeUnit", TIMEUNITS)
def test_timeunit_against_js(
    driver, data: pd.DataFrame, timezone: str, timeUnit: str
) -> None:
    transform = {"timeUnit": timeUnit, "field": "t", "as": "unit"}

    got = altair_transform.apply(data, transform)

    data["t"] = data["t"].apply(lambda x: x.isoformat())
    want = driver.apply(data, transform)

    want["t"] = (
        pd.to_datetime(1e6 * want["t"])
        .dt.tz_localize("UTC")
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)
    )
    want["unit"] = (
        pd.to_datetime(want["unit"]).dt.tz_convert(timezone).dt.tz_localize(None)
    )

    cols = ["t", "unit"]
    print(want[cols])
    print(got[cols])
    print(want[cols] - got[cols])

    assert_frame_equal(want[cols], got[cols])

    # want["t"] = pd.to_datetime(want["t"])
    # want["unit"] = pd.to_datetime(want["unit"])
    # want["unit_end"] = pd.to_datetime(want["unit_end"])

    # assert_frame_equal(
    #     got[sorted(got.columns)],
    #     want[sorted(want.columns)],
    #     check_dtype=False,
    #     check_index_type=False,
    #     check_less_precise=True,
    # )
