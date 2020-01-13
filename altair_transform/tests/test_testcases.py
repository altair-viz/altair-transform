import json
import pkgutil

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from altair_transform import apply
from altair_transform.driver import _load


def iter_test_cases():
    raw = pkgutil.get_data("altair_transform", "tests/testcases.json")
    cases = json.loads(raw.decode())
    for case in cases:
        data = _load(case["data"])
        for t in case["transforms"]:
            yield (t["transform"], data, _load(t["out"]))


@pytest.mark.parametrize("transform, data, out", iter_test_cases())
def test_testcase(transform: dict, data: pd.DataFrame, out: pd.DataFrame) -> None:
    got = apply(data, transform)
    assert_frame_equal(got, out, check_dtype=False, check_index_type=False)
