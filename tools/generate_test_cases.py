import json
import os
import sys
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)
from altair_transform.driver import apply, _serialize


def data():
    rand = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "x": rand.randint(0, 100, 12),
            "y": rand.randint(0, 100, 12),
            "t": pd.date_range("2012-01-15", freq="M", periods=12),
            "i": range(12),
            "c": list("AAABBBCCCDDD"),
            "d": list("ABCABCABCABC"),
        }
    )
    df["t"] = df["t"].apply(lambda x: x.isoformat())
    return df


REGRESSION_TESTS = [
    {"regression": "y", "on": "x", "method": method, "params": params}
    for method in ["linear", "log", "exp", "pow", "quad", "poly",]
    for params in [True, False]
]

BIN_TESTS = [
    {"bin": True, "field": "x", "as": "xbin"},
    {"bin": True, "field": "x", "as": ["xbin1", "xbin2"]},
    {"bin": {"maxbins": 20}, "field": "x", "as": "xbin"},
    {"bin": {"nice": False}, "field": "x", "as": "xbin"},
    {"bin": {"anchor": 3.5}, "field": "x", "as": "xbin"},
    {"bin": {"step": 20}, "field": "x", "as": "xbin"},
    {"bin": {"base": 2}, "field": "x", "as": "xbin"},
    {"bin": {"extent": [20, 80]}, "field": "x", "as": "xbin"},
]


CASES = [
    {"data": data(), "transforms": REGRESSION_TESTS + BIN_TESTS},
]


def generate_test_cases() -> Iterator[Tuple[list, dict, list]]:
    cases = []
    total = 0
    for case in CASES:
        data = case["data"]
        transforms = case["transforms"]
        cases.append(
            {
                "data": _serialize(data),
                "transforms": [
                    {"transform": t, "out": _serialize(apply(data, t))}
                    for t in transforms
                ],
            }
        )
        total += len(transforms)
    filename = os.path.join(ROOT_DIR, "altair_transform", "tests", "testcases.json")
    print(f"writing {total} test cases to {filename}")
    with open(filename, "w") as f:
        json.dump(cases, f, indent=1)


if __name__ == "__main__":
    generate_test_cases()
