import json
import os
import sys
from typing import Iterator, Tuple

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)
from altair_transform.driver import apply, _serialize

CASES = [
    {
        "data": pd.DataFrame({"x": range(1, 6), "y": [1, 2, 3, 5, 8]}),
        "transforms": [
            {"regression": "y", "on": "x", "method": method, "params": True}
            for method in [
                "linear",
                "log",
                "exp",
                "pow",
                # rSquared is incorrect for quad and poly in vega 5.8
                # "quad",
                # "poly",
            ]
        ]
        + [
            {"regression": "y", "on": "x", "method": method, "params": False}
            for method in ["linear", "log", "exp", "pow", "quad", "poly"]
        ],
    },
]


def generate_test_cases() -> Iterator[Tuple[list, dict, list]]:
    cases = []
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
    filename = os.path.join(ROOT_DIR, "altair_transform", "tests", "testcases.json")
    print(f"writing {len(cases)} test cases to {filename}")
    with open(filename, "w") as f:
        json.dump(cases, f, indent=1)


if __name__ == "__main__":
    generate_test_cases()
