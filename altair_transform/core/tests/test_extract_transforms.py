from altair_transform.core.extract_transforms import _encoding_to_transform


def test_extract_simple_aggregate():
    encoding = {
        "x": {"aggregate": "sum", "field": "people", "type": "quantitative"},
        "y": {"field": "age", "type": "ordinal"},
    }

    expected_encoding = {
        "x": {"field": "sum_people", "type": "quantitative"},
        "y": {"field": "age", "type": "ordinal"},
    }

    expected_transform = [
        {
            "aggregate": [{"op": "sum", "field": "people", "as": "sum_people"}],
            "groupby": ["y"],
        }
    ]

    got_encoding, got_transform = _encoding_to_transform(encoding)
    assert got_encoding == expected_encoding
    assert got_transform == expected_transform
