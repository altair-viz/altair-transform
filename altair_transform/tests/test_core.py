import pytest

import numpy as np
import pandas as pd

from altair_transform import apply_transform


@pytest.fixture
def data():
    rand = np.random.RandomState(42)
    return pd.DataFrame({
        'x': rand.randint(0, 100, 10),
        'y': rand.randint(0, 100, 10)
    })


def test_calculate_transform(data):
    transform = [{'calculate': 'datum.x + datum.y', 'as': 'z'}]
    out1 = apply_transform(data, transform)

    out2 = data.copy()
    out2['z'] = data.x + data.y

    assert out1.equals(out2)


def test_filter_transform(data):
    transform = [{'filter': 'datum.x < datum.y'}]
    out1 = apply_transform(data, transform)

    out2 = data.copy()
    out2 = out2[data.x < data.y]

    assert out1.equals(out2)
