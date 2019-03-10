import pytest

import numpy as np
import pandas as pd

from altair_transform import apply_transform
from altair_transform.core import AGG_REPLACEMENTS


AGGREGATES = ['argmax', 'argmin', 'average', 'count', 'distinct',
              'max', 'mean',  'median', 'min', 'missing', 'q1',
              'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp',
              'sum', 'valid', 'values', 'variance', 'variancep']

AGG_SKIP = ['ci0', 'ci1']


@pytest.fixture
def data():
    rand = np.random.RandomState(42)
    return pd.DataFrame({
        'x': rand.randint(0, 100, 10),
        'y': rand.randint(0, 100, 10),
        'c': list('AABBCCCDDD'),
    })


def test_calculate_transform(data):
    transform = {'calculate': 'datum.x + datum.y', 'as': 'z'}
    out1 = apply_transform(data, transform)

    out2 = data.copy()
    out2['z'] = data.x + data.y

    assert out1.equals(out2)


def test_filter_transform(data):
    transform = {'filter': 'datum.x < datum.y'}
    out1 = apply_transform(data, transform)

    out2 = data.copy()
    out2 = out2[data.x < data.y]

    assert out1.equals(out2)


@pytest.mark.parametrize('groupby', [True, False])
@pytest.mark.parametrize('op', set(AGGREGATES) - set(AGG_SKIP))
def test_aggregate_transform(data, groupby, op):
    field = 'x'
    col = 'z'
    group = 'c'

    transform = {'aggregate': [{'op': op, 'field': field, 'as': col}]}
    if groupby:
        transform['groupby'] = [group]

    op = AGG_REPLACEMENTS.get(op, op)
    out = apply_transform(data, transform)

    def validate(group):
        return group[field].aggregate(op) == group[col]

    if groupby:
        assert out.groupby(group).apply(validate).all()
    else:
        assert validate(out).all()


def test_multiple_transforms(data):
    transform = [
        {'calculate': '0.5 * (datum.x + datum.y)', 'as': 'xy_mean'},
        {'filter': 'datum.x < datum.xy_mean'}
    ]
    out1 = apply_transform(data, transform)
    out2 = data.copy()
    out2['xy_mean'] = 0.5 * (data.x + data.y)
    out2 = out2[out2.x < out2.xy_mean]

    assert out1.equals(out2)
