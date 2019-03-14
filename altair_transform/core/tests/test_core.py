import pytest

import numpy as np
import pandas as pd

from altair.utils.data import to_values
from altair_transform import apply
from altair_transform.core.aggregate import AGG_REPLACEMENTS


AGGREGATES = ['argmax', 'argmin', 'average', 'count', 'distinct',
              'max', 'mean',  'median', 'min', 'missing', 'q1',
              'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp',
              'sum', 'valid', 'values', 'variance', 'variancep']

AGG_SKIP = ['ci0', 'ci1']  # require scipy


@pytest.fixture
def data():
    rand = np.random.RandomState(42)
    return pd.DataFrame({
        'x': rand.randint(0, 100, 12),
        'y': rand.randint(0, 100, 12),
        'c': list('AAABBBCCCDDD'),
    })


def test_calculate_transform(data):
    transform = {'calculate': 'datum.x + datum.y', 'as': 'z'}
    out1 = apply(data, transform)

    out2 = data.copy()
    out2['z'] = data.x + data.y

    assert out1.equals(out2)


def test_filter_transform(data):
    transform = {'filter': 'datum.x < datum.y'}
    out1 = apply(data, transform)

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
    out = apply(data, transform)

    def validate(group):
        return np.allclose(group[field].aggregate(op), group[col])

    if groupby:
        assert out.groupby(group).apply(validate).all()
    else:
        assert validate(out)


@pytest.mark.parametrize('lookup_key', ['c', 'c2'])
def test_lookup_transform(data, lookup_key):
    lookup = pd.DataFrame({
        lookup_key: list('ABCD'),
        'z': [3, 1, 4, 5]
    })
    transform = {
        'lookup': 'c',
        'from': {
            'data': to_values(lookup),
            'key': lookup_key,
            'fields': ['z']
        }
    }
    out1 = apply(data, transform)
    out2 = pd.merge(data, lookup, left_on='c', right_on=lookup_key)
    if lookup_key != 'c':
        out2 = out2.drop(lookup_key, axis=1)
    assert out1.equals(out2)


@pytest.mark.parametrize('lookup_key', ['c', 'c2'])
@pytest.mark.parametrize('default', [None, 'missing'])
def test_lookup_transform_default(data, lookup_key, default):
    lookup = pd.DataFrame({
        lookup_key: list('ABC'),
        'z': [3, 1, 4]
    })
    transform = {
        'lookup': 'c',
        'from': {
            'data': to_values(lookup),
            'key': lookup_key,
            'fields': ['z']
        }
    }
    if default is not None:
        transform['default'] = default

    out = apply(data, transform)
    undef = (out['c'] == 'D')
    if default is None:
        assert out.loc[undef, 'z'].isnull().all()
    else:
        assert (out.loc[undef, 'z'] == default).all()


def test_bin_transform(data):
    transform = {'bin': True, 'field': 'x', 'as': 'xbin'}
    out = apply(data, transform)
    assert 'xbin' in out.columns

    transform = {'bin': True, 'field': 'x', 'as': ['xbin1', 'xbin2']}
    out = apply(data, transform)
    assert 'xbin1' in out.columns
    assert 'xbin2' in out.columns


def test_window_transform(data):
    transform = {
        'window': [{'op': 'sum', 'field': 'x', 'as': 'xsum'}],
        'groupby': ['y'],
    }
    out = apply(data, transform)
    expected = data.groupby('y').rolling(len(data), min_periods=1)
    expected = expected['x'].sum().reset_index('y', drop=True).sort_index()
    print(out['xsum'])
    print(expected)
    assert out['xsum'].equals(expected)


def test_multiple_transforms(data):
    transform = [
        {'calculate': '0.5 * (datum.x + datum.y)', 'as': 'xy_mean'},
        {'filter': 'datum.x < datum.xy_mean'}
    ]
    out1 = apply(data, transform)
    out2 = data.copy()
    out2['xy_mean'] = 0.5 * (data.x + data.y)
    out2 = out2[out2.x < out2.xy_mean]

    assert out1.equals(out2)
